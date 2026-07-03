"""r1 risk branch: StyleGAN generative-prior gradient inversion.

Two complementary uses of a pretrained CelebA-HQ StyleGAN2 face prior:

  (A) latent gradient-matching  -- optimise latent codes W/W+ so the victim
      model's gradient on G(w) matches the observed gradient (cosine + L2 + TV +
      latent prior + diversity), over many restarts. Constraining the search to
      the face manifold is what pixel-space batch-128 matching lacked (v3's
      cosine-matched images collapsed to noise). Best for the models with no
      analytic leak / only a shared face-bank fallback (model2, model9, model11).

  (B) manifold projection (refine) -- project existing blurry/low-res
      reconstructions onto the StyleGAN manifold (image-space GAN inversion),
      sharpening them into on-manifold faces. Per-image, label-free, low-risk;
      good for the blurry strided-CNN models (model2/3/7).

Outputs go to a FRESH tree recon_work/runs/r1/<model>/ -- never touching v1/v2/v3
(candidates*/ outputs*/). Candidate schema matches the pipeline:
torch.save({"images": (128,3,64,64) float32 in [0,1], ...}).

GPU-heavy (double-backward through StyleGAN synthesis) -> run via sbatch.
Env: FL_DEVICE, FL_WORK_ROOT, FL_SG_REPO, FL_SG_PKL.
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

import utils as U
import attack_opt as AO       # build_any, real_grad_vector, compute_grad_vector
import infer_labels as IL
import gan_prior as GP

N_CLASSES = 200


# ---------------------------------------------------------------------------
# Victim setup
# ---------------------------------------------------------------------------
def setup_victim(name, device):
    model, grad = AO.build_any(name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    name2param = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in name2param]
    real = AO.real_grad_vector(grad, keys).to(device)
    real_unit = real / (real.norm() + 1e-12)
    return model, grad, keys, name2param, real, real_unit


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def fake_grad_vec(model, imgs, keys, name2param, hard_labels=None, label_logits=None):
    """Gradient of CE(logits, labels) w.r.t. the selected victim params, with
    create_graph=True so it stays differentiable back to the latents. Labels are
    either hard (LongTensor) or soft (learnable logits)."""
    logits = model(imgs)
    if label_logits is not None:
        logp = F.log_softmax(logits, dim=1)
        q = F.softmax(label_logits, dim=1)
        loss = -(q * logp).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(logits, hard_labels)
    params = [name2param[k] for k in keys]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.cat([g.reshape(-1) for g in grads])


def diversity_loss(w):
    """Mean positive pairwise cosine similarity of centred latents (minimise to
    push the k latents apart). Returns 0 for k<2."""
    k = w.shape[0]
    if k < 2:
        return w.new_zeros(())
    f = w.reshape(k, -1)
    f = f - f.mean(dim=0, keepdim=True)
    f = F.normalize(f, dim=1)
    sim = f @ f.t()
    off = sim - torch.eye(k, device=w.device)
    return off.clamp(min=0.0).sum() / (k * (k - 1))


@torch.no_grad()
def clamp_trunc(w, prior, radius):
    """Project each latent back into an L2 ball of `radius` around w_avg. Keeps
    the optimisation on the natural-face manifold (prevents degenerate,
    gradient-cosine-gaming outputs). radius<=0 disables."""
    if radius is None or radius <= 0:
        return
    wa = prior.w_avg.to(w.device)
    wa = wa[None, None, :] if prior.w_plus else wa[None, :]
    d = w.data - wa
    flat = d.reshape(d.shape[0], -1)
    n = flat.norm(dim=1, keepdim=True)
    scale = (radius / (n + 1e-8)).clamp(max=1.0)
    w.data = wa + (flat * scale).reshape_as(d)


# ---------------------------------------------------------------------------
# (A) Latent gradient-matching
# ---------------------------------------------------------------------------
def latent_gradient_match(name, prior, device, k=8, restarts=32, steps=400,
                          lr=0.05, l_cos=1.0, l_l2=0.1, l_tv=1e-4, l_w=1e-4,
                          l_div=1e-2, jitter=0.05, soft_labels=True, seed=0,
                          log_every=100, trunc=0.0):
    model, grad, keys, name2param, real, real_unit = setup_victim(name, device)
    assigned = IL.assign_labels(name, U.N_IMAGES).to(device)  # (128,)
    radius = trunc
    if trunc is not None and trunc < 0:            # auto-calibrate from W stats
        rw = prior.w_radius(device, pct=0.9)
        radius = rw * (prior.num_ws ** 0.5) if prior.w_plus else rw
        print(f"[{name}] auto trunc radius={radius:.2f} (w_p90={rw:.2f}, "
              f"w_plus={prior.w_plus})")
    print(f"[{name}] latent-match: family={grad['family']} keys={len(keys)} "
          f"k={k} restarts={restarts} steps={steps} soft_labels={soft_labels} "
          f"trunc_radius={radius:.2f}")
    pool_imgs, pool_q, pool_w = [], [], []
    t0 = time.time()
    for r in range(restarts):
        torch.manual_seed(seed + 7 * r + 1)
        from_mean = (r % 4 != 0)   # 3/4 near-mean, 1/4 random-z for diversity
        w = prior.init_w(k, device, jitter=jitter, from_mean=from_mean)
        idx = (torch.arange(k, device=device) + r * k) % U.N_IMAGES
        base_cls = assigned[idx]
        if soft_labels:
            label_logits = torch.full((k, N_CLASSES), -4.0, device=device)
            label_logits[torch.arange(k, device=device), base_cls] = 4.0
            label_logits.requires_grad_(True)
            opt = torch.optim.Adam([w, label_logits], lr=lr)
            vars_ = [w, label_logits]
        else:
            label_logits = None
            opt = torch.optim.Adam([w], lr=lr)
            vars_ = [w]
        best_cos, best_img = -1.0, None
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            imgs = prior.synth(w)
            fake = fake_grad_vec(model, imgs, keys, name2param,
                                 hard_labels=(None if soft_labels else base_cls),
                                 label_logits=label_logits)
            fake_unit = fake / (fake.norm() + 1e-12)
            cos = (fake_unit * real_unit).sum()
            l2 = F.mse_loss(fake, real)
            tv = U.total_variation(imgs).mean()
            wpri = prior.w_prior_loss(w)
            div = diversity_loss(w)
            total = (l_cos * (1.0 - cos) + l_l2 * l2 + l_tv * tv
                     + l_w * wpri + l_div * div)
            gvars = torch.autograd.grad(total, vars_)
            for v, g in zip(vars_, gvars):
                v.grad = g
            opt.step()
            clamp_trunc(w, prior, radius)          # keep latents on-manifold
            c = cos.item()
            if c > best_cos:
                best_cos, best_img = c, imgs.detach().float().cpu()
            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  {name} r{r:02d} it{it:4d} cos={c:.4f} "
                      f"l2={l2.item():.2e} tv={tv.item():.3f}", flush=True)
        pool_imgs.append(best_img)
        pool_q.append(torch.full((k,), best_cos))
        pool_w.append(w.detach().float().cpu())
        print(f"  [{name}] restart {r:02d}/{restarts} best_cos={best_cos:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    pool = torch.cat(pool_imgs, 0)
    quality = torch.cat(pool_q, 0)
    return pool, quality, (model, grad, keys, name2param, real_unit, assigned)


# ---------------------------------------------------------------------------
# (B) Manifold projection (refine existing reconstructions)
# ---------------------------------------------------------------------------
def manifold_project(prior, targets, device, steps=300, lr=0.05, l_tv=1e-4,
                     l_w=1e-3, batch=16, log_every=150):
    """Project each target image onto the StyleGAN manifold via image-space
    inversion. targets: (M,3,64,64) in [0,1]. Returns (M,3,64,64) [0,1]."""
    outs = []
    M = targets.shape[0]
    for s in range(0, M, batch):
        tb = targets[s:s + batch].to(device).clamp(0, 1)
        b = tb.shape[0]
        w = prior.init_w(b, device, jitter=0.02, from_mean=True)
        opt = torch.optim.Adam([w], lr=lr)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            img = prior.synth(w)
            l_img = F.mse_loss(img, tb)
            tv = U.total_variation(img).mean()
            wp = prior.w_prior_loss(w)
            (l_img + l_tv * tv + l_w * wp).backward()
            opt.step()
            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  project [{s}:{s+b}] it{it:4d} mse={l_img.item():.4e}",
                      flush=True)
        outs.append(prior.synth(w).detach().float().cpu())
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------------
# Evaluation proxy + selection
# ---------------------------------------------------------------------------
@torch.no_grad()
def _gray_feats(imgs, size=24):
    g = imgs.mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="area").reshape(imgs.shape[0], -1)
    return F.normalize(g - g.mean(dim=1, keepdim=True), dim=1)


def gradient_replay_cos(model, imgs, labels, keys, name2param, real_unit,
                        device, chunk=128):
    """Aggregate-gradient cosine of a full candidate set vs the observed gradient
    (higher = better explains the gradient). imgs treated as one batch."""
    imgs = imgs.to(device)
    labels = labels.to(device)
    logits = model(imgs)
    loss = F.cross_entropy(logits, labels)
    params = [name2param[k] for k in keys]
    grads = torch.autograd.grad(loss, params, create_graph=False)
    vec = torch.cat([g.reshape(-1) for g in grads])
    return float((vec / (vec.norm() + 1e-12) * real_unit).sum().item())


def select_diverse(imgs, quality, n=128, size=24, qbonus=0.3):
    """Greedy max-min diversity selection (quality-biased). No blur, no noise pad."""
    feats = _gray_feats(imgs, size)
    N = imgs.shape[0]
    q = quality.float()
    q = (q - q.min()) / (q.max() - q.min() + 1e-9)
    start = int(torch.argmax(quality))
    chosen = [start]
    dist = (1.0 - feats @ feats[start][:, None]).squeeze(1)
    while len(chosen) < min(n, N):
        score = dist + qbonus * q
        score[chosen] = -1e9
        nxt = int(torch.argmax(score))
        chosen.append(nxt)
        nd = (1.0 - feats @ feats[nxt][:, None]).squeeze(1)
        dist = torch.minimum(dist, nd)
    return imgs[chosen], chosen


def ensure_128(imgs, quality, prior, device):
    """Select 128 diverse; if the pool is short, top up with fresh random faces
    (on-manifold, diverse) rather than noise-padding."""
    if imgs.shape[0] >= U.N_IMAGES:
        sel, _ = select_diverse(imgs, quality, n=U.N_IMAGES)
        return sel
    need = U.N_IMAGES - imgs.shape[0]
    extra = prior.sample(need * 3, device).float().cpu()
    pool = torch.cat([imgs, extra], 0)
    q = torch.cat([quality.float(), torch.full((extra.shape[0],), quality.float().mean())])
    sel, _ = select_diverse(pool, q, n=U.N_IMAGES)
    return sel


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_pack(name, imgs128, pool, quality, meta, subdir):
    out_dir = U.work_path(subdir, name)
    U.ensure_dir(out_dir)
    imgs128 = imgs128.float().clamp(0, 1).contiguous()
    torch.save({"images": imgs128, **meta}, os.path.join(out_dir, "selected_128.pt"))
    torch.save({"images": pool.float().clamp(0, 1), "quality": quality},
               os.path.join(out_dir, "pool.pt"))
    U.save_image_grid(imgs128[:128], os.path.join(out_dir, "grid_selected.png"))
    U.save_image_grid(pool[:64].float().clamp(0, 1),
                      os.path.join(out_dir, "grid_pool_head.png"))
    U.save_json(meta, os.path.join(out_dir, "metrics.json"))
    print(f"[{name}] saved -> {out_dir}", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_smoke(prior, device, name, subdir):
    """Cheap end-to-end validation: sample faces + a few gradient-match steps."""
    print(f"[smoke] device={device} native_res={prior.native_res} "
          f"num_ws={prior.num_ws} w_plus={prior.w_plus}", flush=True)
    faces = prior.sample(32, device).float().cpu()
    gp = U.work_path(subdir, "_smoke")
    U.ensure_dir(gp)
    U.save_image_grid(faces, os.path.join(gp, "prior_samples.png"))
    print(f"[smoke] sampled 32 prior faces, stats: min={faces.min():.3f} "
          f"max={faces.max():.3f} mean={faces.mean():.3f}", flush=True)
    model, grad, keys, name2param, real, real_unit = setup_victim(name, device)
    assigned = IL.assign_labels(name, U.N_IMAGES).to(device)
    k = 4
    torch.manual_seed(0)
    w = prior.init_w(k, device, jitter=0.05, from_mean=True)
    opt = torch.optim.Adam([w], lr=0.05)
    cos0 = None
    for it in range(30):
        opt.zero_grad(set_to_none=True)
        imgs = prior.synth(w)
        fake = fake_grad_vec(model, imgs, keys, name2param,
                             hard_labels=assigned[:k])
        cos = (fake / (fake.norm() + 1e-12) * real_unit).sum()
        (1 - cos).backward()
        assert w.grad is not None, "double-backward failed: w.grad is None"
        opt.step()
        if cos0 is None:
            cos0 = cos.item()
        if it % 10 == 0 or it == 29:
            print(f"[smoke] {name} it{it:02d} cos={cos.item():.4f} "
                  f"(grad OK, |w.grad|={w.grad.norm().item():.3e})", flush=True)
    U.save_image_grid(prior.synth(w).detach().float().cpu(),
                      os.path.join(gp, f"{name}_match4.png"))
    print(f"[smoke] {name} cos {cos0:.4f} -> {cos.item():.4f} "
          f"(double-backward through StyleGAN works)", flush=True)


def run_latent(prior, device, name, subdir, args):
    pool, quality, ctx = latent_gradient_match(
        name, prior, device, k=args.k, restarts=args.restarts, steps=args.steps,
        lr=args.lr, l_cos=args.l_cos, l_l2=args.l_l2, l_tv=args.l_tv,
        l_w=args.l_w, l_div=args.l_div, jitter=args.jitter,
        soft_labels=not args.hard_labels, seed=args.seed, log_every=args.log_every,
        trunc=args.trunc)
    model, grad, keys, name2param, real_unit, assigned = ctx
    imgs128 = ensure_128(pool, quality, prior, device)
    replay = gradient_replay_cos(model, imgs128, assigned, keys, name2param,
                                 real_unit, device)
    # baseline: random prior faces (shows the matching adds signal)
    base = prior.sample(U.N_IMAGES, device).float().cpu()
    base_replay = gradient_replay_cos(model, base, assigned, keys, name2param,
                                      real_unit, device)
    meta = {"source": "stylegan_latent", "model": name, "family": grad["family"],
            "k": args.k, "restarts": args.restarts, "steps": args.steps,
            "pool_size": int(pool.shape[0]),
            "pool_cos_mean": float(quality.mean()),
            "pool_cos_max": float(quality.max()),
            "replay_cos_selected128": replay,
            "replay_cos_random128": base_replay,
            "w_plus": prior.w_plus, "generator": os.path.basename(
                os.environ.get("FL_SG_PKL", GP.DEFAULT_PKL))}
    print(f"[{name}] replay_cos selected={replay:.4f} random={base_replay:.4f} "
          f"pool_cos_max={quality.max():.4f}", flush=True)
    save_pack(name, imgs128, pool, quality, meta, subdir)


def run_project(prior, device, name, subdir, args):
    src = args.project_source or U.work_path(
        "candidates_v3", name, "selected_128.pt")
    if not os.path.exists(src):
        src = U.work_path("outputs_v3", "face_bank_v3.pt")
    if not os.path.exists(src):
        src = U.work_path("outputs", "face_bank.pt")
    print(f"[{name}] project source: {src}", flush=True)
    targets = torch.load(src, weights_only=False)["images"].float().clamp(0, 1)
    if targets.shape[0] > U.N_IMAGES:
        targets = targets[:U.N_IMAGES]
    proj = manifold_project(prior, targets, device, steps=args.steps, lr=args.lr,
                            l_tv=args.l_tv, l_w=1e-3, batch=args.k)
    # quality = negative reconstruction gap to its target (closer = better)
    gap = ((proj - targets[:proj.shape[0]]) ** 2).reshape(proj.shape[0], -1).mean(1)
    quality = -gap
    imgs128 = ensure_128(proj, quality, prior, device)
    meta = {"source": "stylegan_project", "model": name, "src": src,
            "steps": args.steps, "mean_recon_mse": float(gap.mean()),
            "generator": os.path.basename(os.environ.get("FL_SG_PKL", GP.DEFAULT_PKL))}
    # side-by-side grid: target (top) vs projection (bottom) for first 32
    m = min(32, proj.shape[0])
    sbs = torch.cat([targets[:m], proj[:m]], 0)
    U.ensure_dir(U.work_path(subdir, name))
    U.save_image_grid(sbs, U.work_path(subdir, name, "grid_project_target_vs_proj.png"),
                      ncol=m)
    save_pack(name, imgs128, proj, quality, meta, subdir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "latent", "project"], default="smoke")
    ap.add_argument("--models", nargs="+", default=["model2"])
    ap.add_argument("--subdir", default="runs/r1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l-cos", type=float, default=1.0)
    ap.add_argument("--l-l2", type=float, default=0.1)
    ap.add_argument("--l-tv", type=float, default=1e-4)
    ap.add_argument("--l-w", type=float, default=1e-4)
    ap.add_argument("--l-div", type=float, default=1e-2)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--trunc", type=float, default=0.0,
                    help="latent truncation-ball radius around w_avg; "
                         "0=off, <0=auto-calibrate (W p90)")
    ap.add_argument("--hard-labels", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out-size", type=int, default=64)
    ap.add_argument("--w-plus", dest="w_plus", action="store_true", default=True)
    ap.add_argument("--no-w-plus", dest="w_plus", action="store_false")
    ap.add_argument("--project-source", default=None)
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable -> cpu")
        device = "cpu"
    print(f"=== attack_stylegan mode={args.mode} models={args.models} "
          f"device={device} ===", flush=True)
    G = GP.load_generator(pkl=args.pkl, repo=args.repo, device=device)
    prior = GP.StyleGANPrior(G, out_size=args.out_size, w_plus=args.w_plus).to(device)
    print(f"[gen] loaded {os.path.basename(os.environ.get('FL_SG_PKL', GP.DEFAULT_PKL))} "
          f"native_res={prior.native_res} num_ws={prior.num_ws} w_dim={prior.w_dim}",
          flush=True)

    for name in args.models:
        try:
            if args.mode == "smoke":
                run_smoke(prior, device, name, args.subdir)
            elif args.mode == "latent":
                run_latent(prior, device, name, args.subdir, args)
            elif args.mode == "project":
                run_project(prior, device, name, args.subdir, args)
        except Exception as e:
            import traceback
            print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()

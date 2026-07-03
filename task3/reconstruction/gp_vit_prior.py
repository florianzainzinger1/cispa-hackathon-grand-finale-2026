"""Agent G / Lever L4 — generative-prior evaluation for the HARD models.

Focus: the two ViTs (model9, model11) and strided CNNs (model2,3,7). These are
the models NOT covered by Agent R's verbatim retrieval (ground_truth.pt has only
MLP+CNN source_model names, no ViT), so the ViTs are the genuine home for a face
prior.

This script is deliberately CONSERVATIVE and HONEST. Prior sessions proved:
  * StyleGAN *latent-matching* -> realistic-but-wrong-identity, HURT leaderboard.
  * StyleGAN *projection* of a real-face recon -> tiny +, and reproduces garbage
    when the input is garbage.
  * ViT patch-embed carries little recoverable identity.

So we do NOT re-run latent matching (known dead). We:
  1. PROJECT the current-best per-model recon onto the StyleGAN manifold (image
     inversion) — low-risk sharpening that preserves the coarse identity.
  2. Evaluate {current_best, r2_latent (if present), projection} against every
     available signal:
       - gradient replay cosine (re-feed, cosine vs observed) — but treated as a
         WEAK proxy (it mis-ranked model2 latent as good).
       - self-SSIM structure vs the source (does projection stay faithful?).
       - if ground_truth.pt matches this challenge model, the TRUE Hungarian SSIM.
  3. Emit a keep/defer decision per model. Default = DEFER to bank.

Reuses the r1/r2 no-plugin StyleGAN standup (gan_prior.py, attack_stylegan.py).
Env: FL_DEVICE, FL_WORK_ROOT, FL_DATA_ROOT, FL_SG_REPO, FL_SG_PKL.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np

import utils as U
import attack_opt as AO
import infer_labels as IL
import gan_prior as GP
import attack_stylegan as SG

GT_PATH = None  # set from work root at runtime


# ---------------------------------------------------------------------------
# alignment + SSIM (skimage)
# ---------------------------------------------------------------------------
def _ssim_gray(a, b):
    from skimage.metrics import structural_similarity as ssim
    return ssim(a, b, data_range=1.0)


def aligned_ssim_pair(a, b, max_shift=3):
    """Best SSIM of grayscale a vs b over integer shifts up to +/-max_shift."""
    from skimage.metrics import structural_similarity as ssim
    ag = a.mean(0).numpy() if a.ndim == 3 else a.numpy()
    bg = b.mean(0).numpy() if b.ndim == 3 else b.numpy()
    best = -1.0
    H, W = ag.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            bs = np.roll(np.roll(bg, dy, axis=0), dx, axis=1)
            s = ssim(ag, bs, data_range=1.0)
            if s > best:
                best = s
    return best


def hungarian_ssim(A, B, max_shift=0):
    """Hungarian one-to-one grayscale SSIM between two (N,3,64,64) sets.
    max_shift>0 aligns each pair (expensive); 0 = no alignment."""
    from skimage.metrics import structural_similarity as ssim
    from scipy.optimize import linear_sum_assignment
    A = A.float().clamp(0, 1)
    B = B.float().clamp(0, 1)
    n = A.shape[0]
    Ag = A.mean(1).numpy()
    Bg = B.mean(1).numpy()
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for k in range(n):
            if max_shift > 0:
                C[i, k] = aligned_ssim_pair(A[i], B[k], max_shift)
            else:
                C[i, k] = ssim(Ag[i], Bg[k], data_range=1.0)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


# ---------------------------------------------------------------------------
# gradient replay proxy (weak) — reuse the r1/r2 helper style
# ---------------------------------------------------------------------------
def victim_ctx(name, device):
    model, grad = AO.build_any(name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    n2p = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in n2p]
    real = AO.real_grad_vector(grad, keys).to(device)
    ru = real / (real.norm() + 1e-12)
    return model, grad, keys, n2p, ru


def replay_cos(model, imgs, labels, keys, n2p, ru, device):
    imgs = imgs.to(device)
    labels = labels.to(device)
    logits = model(imgs)
    loss = F.cross_entropy(logits, labels)
    g = torch.autograd.grad(loss, [n2p[k] for k in keys])
    vec = torch.cat([x.reshape(-1) for x in g])
    return float((vec / (vec.norm() + 1e-12) * ru).sum())


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def load_current_best(name):
    p = U.work_path("outputs_final", "submission_current_best.pt")
    d = torch.load(p, weights_only=False)
    e = d[name]
    return (e["images"] if isinstance(e, dict) else e).float().clamp(0, 1)


def load_r2_latent(name):
    p = U.work_path("runs", "r2", name, "selected_128.pt")
    if not os.path.exists(p):
        p = U.work_path("runs", "r1", name, "selected_128.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)


def find_gt_for(name, device):
    """Return (gt_images, match_cos) if ground_truth.pt has a >0.9-cos match for
    this challenge model, else (None, best_cos)."""
    if GT_PATH is None or not os.path.exists(GT_PATH):
        return None, None
    gt = torch.load(GT_PATH, weights_only=False)
    model, grad, keys, n2p, ru = victim_ctx(name, device)
    best_key, best_cos = None, -2.0
    for gk, entry in gt.items():
        imgs = entry["images"].float().clamp(0, 1)
        labels = entry["labels"].long()
        c = replay_cos(model, imgs, labels, keys, n2p, ru, device)
        if c > best_cos:
            best_cos, best_key = c, gk
    if best_cos > 0.9:
        return gt[best_key]["images"].float().clamp(0, 1), best_cos
    return None, best_cos


# ---------------------------------------------------------------------------
# main per-model evaluation
# ---------------------------------------------------------------------------
def eval_model(name, prior, device, do_project=True, proj_steps=300):
    print(f"\n========== {name} ==========", flush=True)
    model, grad, keys, n2p, ru = victim_ctx(name, device)
    fam = grad["family"]
    labels = IL.assign_labels(name, U.N_IMAGES).to(device)

    cur = load_current_best(name)
    sources = {"current_best": cur}

    r2 = load_r2_latent(name)
    if r2 is not None:
        sources["r2_latent"] = r2

    if do_project:
        proj = SG.manifold_project(prior, cur, device, steps=proj_steps,
                                   lr=0.05, l_tv=1e-4, l_w=1e-3, batch=16)
        sources["projection"] = proj.float().clamp(0, 1)

    # ground-truth oracle if available
    gt_imgs, gt_cos = find_gt_for(name, device)
    has_gt = gt_imgs is not None
    print(f"[{name}] family={fam} gt_match={'YES cos=%.3f'%gt_cos if has_gt else 'no (best cos=%.3f)'%(gt_cos or -1)}", flush=True)

    rows = {}
    for sname, imgs in sources.items():
        rc = replay_cos(model, imgs, labels, keys, n2p, ru, device)
        row = {"replay_cos": rc, "tv": float(U.total_variation(imgs).mean())}
        if has_gt:
            row["true_ssim_aligned"] = hungarian_ssim(imgs, gt_imgs, max_shift=3)
        rows[sname] = row
        extra = f" TRUE_SSIM={row.get('true_ssim_aligned'):.4f}" if has_gt else ""
        print(f"  [{name}] {sname:14s} replay_cos={rc:+.4f} tv={row['tv']:.4f}{extra}", flush=True)

    # decision
    if has_gt:
        ranked = sorted(rows.items(), key=lambda kv: -kv[1]["true_ssim_aligned"])
        best = ranked[0][0]
        decision = ("KEEP:" + best) if best != "current_best" and \
            rows[best]["true_ssim_aligned"] > rows["current_best"]["true_ssim_aligned"] + 1e-4 \
            else "DEFER:current_best"
        metric = "true_ssim"
    else:
        # no oracle -> replay cos is the only proxy, but it is KNOWN misleading for
        # latent-match. Only 'projection' is low-risk (identity-preserving). Keep
        # projection only if it does not *lose* replay cos vs current_best AND
        # keeps tv similar (no hallucination). Default DEFER.
        cb = rows["current_best"]
        decision = "DEFER:current_best"
        if "projection" in rows:
            pj = rows["projection"]
            if pj["replay_cos"] >= cb["replay_cos"] - 0.02 and pj["tv"] <= cb["tv"] * 1.05:
                decision = "CANDIDATE_KEEP:projection (needs leaderboard confirm)"
        metric = "replay_cos(weak)"

    print(f"  [{name}] DECISION [{metric}] -> {decision}", flush=True)

    # save projection output to my namespace if it exists
    outdir = U.work_path("outputs_gp")
    U.ensure_dir(outdir)
    if "projection" in sources:
        torch.save({"images": sources["projection"], "source": "stylegan_project",
                    "model": name, "decision": decision, "rows": rows},
                   os.path.join(outdir, f"gp_{name}.pt"))
        # side-by-side grid cur vs proj (first 32)
        m = min(32, cur.shape[0])
        sbs = torch.cat([cur[:m], sources["projection"][:m]], 0)
        U.save_image_grid(sbs, os.path.join(outdir, f"gp_{name}_cur_vs_proj.png"), ncol=m)

    return {"model": name, "family": fam, "has_gt": has_gt, "gt_cos": gt_cos,
            "rows": rows, "decision": decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["model9", "model11"])
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--proj-steps", type=int, default=300)
    args = ap.parse_args()

    global GT_PATH
    GT_PATH = U.work_path("outputs_retr", "ground_truth.pt")

    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"=== gp_vit_prior device={device} models={args.models} "
          f"gt={'present' if os.path.exists(GT_PATH) else 'absent'} ===", flush=True)

    prior = None
    if not args.no_project:
        G = GP.load_generator(device=device)
        prior = GP.StyleGANPrior(G, out_size=64, w_plus=True).to(device)
        print(f"[gen] StyleGAN native_res={prior.native_res} num_ws={prior.num_ws}", flush=True)

    summary = []
    for name in args.models:
        try:
            summary.append(eval_model(name, prior, device,
                                      do_project=not args.no_project,
                                      proj_steps=args.proj_steps))
        except Exception as e:
            import traceback
            print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    out = U.work_path("reports", "gp", "gp_vit_prior_summary.json")
    U.ensure_dir(os.path.dirname(out))
    U.save_json(summary, out)
    print(f"\n[done] summary -> {out}", flush=True)


if __name__ == "__main__":
    main()

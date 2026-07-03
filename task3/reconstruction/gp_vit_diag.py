"""Agent G (agent5) — decisive ViT leak diagnostic for models 9 & 11.

Goal: BEFORE spending A100 time on DPS diffusion, establish whether the ViT
gradients carry recoverable PER-SAMPLE identity, or only a few aggregate modes.
If a batch of RANDOM faces already explains the observed gradient as well as any
optimized set, no generative prior can beat the diverse bank on SSIM (it would
only synthesize realistic-but-wrong identities, the confirmed leaderboard-loser).

Three cheap CPU tests per ViT:

  (A) PATCH-EMBED LEAK RANK.  The patch-embed conv weight gradient is
      gW_pe = sum_{b,p} (dL/dz_{b,p}) * patch_{b,p}^T   (shape (dim, 3*patch*patch)).
      Its right-singular vectors live in *patch-content* space (3*P*P dims). We
      report the singular spectrum + effective rank + energy in the top-k modes.
      Few dominant modes => the batch leaks only a low-freq colour/structure
      subspace (the "~8 modes" hypothesis), and per-sample recon is impossible.

  (B) PATCH-EMBED MEAN-CONTENT RECON (the projection candidate).  The top right-
      singular vectors of gW_pe, reshaped to (3,P,P) and tiled across the patch
      grid, are the only image-space signal the patch embed exposes. We build a
      best-effort low-freq field from them and score it (aligned SSIM) against the
      current bank as a sanity check — this is candidate route #2 in the brief.

  (C) IDENTIFIABILITY PROBE (the decisive one).  Replay-cosine (re-feed a full
      batch, cosine of produced vs observed full-gradient) for:
        * current-best bank recon,
        * K sets of RANDOM CelebA-ish faces (uniform-noise-smoothed + bank draws),
        * (optional) r1/r2 latent set if present.
      If random ~ bank on replay cosine, the gradient does not discriminate the
      identity of the batch -> DEFER to bank is the honest verdict.

Pure CPU, no external weights. Writes reports/gp/vit_diag.json and PNGs.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

import utils as U
import attack_opt as AO
import infer_labels as IL


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
    logits = model(imgs.to(device))
    loss = F.cross_entropy(logits, labels.to(device))
    g = torch.autograd.grad(loss, [n2p[k] for k in keys])
    vec = torch.cat([x.reshape(-1) for x in g])
    return float((vec / (vec.norm() + 1e-12) * ru).sum())


def eff_rank(s):
    """Participation-ratio effective rank of a singular-value vector."""
    s = s.astype(np.float64)
    p = s ** 2
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def aligned_ssim_pair(a, b, max_shift=3):
    from skimage.metrics import structural_similarity as ssim
    ag = a.mean(0).numpy() if a.ndim == 3 else a.numpy()
    bg = b.mean(0).numpy() if b.ndim == 3 else b.numpy()
    best = -1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            bs = np.roll(np.roll(bg, dy, axis=0), dx, axis=1)
            s = ssim(ag, bs, data_range=1.0)
            best = max(best, s)
    return best


# ---------------------------------------------------------------------------
def patch_embed_analysis(name, grad, patch, dim):
    """(A) + (B): singular spectrum of the patch-embed weight gradient."""
    gpe = grad["gradients"]["patch_embed.proj.weight"].float()   # (dim,3,P,P)
    M = gpe.reshape(dim, -1)                                      # (dim, 3*P*P)
    U_, S, Vh = torch.linalg.svd(M, full_matrices=False)
    S = S.numpy()
    top = S / (S.sum() + 1e-12)
    er = eff_rank(S)
    # energy in top-k
    cum = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    k_for = {frac: int(np.searchsorted(cum, frac) + 1) for frac in (0.5, 0.9, 0.95, 0.99)}
    # top singular directions reshaped to a patch (3,P,P) -> tile to (3,64,64)
    grid = 64 // patch
    fields = []
    for j in range(min(8, Vh.shape[0])):
        v = Vh[j].reshape(3, patch, patch)
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        tiled = v.repeat(1, grid, grid)                          # (3,64,64)
        fields.append(tiled)
    fields = torch.stack(fields)                                 # (8,3,64,64)
    return {
        "sing_top20": S[:20].tolist(),
        "eff_rank": er,
        "k_for_energy": k_for,
        "ratio_s0_s1": float(S[0] / (S[1] + 1e-12)),
    }, fields


def make_random_faces(n, device, bank=None, seed=0):
    """Diverse plausible faces: half from the bank (if available), half smoothed
    low-freq noise in a face-ish colour range. Used only as a null hypothesis."""
    g = torch.Generator().manual_seed(seed)
    out = []
    if bank is not None and bank.shape[0] >= n // 2:
        perm = torch.randperm(bank.shape[0], generator=g)[: n // 2]
        out.append(bank[perm].clone())
        m = n - n // 2
    else:
        m = n
    if m > 0:
        z = torch.rand(m, 3, 8, 8, generator=g)
        z = F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)
        z = z * 0.6 + 0.2
        out.append(z)
    faces = torch.cat(out, 0)[:n]
    return faces.clamp(0, 1).to(device)


def load_bank():
    p = U.work_path("outputs", "face_bank.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
    return None


def load_current_best(name):
    p = U.work_path("outputs_final", "submission_current_best.pt")
    d = torch.load(p, weights_only=False)
    e = d[name]
    return (e["images"] if isinstance(e, dict) else e).float().clamp(0, 1)


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    outrep = U.work_path("reports", "gp")
    outimg = U.work_path("outputs_gp", "_diag")
    U.ensure_dir(outrep)
    U.ensure_dir(outimg)
    bank = load_bank()
    summary = {}

    arch = {"model9": (8, 192), "model11": (16, 256)}
    for name in ["model9", "model11"]:
        print(f"\n===== {name} =====", flush=True)
        model, grad, keys, n2p, ru = victim_ctx(name, device)
        labels = IL.assign_labels(name, U.N_IMAGES).to(device)
        patch, dim = arch[name]

        # (A)+(B) patch-embed leak
        pe, fields = patch_embed_analysis(name, grad, patch, dim)
        U.save_image_grid(fields, os.path.join(outimg, f"{name}_pe_modes.png"), ncol=8)
        print(f"  patch-embed eff_rank={pe['eff_rank']:.1f} "
              f"s0/s1={pe['ratio_s0_s1']:.2f} k@90%={pe['k_for_energy'][0.9]} "
              f"k@99%={pe['k_for_energy'][0.99]}", flush=True)

        # (C) identifiability probe
        cb = load_current_best(name)
        c_cb = replay_cos(model, cb, labels, keys, n2p, ru, device)
        rand_cos = []
        for s in range(4):
            faces = make_random_faces(U.N_IMAGES, device, bank=bank, seed=100 + s)
            rand_cos.append(replay_cos(model, faces, labels, keys, n2p, ru, device))
        rc = float(np.mean(rand_cos))
        # also: a pure-mean-gray batch (trivial null)
        gray = torch.full((U.N_IMAGES, 3, 64, 64), 0.5)
        c_gray = replay_cos(model, gray, labels, keys, n2p, ru, device)

        # r1/r2 latent if present
        lat_cos = None
        for sub in ["r2", "r1"]:
            p = U.work_path("runs", sub, name, "selected_128.pt")
            if os.path.exists(p):
                im = torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
                lat_cos = replay_cos(model, im, labels, keys, n2p, ru, device)
                break

        disc = c_cb - rc
        print(f"  replay_cos  bank={c_cb:+.4f}  random={rc:+.4f}"
              f"  gray={c_gray:+.4f}"
              + (f"  latent={lat_cos:+.4f}" if lat_cos is not None else ""), flush=True)
        print(f"  DISCRIMINATION bank-random = {disc:+.4f}  "
              f"(random seeds {[round(x,3) for x in rand_cos]})", flush=True)

        summary[name] = {
            "patch_embed": pe,
            "replay_cos_bank": c_cb,
            "replay_cos_random_mean": rc,
            "replay_cos_random_seeds": rand_cos,
            "replay_cos_gray": c_gray,
            "replay_cos_latent": lat_cos,
            "discrimination_bank_minus_random": disc,
        }

    U.save_json(summary, os.path.join(outrep, "vit_diag.json"))
    print(f"\n[done] -> {os.path.join(outrep, 'vit_diag.json')}", flush=True)


if __name__ == "__main__":
    main()

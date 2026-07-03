"""APRIL — analytic ViT gradient inversion for models 9 & 11 (agent3, ViT-analytic route).

Leak (arXiv 2112.14087): with a LEARNABLE positional embedding, z = token + pos_embed, so
    dL/d(pos_embed)[t] = sum_b dL/d(token_{b,t})           (batch-summed per-position token grad)
and since token = W_pe x + b_pe (patch_embed Conv2d, kernel=stride=patch, non-overlapping),
    gE = dL/d W_pe = sum_{b,t} g_{b,t} x_{b,t}^T = G_col @ X   with  G_col = [g_1..g_N] (dim x N).
For B=1, G_col columns are the per-position pos-embed grads -> X = pinv(G_col) @ gE recovers all N
patches EXACTLY; tile -> image. For B>1 the pos-embed grad is a BATCH SUM, so this recovers the
per-position BATCH-MEAN patch (one mean face), NOT per-sample identity -- we verify this empirically.

Modes (argv): probe | selftest | attack | all (default all). Models default 9,11.
Self-test at B=1/4/16/128 both PROVES the implementation (B=1 -> ~0 error) and MEASURES the collapse.
Namespace: vit_*  ->  outputs_vit/ , reports/vit/. GPU via vit_job.sh (sbatch).
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

import utils as U
import vit as VIT
import infer_labels as IL

DEVICE = os.environ.get("FL_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"

CFG = {
    "model9": dict(patch=8, dim=192, use_cls=True),
    "model11": dict(patch=16, dim=256, use_cls=False),
}


# ---------------------------------------------------------------------------
# APRIL closed form
# ---------------------------------------------------------------------------
def april_tensors(g, use_cls):
    """Return (gE2 [dim,3p^2], gb [dim], G_pos [Npatch,dim], dim, p)."""
    gE = g["patch_embed.proj.weight"].float()        # (dim,3,p,p)
    dim, _, p, _ = gE.shape
    gE2 = gE.reshape(dim, -1)                          # (dim, 3p^2)
    gb = g.get("patch_embed.proj.bias")
    gb = gb.float() if gb is not None else None
    pe = g["pos_embed"].float().squeeze(0)            # (Ntok, dim)
    G_pos = pe[1:] if use_cls else pe                  # drop cls row -> (Npatch, dim)
    return gE2, gb, G_pos, dim, p


def tile_patches(patches, grid, p):
    """patches (grid*grid,3,p,p) row-major -> (3, grid*p, grid*p)."""
    canvas = torch.zeros(3, grid * p, grid * p, device=patches.device)
    for i in range(patches.shape[0]):
        r, c = divmod(i, grid)
        canvas[:, r * p:(r + 1) * p, c * p:(c + 1) * p] = patches[i]
    return canvas


def april_recover(G_pos, gE2, grid, p, rcond=1e-3):
    """X = pinv(G_pos^T) @ gE2 -> Npatch patches -> tiled image (raw, unclamped)."""
    Gt = G_pos.t().contiguous()                       # (dim, Npatch) = G_col
    X = torch.linalg.pinv(Gt, rcond=rcond) @ gE2      # (Npatch, 3p^2)
    npatch = X.shape[0]
    patches = X.reshape(npatch, 3, p, p)
    img = tile_patches(patches, grid, p)              # (3,64,64) raw
    return img, patches, X


# ---------------------------------------------------------------------------
# Self-test: proves correctness (B=1) and measures batch collapse
# ---------------------------------------------------------------------------
def ssim_gray(a, b, C1=0.01 ** 2, C2=0.03 ** 2, win=7):
    a = a.mean(0, keepdim=True).unsqueeze(0)
    b = b.mean(0, keepdim=True).unsqueeze(0)
    k = torch.ones(1, 1, win, win, device=a.device) / (win * win)
    mu_a = F.conv2d(a, k, padding=win // 2); mu_b = F.conv2d(b, k, padding=win // 2)
    va = F.conv2d(a * a, k, padding=win // 2) - mu_a ** 2
    vb = F.conv2d(b * b, k, padding=win // 2) - mu_b ** 2
    vab = F.conv2d(a * b, k, padding=win // 2) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def selftest(name, out_dir, batches=(1, 4, 16, 128)):
    cfg = CFG[name]
    state = U.load_state(name)
    model, missing, unexpected = VIT.build_vit(name, state)
    model = model.to(DEVICE)
    for pmt in model.parameters():
        pmt.requires_grad_(True)
    p = cfg["patch"]; grid = 64 // p
    rows = []
    print(f"[selftest {name}] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}", flush=True)
    for Bsz in batches:
        torch.manual_seed(1234 + Bsz)
        x = torch.rand(Bsz, 3, 64, 64, device=DEVICE)
        y = torch.randint(0, 200, (Bsz,), device=DEVICE)
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y, reduction="mean")
        loss.backward()
        g = {k: v.grad.detach() for k, v in model.named_parameters() if v.grad is not None}
        gE2, gb, G_pos, dim, pp = april_tensors(g, cfg["use_cls"])
        img, patches, X = april_recover(G_pos, gE2, grid, pp)
        img = img.detach().cpu()
        target = (x[0] if Bsz == 1 else x.mean(0)).cpu()
        mse_mean = float((U.per_image_minmax(img[None])[0] - U.per_image_minmax(target[None])[0]).pow(2).mean())
        # scale-free: recovered vs (per-sample) and vs batch-mean
        ss_mean = ssim_gray(U.per_image_minmax(img[None])[0].to(DEVICE), U.per_image_minmax(target[None])[0].to(DEVICE))
        # also: best-case per-sample SSIM (max over the batch) -- does it match ANY individual?
        ss_persample = -1.0
        if Bsz > 1:
            imn = U.per_image_minmax(img[None])[0].to(DEVICE)
            for b in range(min(Bsz, 128)):
                ss_persample = max(ss_persample, ssim_gray(imn, U.per_image_minmax(x[b][None])[0].to(DEVICE)))
        rows.append({"B": Bsz, "mse_vs_mean": round(mse_mean, 5),
                     "ssim_vs_mean": round(ss_mean, 4),
                     "ssim_best_persample": round(ss_persample, 4)})
        U.save_image_grid(torch.stack([U.per_image_minmax(target[None])[0], U.per_image_minmax(img[None])[0]]),
                          os.path.join(out_dir, f"selftest_{name}_B{Bsz}.png"), ncol=2, upscale=3)
        print(f"  B={Bsz:4d}  mse_vs_mean={mse_mean:.5f}  ssim_vs_mean={ss_mean:.4f}  "
              f"ssim_best_persample={ss_persample:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Real attack on the provided aggregated gradient
# ---------------------------------------------------------------------------
def patchembed_spectrum(gE2):
    S = torch.linalg.svdvals(gE2)
    s0 = float(S[0])
    eff = int((S > 0.01 * S[0]).sum())
    e = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    return {"eff_rank_1pct": eff, "sv0": s0,
            "sv": [round(float(S[i]), 5) for i in [0, 1, 3, 7, 15, min(31, len(S) - 1)]],
            "energy_top8": round(float(e[min(7, len(e) - 1)]), 4)}


def measure_selfconsistency(name, X, use_cls):
    """Re-feed recovered images through the ViT with inferred labels; cosine of the
    produced patch_embed+pos_embed grads vs observed (label-free-ish; labels approx)."""
    state = U.load_state(name)
    model, _, _ = VIT.build_vit(name, state)
    model = model.to(DEVICE)
    for pmt in model.parameters():
        pmt.requires_grad_(True)
    labels = IL.assign_labels(name, X.shape[0]).to(DEVICE)
    model.zero_grad(set_to_none=True)
    out = model(X.to(DEVICE))
    loss = F.cross_entropy(out, labels, reduction="mean")
    loss.backward()
    g = {k: v.grad.detach() for k, v in model.named_parameters() if v.grad is not None}
    grad_obs = U.load_grad(name)["gradients"]
    keys = ["patch_embed.proj.weight", "pos_embed"]
    obs = torch.cat([grad_obs[k].reshape(-1).float() for k in keys])
    hat = torch.cat([g[k].reshape(-1).float().cpu() for k in keys])
    return float(F.cosine_similarity(obs, hat, dim=0))


def load_bank(name):
    """ViT fallback baseline = c3 slot for this model (diverse real faces)."""
    for path in [U.work_path("outputs_v3", "submission_v3.pt"),
                 U.work_path("outputs", "submission.pt")]:
        if os.path.exists(path):
            d = torch.load(path, weights_only=False)
            if name in d:
                return d[name].float().clamp(0, 1)
    return None


def attack(name, out_dir):
    cfg = CFG[name]
    grad = U.load_grad(name)
    g = grad["gradients"]
    has_pos = "pos_embed" in g and g["pos_embed"] is not None
    p = cfg["patch"]; grid = 64 // p
    gE2, gb, G_pos, dim, pp = april_tensors(g, cfg["use_cls"])
    spec = patchembed_spectrum(gE2)
    pos_norm = float(G_pos.norm())
    print(f"[attack {name}] pos_embed_grad_present={has_pos} pos_grad_norm={pos_norm:.4f} "
          f"patchembed_eff_rank={spec['eff_rank_1pct']} energy_top8={spec['energy_top8']}", flush=True)

    img, patches, X = april_recover(G_pos, gE2, grid, pp)
    mean_face = U.per_image_minmax(img.detach().cpu()[None])[0]      # (3,64,64)
    U.save_image_grid(mean_face[None], os.path.join(out_dir, f"april_meanface_{name}.png"), ncol=1, upscale=4)
    U.save_image_grid(U.per_image_minmax(patches.detach().cpu()),
                      os.path.join(out_dir, f"april_patches_{name}.png"), ncol=grid, upscale=4)

    # candidate set: the APRIL mean face duplicated to 128 (honest APRIL-only output)
    april128 = mean_face[None].repeat(128, 1, 1, 1).clamp(0, 1)

    # measurement: self-consistency of april mean-dup vs bank
    cos_april = measure_selfconsistency(name, april128, cfg["use_cls"])
    bank = load_bank(name)
    cos_bank = measure_selfconsistency(name, bank, cfg["use_cls"]) if bank is not None else None

    # distinctness (Hungarian coverage): april128 has 1 unique -> effective 1
    info = {
        "name": name, "patch": p, "dim": dim, "use_cls": cfg["use_cls"],
        "pos_embed_grad_present": has_pos, "pos_grad_norm": pos_norm,
        "patchembed_spectrum": spec,
        "selfcons_cos_april_meandup": round(cos_april, 4),
        "selfcons_cos_bank": round(cos_bank, 4) if cos_bank is not None else None,
        "april_effective_unique": 1,
        "bank_effective_unique": 128 if bank is not None else None,
    }
    # save the honest APRIL-only artifact (mean face x128) + the raw mean face
    torch.save({"images": april128, "mean_face": mean_face, "patches": patches.detach().cpu(),
                "method": "april_analytic", "info": info},
               U.work_path("outputs_vit", f"vit_{name[5:]}.pt"))
    print(f"  saved outputs_vit/vit_{name[5:]}.pt ; selfcons cos april={cos_april:.4f} "
          f"bank={cos_bank if cos_bank is None else round(cos_bank,4)}", flush=True)
    return info


def main():
    args = sys.argv[1:]
    mode = "all"
    for m in ["probe", "selftest", "attack", "all"]:
        if m in args:
            mode = m
    names = [a for a in args if a.startswith("model")] or ["model9", "model11"]
    out_dir = U.work_path("reports", "vit")
    U.ensure_dir(out_dir)
    U.ensure_dir(U.work_path("outputs_vit"))
    summary = {}
    for name in names:
        entry = {}
        try:
            if mode in ("selftest", "all"):
                entry["selftest"] = selftest(name, out_dir)
            if mode in ("attack", "all", "probe"):
                entry["attack"] = attack(name, out_dir)
        except Exception as e:
            import traceback
            entry["error"] = str(e)
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
        summary[name] = entry
    U.save_json(summary, os.path.join(out_dir, "april_summary.json"))
    print("wrote", os.path.join(out_dir, "april_summary.json"), flush=True)


if __name__ == "__main__":
    main()

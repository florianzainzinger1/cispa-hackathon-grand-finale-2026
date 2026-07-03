"""WS1 — Archetypal (extreme-point) vertex selection for the leaky-MLP analytic
attack.  NEW MODULE (does not modify v3_select / a_factorize / postprocess).

Why: the analytic ratio rows r_i = gW[i]/gb[i] are CONVEX COMBINATIONS of the
128 batch images.  Single-sample neurons are the VERTICES of the row cloud (clean
faces); interior points are blend-ghosts (smooth, low-TV -> the grain-free blur
that HURTS one-to-one SSIM).  The current selector ranks by quality
q = rng_frac + 0.3*contrast - 2*tv, whose -2*tv term REWARDS smooth interior
blends over real vertex faces.  We replace selection with SPA (Successive
Projection Algorithm): a deterministic separable-NMF vertex finder that can only
ever pick existing extreme rows, so it never returns an interior blend.

Scope: leaky-MLP models {1,4,5,8} only (net.0 = Linear(1024,12288); in_dim==12288
== the analytic direct_image path).  Ratio recovers x at TRUE [0,1] scale for any
activation (act' cancels in gW/gb), so we output .clamp(0,1) -- never per-image
minmax, which would destroy the SSIM luminance match.

Login-node safe: numpy / scipy / torch only.  Read-only w.r.t. existing outputs;
writes candidates_arch/<name>/ and (via the driver) outputs_final/base_archetypal.pt.
"""
import os
import sys
import json
import numpy as np
import torch

import utils as U
import rec_lib as R
import retr_lib as RL

LEAKY_MLP = ["model1", "model4", "model5", "model8"]   # net.0 Linear(1024,12288)
# activation cancels in the ratio -> clamp is correct for all four.
SCALE = {"model1": "clamp", "model4": "clamp", "model5": "clamp", "model8": "clamp"}


# ----------------------------------------------------------------------------
# ratio rows
# ----------------------------------------------------------------------------
def load_ratio(name):
    g = U.load_grad(name)["gradients"]
    gW = g["net.0.weight"].float()          # (1024, 12288)
    gb = g["net.0.bias"].float()            # (1024,)
    ratio, reliable = R.ratio_rows(gW, gb)  # ratio (1024,12288) w/ NaN, reliable (1024,) bool
    return gW, gb, ratio, reliable


def _topk_subspace(gW, k):
    """retr_lib.topk_right_subspace with a robust fallback: the gram eigh can fail
    to converge on ill-conditioned / noisy gW (seen on synthetic DP-noised grads).
    Fall back to a ridge-regularised gram, then to full SVD."""
    try:
        return RL.topk_right_subspace(gW, k=k)
    except Exception:
        gWf = gW.float()
        try:
            Mg = gWf @ gWf.t()
            Mg = Mg + 1e-3 * float(Mg.diagonal().mean().clamp_min(1e-12)) * torch.eye(Mg.shape[0])
            evals, evecs = torch.linalg.eigh(Mg)
            order = torch.argsort(evals, descending=True)[:k]
            Uu = evecs[:, order]
            s = evals[order].clamp_min(0).sqrt()
            V = (gWf.t() @ Uu) / s.clamp_min(1e-8)
            V = V.t().contiguous()
            V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
            return V, s
        except Exception:
            Uu, S, Vh = torch.linalg.svd(gWf, full_matrices=False)
            return Vh[:k].contiguous(), S[:k]


def _facelike_mask(rows_flat, rng_cut=0.50):
    """rows_flat (N,12288) -> bool mask of rows that look like an image (mostly
    in-range), rejecting pure-noise / degenerate rows before SPA."""
    st = U.image_stats(rows_flat.reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W))
    return st["finite"] & (st["rng_frac"] > rng_cut)


# ----------------------------------------------------------------------------
# SPA (Successive Projection Algorithm) in the rank-k image subspace
# ----------------------------------------------------------------------------
def spa_select(ratio, gW, reliable, n=128, rng_cut=0.50, k_sub=128):
    finite = torch.isfinite(ratio).all(dim=1)
    keep = reliable & finite
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx, idx, torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    rows = ratio[idx]                                   # (Nr, 12288) exact scale
    fmask = _facelike_mask(rows, rng_cut=rng_cut)
    idx = idx[fmask]
    rows = ratio[idx]
    # precondition: project into top-k right-singular subspace of gW (denoise+reduce)
    k = min(k_sub, gW.shape[0])
    V, s = _topk_subspace(gW, k)                        # V (k,12288) rows orthonormal
    Y = rows @ V.t()                                    # (Nr, k) coords, isometry up to rotation
    c = Y.mean(0)
    Z = (Y - c).clone()
    picks = []
    for _ in range(min(n, Z.shape[0])):
        norms = Z.pow(2).sum(1)
        i = int(torch.argmax(norms))
        if float(norms[i]) <= 1e-10:
            break
        picks.append(i)
        u = Z[i] / (Z[i].norm() + 1e-12)
        Z = Z - torch.outer(Z @ u, u)                  # deflate
    sel_local = torch.tensor(picks, dtype=torch.long)
    # farthest-point fill in Y if SPA exhausted early (rank-deficient)
    if sel_local.numel() < n and idx.numel() > sel_local.numel():
        chosen = set(sel_local.tolist())
        Yc = Y
        selY = Yc[sel_local] if sel_local.numel() else Yc.mean(0, keepdim=True)
        mind = torch.cdist(Yc, selY).min(dim=1).values
        for j in sel_local.tolist():
            mind[j] = -1.0
        while sel_local.numel() < n:
            j = int(torch.argmax(mind))
            if j in chosen or float(mind[j]) < 0:
                break
            chosen.add(j)
            sel_local = torch.cat([sel_local, torch.tensor([j])])
            mind = torch.minimum(mind, torch.cdist(Yc, Yc[j:j + 1]).squeeze(1))
            mind[j] = -1.0
    sel_global = idx[sel_local]
    return sel_global, idx, Y, sel_local


# ----------------------------------------------------------------------------
# OLD selection (quality + greedy dedup) reproduced on the ratio-row cloud,
# so purity is an apples-to-apples comparison against SPA.
# ----------------------------------------------------------------------------
def quality_select(ratio, reliable, n=128, rng_cut=0.50, dedup_eps=0.03):
    finite = torch.isfinite(ratio).all(dim=1)
    idx = torch.nonzero(reliable & finite, as_tuple=False).flatten()
    rows = ratio[idx]
    fmask = _facelike_mask(rows, rng_cut=rng_cut)
    idx = idx[fmask]
    imgs = ratio[idx].reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W).clamp(0, 1)
    st = U.image_stats(imgs)
    contrast = torch.clamp(st["std"] / 0.25, max=1.0)
    q = st["rng_frac"] + 0.3 * contrast - 2.0 * st["tv"]      # v3_select quality
    # 16x16 gray feats for dedup
    g = torch.nn.functional.interpolate(imgs.mean(1, keepdim=True), size=(16, 16),
                                        mode="bilinear", align_corners=False)
    feats = g.reshape(g.shape[0], -1)
    order = torch.argsort(q, descending=True).tolist()
    chosen, cfeats = [], []
    for i in order:
        if len(chosen) >= n:
            break
        fi = feats[i]
        if cfeats:
            d = torch.stack([(fi - cf).pow(2).mean().sqrt() for cf in cfeats])
            if float(d.min()) < dedup_eps:
                continue
        chosen.append(i)
        cfeats.append(fi)
    sel_local = torch.tensor(chosen, dtype=torch.long)
    return idx[sel_local], idx, sel_local


# ----------------------------------------------------------------------------
# vertex-purity metric: convex-hull (simplex) reconstruction residual.
# High residual = true vertex (can't be blended from the others); ~0 = interior.
# ----------------------------------------------------------------------------
def convex_hull_purity(Y, sel_local, m_nn=16):
    from scipy.optimize import nnls
    Yn = Y.detach().cpu().numpy().astype(np.float64)
    sel = sel_local.detach().cpu().numpy()
    c = Yn.mean(0)
    BIG = 1e3
    pur = np.zeros(len(sel))
    for t, kk in enumerate(sel):
        others = sel[sel != kk]
        if others.size == 0:
            pur[t] = 1.0
            continue
        d = ((Yn[others] - Yn[kk]) ** 2).sum(1)
        nn = others[np.argsort(d)[:min(m_nn, others.size)]]
        A = Yn[nn].T                                   # (dim, m)
        b = Yn[kk]
        # simplex-constrained NLS: min ||A w - b|| s.t. w>=0, sum w=1
        Aa = np.vstack([A, BIG * np.ones((1, A.shape[1]))])
        ba = np.concatenate([b, [BIG]])
        try:
            w, _ = nnls(Aa, ba)
        except Exception:
            w = np.zeros(A.shape[1])
        resid = float(np.linalg.norm(A @ w - b))
        denom = float(np.linalg.norm(b - c)) + 1e-9
        pur[t] = resid / denom
    return pur


# ----------------------------------------------------------------------------
# per-model + driver
# ----------------------------------------------------------------------------
def denoise_selection(ratio, idx, Y, sel_local, max_k=5):
    """Within-identity averaging (sqrt-k denoise). For each SPA vertex, average it
    with its <=max_k nearest rows in the subspace (same face + independent noise ->
    the true face survives, grain cancels). A vertex's nearest neighbours are its
    own noisy siblings, so this de-grains WITHOUT blending distinct identities."""
    reps = []
    counts = []
    for li in sel_local.tolist():
        d = (Y - Y[li]).pow(2).sum(1).sqrt()
        order = torch.argsort(d)[:max_k]            # nearest incl. self
        rows = ratio[idx[order]]                    # (k,12288)
        reps.append(rows.mean(0))
        counts.append(int(order.numel()))
    imgs = torch.stack(reps).reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W)
    return imgs, counts


def run_model(name, n=128, do_peel=False, denoise=False, verbose=True):
    gW, gb, ratio, reliable = load_ratio(name)
    scale = SCALE.get(name, "clamp")

    sel_g, idx, Y, sel_l = spa_select(ratio, gW, reliable, n=n)
    if denoise:
        picks, dcounts = denoise_selection(ratio, idx, Y, sel_l)
    else:
        picks = ratio[sel_g].reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W)
    imgs = U.per_image_minmax(picks).clamp(0, 1) if scale == "minmax" else picks.clamp(0, 1)

    # purity: SPA picks vs OLD quality picks, both on the same ratio cloud (in Y space)
    pur_spa = convex_hull_purity(Y, sel_l)
    q_g, q_idx, q_l = quality_select(ratio, reliable, n=n)
    # map quality picks into the SAME subspace coords used for SPA purity
    q_rows = ratio[q_g]
    k = min(128, gW.shape[0])
    V, _ = _topk_subspace(gW, k)
    Yq = q_rows @ V.t()
    pur_q = convex_hull_purity(Yq, torch.arange(Yq.shape[0]))

    tv_mean = float(U.total_variation(imgs).mean())
    info = {
        "name": name, "activation": U.load_grad(name).get("activation"),
        "scale": scale, "denoise": denoise,
        "n_reliable_facelike_rows": int(idx.numel()),
        "n_selected": int(imgs.shape[0]),
        "spa_vertices_found": int(sel_l.numel()),
        "tv_mean_selected": tv_mean,
        "purity_SPA_mean": float(np.mean(pur_spa)), "purity_SPA_min": float(np.min(pur_spa)),
        "purity_OLDquality_mean": float(np.mean(pur_q)), "purity_OLDquality_min": float(np.min(pur_q)),
        "purity_delta_mean": float(np.mean(pur_spa) - np.mean(pur_q)),
    }
    if denoise:
        info["denoise_neighbors_mean"] = float(np.mean(dcounts))
    # save artifacts (new namespace dir; never touches candidates_v3)
    out_dir = U.work_path("candidates_arch", name)
    U.ensure_dir(out_dir)
    tag = "denoised" if denoise else "archetypal"
    torch.save({"images": imgs, "info": info}, os.path.join(out_dir, f"selected_128_{tag}.pt"))
    U.save_image_grid(imgs, os.path.join(out_dir, f"grid_{tag}.png"), ncol=16)
    if not denoise:
        q_imgs = ratio[q_g].reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W).clamp(0, 1)
        U.save_image_grid(q_imgs, os.path.join(out_dir, "grid_oldquality.png"), ncol=16)
    if verbose:
        print(json.dumps(info), flush=True)
    return imgs, info


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    denoise = "--denoised" in sys.argv
    names = args or LEAKY_MLP
    base = {}
    report = {}
    for name in names:
        imgs, info = run_model(name, denoise=denoise)
        base[name] = imgs.float().clamp(0, 1).contiguous()
        report[name] = info
    # base file holds ONLY the changed leaky-MLP models (per the brief)
    fn = "base_archetypal_denoised.pt" if denoise else "base_archetypal.pt"
    rn = "purity_report_denoised.json" if denoise else "purity_report.json"
    out = U.work_path("outputs_final", fn)
    torch.save(base, out)
    U.save_json(report, U.work_path("reports", "arch", rn))
    print("WROTE", out, flush=True)
    print("REPORT", U.work_path("reports", "arch", rn), flush=True)
    # quick sanity: shapes + range + uniqueness
    for k, v in base.items():
        assert v.shape == (128, 3, 64, 64), (k, v.shape)
        assert torch.isfinite(v).all() and float(v.min()) >= 0 and float(v.max()) <= 1
        u = torch.unique(v.reshape(128, -1), dim=0).shape[0]
        print(f"  {k}: shape ok, unique_rows={u}, tv_mean={float(U.total_variation(v).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()

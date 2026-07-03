"""a1: SVD subspace projection for MLP models (1, 4, 5, 8).

KEY INSIGHT (from the fresh eigenface analysis):
The first-layer weight gradient gW (1024 x 12288) has rank <= 128 (the batch
size). Its top-128 right singular vectors span the IMAGE SUBSPACE — the linear
span of the 128 true input images. The remaining 128..1024 singular components
are noise (activation bleed, multi-sample mixing).

The existing ratio-row approach computes gW[row] / gb[row] per neuron, but each
row lives in the FULL 12288-dimensional space. By PROJECTING every ratio-row
onto the top-K right-singular subspace of gW, we:

  1. Strip the noise-floor components (singular indices 128..1024)
  2. Preserve all true image signal (the rows are linear combinations of the
     128 images, which by definition lie in the top-128 subspace)
  3. Denoise mixed-neuron reconstructions without blurring (subspace projection
     is linear, not spatial averaging)

This is especially impactful for model8 (GELU) where the smooth activation
creates a 5.75% noise floor — many neurons fire for multiple samples, making
ratio-rows noisy mixtures. Subspace projection removes the noise components
that are orthogonal to the true image subspace.

ADDITIONAL: activation-aware ratio correction for non-ReLU models. For GELU/
sigmoid/tanh, the bias gradient gb[j] includes the activation derivative
act'(z_j), biasing the ratio. We optionally correct for this.

Writes to candidates_a1/<model>/. Never touches existing candidates or outputs.
Login-node safe (SVD of 1024x12288 is bounded).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from skimage.metrics import structural_similarity as ssim

import utils as U

MLP_MODELS = ["model8", "model1", "model4", "model5"]
EPS = 1e-8


# ---------------------------------------------------------------------------
# Activation-aware helpers
# ---------------------------------------------------------------------------
def gelu_deriv(x):
    """Approximate derivative of GELU at x."""
    # GELU(x) = x * Phi(x), GELU'(x) = Phi(x) + x * phi(x)
    # where Phi = standard normal CDF, phi = standard normal PDF
    phi = torch.exp(-0.5 * x**2) / (2.0 * 3.14159265358979) ** 0.5
    Phi = 0.5 * (1.0 + torch.erf(x / 2.0**0.5))
    return Phi + x * phi


def sigmoid_deriv(x):
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def tanh_deriv(x):
    return 1.0 - torch.tanh(x) ** 2


DERIV_FN = {
    "gelu": gelu_deriv,
    "sigmoid": sigmoid_deriv,
    "tanh": tanh_deriv,
    "relu": None,  # binary; no smooth correction possible
}


# ---------------------------------------------------------------------------
# Core: first-layer identification
# ---------------------------------------------------------------------------
def first_image_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2 and w.shape[1] == U.IMG_NUMEL:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


# ---------------------------------------------------------------------------
# Core: ratio computation with optional activation correction
# ---------------------------------------------------------------------------
def ratio_rows(gW, gb, activation="relu"):
    """Compute gW[row] / gb[row] with optional activation-derivative correction.

    For non-ReLU activations, gb[j] = sum_n c_{n,j} where c_{n,j} includes
    act'(z_j). We estimate the activation derivative from the magnitude of gb
    itself and partially correct for it, making the ratio a more equal-weighted
    average of the inputs (which improves single-sample-dominant detection).
    """
    gb = gb.float()
    gW = gW.float()
    reliable = gb.abs() > EPS

    # Optional: activation-aware correction
    deriv_fn = DERIV_FN.get(activation)
    if deriv_fn is not None and activation != "relu":
        # We don't have the actual pre-activation z_j, but gb magnitude is a
        # proxy for the activation derivative magnitude. For GELU, large |gb|
        # implies the neuron is in the linear regime (act' ≈ 1), small |gb|
        # implies it's in the saturation regime (act' ≈ 0). We apply a soft
        # correction: scale gb by a factor that reduces the impact of the
        # activation derivative variation.
        #
        # Heuristic: act'(z) ≈ |gb| / median(|gb|) for the reliable neurons,
        # then correct by dividing out this estimated derivative.
        gb_abs = gb.abs()
        median_gb = gb_abs[reliable].median().clamp(min=EPS)
        # Soft correction factor: blend between raw and corrected
        # to avoid amplifying noise in the tails
        correction = (gb_abs / median_gb).clamp(0.1, 10.0)
        gb_corrected = gb * correction
        denom = torch.where(gb_corrected.abs() > EPS, gb_corrected,
                            torch.full_like(gb_corrected, float("nan")))
    else:
        denom = torch.where(gb.abs() > EPS, gb,
                            torch.full_like(gb, float("nan")))

    ratio = gW / denom.unsqueeze(1)
    return ratio, reliable


# ---------------------------------------------------------------------------
# Core: SVD subspace projection
# ---------------------------------------------------------------------------
def svd_subspace_project(gW, ratio, k=128, verbose=True):
    """Project ratio-row reconstructions onto the top-k right singular subspace
    of gW, stripping noise-floor components.

    Args:
        gW: (out_features, in_features) weight gradient
        ratio: (out_features, in_features) ratio-row images
        k: subspace dimension (default 128 = batch size)
        verbose: print SVD diagnostics

    Returns:
        projected: (out_features, in_features) projected ratio-rows
        svd_info: dict with diagnostics
    """
    gW = gW.float()
    out, inp = gW.shape

    # Economy SVD (bounded: 1024 x 12288 → takes ~1 sec on CPU)
    U_, S, Vh = torch.linalg.svd(gW, full_matrices=False)
    S = S.cpu()
    s0 = S[0].item()

    # Determine effective rank (1% of max singular value)
    eff_rank = int((S > 0.01 * s0).sum().item())

    # Determine optimal k: use eff_rank if it's close to 128, else 128
    # The key insight: if eff_rank ≈ 128 (sharp drop), use exactly 128.
    # If eff_rank < 128 (noisy model), use eff_rank to avoid projecting
    # onto noise components. If eff_rank >> 128, still use 128.
    k_actual = min(k, eff_rank)

    # Singular value diagnostics
    def svr(i):
        return round((S[i] / (s0 + 1e-12)).item(), 4) if i < S.numel() else None

    svd_info = {
        "eff_rank_1pct": eff_rank,
        "k_used": k_actual,
        "s0": round(s0, 4),
        "sv_ratio@64": svr(64),
        "sv_ratio@127": svr(127),
        "sv_ratio@128": svr(128),
        "sv_ratio@130": svr(130),
        "sv_ratio@200": svr(200),
        "energy_in_top_k": round(
            float((S[:k_actual] ** 2).sum() / (S ** 2).sum()), 4),
    }

    if verbose:
        print(f"  SVD: eff_rank={eff_rank} k_used={k_actual} "
              f"s@127/s@128={svr(127)}/{svr(128)} "
              f"energy_in_top_k={svd_info['energy_in_top_k']}")

    # Top-k right singular vectors (the image subspace basis)
    Vk = Vh[:k_actual]  # (k_actual, in_features)

    # Project each ratio-row: x_proj = (x @ Vk.T) @ Vk
    ratio_flat = ratio.float().reshape(out, inp)
    # Handle NaNs: replace with 0 before projection, restore NaN mask after
    nan_mask = torch.isnan(ratio_flat)
    ratio_safe = torch.nan_to_num(ratio_flat, nan=0.0)

    # Projection: coefficients in the subspace, then reconstruct
    coeffs = ratio_safe @ Vk.T           # (out, k_actual)
    projected = coeffs @ Vk              # (out, in_features)

    # Restore NaN for rows that were entirely NaN
    all_nan_rows = nan_mask.all(dim=1)
    projected[all_nan_rows] = float("nan")

    return projected, svd_info


# ---------------------------------------------------------------------------
# Image quality helpers (from attack_mlp_exact_v2)
# ---------------------------------------------------------------------------
def gray_ds(imgs, size=24):
    g = imgs.mean(1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1).numpy().astype(np.float64)


def ssim_img(a, b):
    return float(ssim(a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy(),
                      channel_axis=2, data_range=1.0))


def trimmed_mean(members, feats, medoid_local):
    d = np.linalg.norm(feats - feats[medoid_local], axis=1)
    thr = np.quantile(d, 0.5) if len(d) > 2 else np.inf
    keep = np.where(d <= thr)[0]
    if len(keep) == 0:
        keep = np.array([medoid_local])
    return members[keep].mean(0).clamp(0, 1)


# ---------------------------------------------------------------------------
# Main per-model pipeline
# ---------------------------------------------------------------------------
def run_model(name, svd_k=128, tau_q=0.35, max_k=128, use_activation_correction=True):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    activation = grad.get("activation", "relu")
    base = first_image_linear(state, g)
    out_dir = U.work_path("candidates_a1", name)
    U.ensure_dir(out_dir)

    if base is None:
        print(f"{name}: no 12288-input linear; skip")
        return None

    gW, gb = g[base + ".weight"], g[base + ".bias"]
    print(f"\n{'='*60}")
    print(f"{name} (activation={activation})")
    print(f"{'='*60}")

    # Step 1: Compute ratio rows (with optional activation correction)
    act_for_ratio = activation if use_activation_correction else "relu"
    ratio, reliable = ratio_rows(gW, gb, activation=act_for_ratio)
    rows_all = ratio.shape[0]

    # Step 2: SVD subspace projection
    projected, svd_info = svd_subspace_project(gW, ratio, k=svd_k)

    # Step 3: Compare projected vs unprojected image quality
    imgs_raw = ratio.reshape(rows_all, U.IMG_C, U.IMG_H, U.IMG_W)
    imgs_proj = projected.reshape(rows_all, U.IMG_C, U.IMG_H, U.IMG_W)

    # Quality stats for both
    st_raw = U.image_stats(imgs_raw)
    st_proj = U.image_stats(imgs_proj)

    # Report improvement
    raw_good = (reliable & st_raw["finite"] &
                (st_raw["rng_frac"] > 0.6) & (st_raw["std"] > 0.02))
    proj_good = (reliable & st_proj["finite"] &
                 (st_proj["rng_frac"] > 0.6) & (st_proj["std"] > 0.02))
    print(f"  Raw  good rows: {int(raw_good.sum())}/{rows_all}")
    print(f"  Proj good rows: {int(proj_good.sum())}/{rows_all}")
    print(f"  Raw  mean TV: {float(st_raw['tv'][raw_good].mean()):.4f}" if raw_good.any() else "  Raw  mean TV: N/A")
    print(f"  Proj mean TV: {float(st_proj['tv'][proj_good].mean()):.4f}" if proj_good.any() else "  Proj mean TV: N/A")

    # Use projected images for clustering
    keep_mask = proj_good
    idx = torch.where(keep_mask)[0]
    n_valid = int(idx.numel())
    if n_valid < 2:
        # Fallback to raw if projection filtered too aggressively
        keep_mask = raw_good
        idx = torch.where(keep_mask)[0]
        n_valid = int(idx.numel())
        print(f"  [fallback] projection too aggressive, using raw ({n_valid} rows)")
        imgs = imgs_raw[idx].clamp(0, 1)
    else:
        imgs = imgs_proj[idx].clamp(0, 1)

    if n_valid < 2:
        print(f"{name}: only {n_valid} valid rows; skip")
        return None

    # Step 4: Clustering (same as attack_mlp_exact_v2 but on projected images)
    feats = gray_ds(imgs, 24)

    from scipy.spatial.distance import cdist
    D = cdist(feats, feats)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    tau = float(np.quantile(nn, tau_q)) * 1.5 + 1e-6
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=tau,
                                  linkage="average", metric="euclidean").fit(feats)
    labels = agg.labels_
    n_clusters = labels.max() + 1

    # Step 5: Per-cluster scoring + representative extraction
    clusters = []
    for c in range(n_clusters):
        loc = np.where(labels == c)[0]
        members = imgs[loc]
        fsub = feats[loc]
        if len(loc) == 1:
            medoid_local = 0
            consistency = 0.0
        else:
            sub = cdist(fsub, fsub)
            medoid_local = int(sub.sum(1).argmin())
            med_img = members[medoid_local]
            consistency = float(np.mean([ssim_img(members[k], med_img)
                                         for k in range(len(loc))]))
        medoid_img = members[medoid_local]
        if len(loc) >= 3 and consistency > 0.5:
            rep = trimmed_mean(members, fsub, medoid_local)
            rep_kind = "trimmed_mean"
        else:
            rep = medoid_img.clamp(0, 1)
            rep_kind = "medoid"
        rst = U.image_stats(rep.unsqueeze(0))
        rng = float(rst["rng_frac"][0])
        std = float(rst["std"][0])
        tv = float(rst["tv"][0])
        reject = (rng < 0.85) or (std < 0.05) or (tv > 0.30)
        conf = (min(len(loc) / 5.0, 1.0) * 0.35 + consistency * 0.35
                + rng * 0.15 + min(std / 0.2, 1.0) * 0.15 - max(tv - 0.15, 0) * 1.0)
        clusters.append({
            "rep": rep, "size": int(len(loc)), "consistency": consistency,
            "rep_kind": rep_kind, "rng": rng, "std": std, "tv": tv,
            "reject": bool(reject), "conf": float(conf),
            "rows": idx[loc].tolist(),
        })

    good = [c for c in clusters if not c["reject"]]
    good.sort(key=lambda c: c["conf"], reverse=True)
    rejected = [c for c in clusters if c["reject"]]

    # Step 6: Build pool of representatives
    reps_sorted = sorted(clusters, key=lambda c: c["conf"], reverse=True)
    reps = torch.stack([c["rep"] for c in reps_sorted]).float()

    # Step 7: Also build un-projected versions for comparison
    imgs_raw_pool = imgs_raw[idx].clamp(0, 1)

    # Step 8: MERGE with v2 and/or v1 candidates (if they exist)
    import postprocess as PP
    v2p = U.work_path("candidates_v2", name, "mlp_exact_selected_128.pt")
    v1p = U.work_path("candidates", name, "selected_128.pt")
    extras = []
    extra_labels = []
    if os.path.exists(v2p):
        v2 = torch.load(v2p, weights_only=False)["images"].float().clamp(0, 1)
        extras.append(v2)
        extra_labels.append("v2")
    if os.path.exists(v1p):
        v1 = torch.load(v1p, weights_only=False)["images"].float().clamp(0, 1)
        extras.append(v1)
        extra_labels.append("v1")

    if extras:
        pool = torch.cat([reps] + extras, 0)
    else:
        pool = reps

    sel, chosen_idx = PP.select_128(pool, n=max_k, denoise=False)
    sel = sel.float()
    ci = torch.tensor(chosen_idx)
    n_from_a1 = int((ci < reps.shape[0]).sum())

    # Step 9: Save outputs
    torch.save({"images": sel, "n_from_a1": n_from_a1,
                "n_clusters": int(n_clusters), "confident": len(good),
                "svd_info": svd_info,
                "activation_correction": use_activation_correction,
                "activation": activation},
               os.path.join(out_dir, "svd_selected_128.pt"))
    torch.save({"clusters": [{"rep": c["rep"], **{k: v for k, v in c.items() if k != "rep"}}
                             for c in clusters]},
               os.path.join(out_dir, "svd_candidates.pt"))

    U.save_image_grid(sel, os.path.join(out_dir, "grid_svd_selected.png"), ncol=16)
    U.save_image_grid(reps[:128], os.path.join(out_dir, "grid_svd_all_reps.png"), ncol=16)

    # Step 10: Comparison grids — projected vs raw (first 32 of each)
    if n_valid >= 32:
        comp_raw = imgs_raw[idx[:32]].clamp(0, 1)
        comp_proj = imgs[:32].clamp(0, 1)
        # Top row: raw, bottom row: projected
        comp = torch.cat([comp_raw[:16], comp_proj[:16]], 0)
        U.save_image_grid(comp, os.path.join(out_dir, "compare_raw_vs_projected.png"), ncol=16)

    meta = {
        "name": name, "activation": activation, "target_layer": base,
        "rows_total": rows_all, "rows_valid_raw": int(raw_good.sum()),
        "rows_valid_projected": int(proj_good.sum()),
        "svd": svd_info,
        "tau": tau, "n_clusters": int(n_clusters),
        "confident_clusters": len(good), "rejected_clusters": len(rejected),
        "selected": int(sel.shape[0]),
        "final_from_a1": n_from_a1,
        "final_from_existing": int(max_k - n_from_a1),
        "trimmed_mean_reps": int(sum(1 for c in clusters if c["rep_kind"] == "trimmed_mean")),
        "cluster_size_hist": np.bincount([c["size"] for c in clusters]).tolist(),
        "top_conf": [round(c["conf"], 3) for c in good[:10]],
        "activation_correction": use_activation_correction,
    }
    U.save_json(meta, os.path.join(out_dir, "svd_metadata.json"))
    print(f"\n{name}: valid={n_valid}/{rows_all} clusters={n_clusters} "
          f"confident={len(good)} | final 128 = {n_from_a1} a1-svd + "
          f"{max_k - n_from_a1} existing -> {out_dir}")
    return meta


def main():
    names = sys.argv[1:] or MLP_MODELS
    summary = {}
    for name in names:
        try:
            summary[name] = run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(summary, U.work_path("candidates_a1", "_svd_summary.json"))


if __name__ == "__main__":
    main()

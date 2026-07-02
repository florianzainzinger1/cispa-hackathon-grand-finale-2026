"""v2: single-sample-neuron reconstruction for MLP models (1,4,5,8).

For the first Linear (12288->1024):  ratio[row] = gradW[row]/gradB[row] is a
weighted average of the images that drive that neuron. If a neuron is activated
by ONE sample, the ratio is that exact image. Many neurons fire for the same
sample, so genuine single-sample reconstructions form TIGHT clusters of
near-identical images; mixtures are inconsistent/outliers. We exploit that:

  1. ratio rows -> filter reliable/finite/in-range
  2. agglomerative cluster on 24x24 gray (data-driven threshold)
  3. score clusters by support + intra-cluster consistency + image quality
  4. representative = trimmed-mean (tight clusters, denoise) or medoid
  5. reject mixtures (low contrast / out-of-range / high TV / low consistency)
  6. diverse select up to 128 (fill with best remaining to reach exactly 128)

READ-ONLY w.r.t. v1: reads v1 selected_128 only for the compare grid. All
outputs go to candidates_v2/. Never writes candidates/ or outputs/.
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

MLP_DEFAULT = [8, 1, 4, 5]           # model8 first (weakest, most headroom)
EPS = 1e-8


def first_image_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2 and w.shape[1] == U.IMG_NUMEL:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def ratio_rows(gW, gb):
    gb = gb.float()
    gW = gW.float()
    reliable = gb.abs() > EPS
    denom = torch.where(gb.abs() > EPS, gb, torch.full_like(gb, float("nan")))
    ratio = gW / denom.unsqueeze(1)
    return ratio, reliable


def gray_ds(imgs, size=24):
    g = imgs.mean(1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1).numpy().astype(np.float64)


def ssim_img(a, b):
    return float(ssim(a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy(),
                      channel_axis=2, data_range=1.0))


def trimmed_mean(members, feats, medoid_local):
    """Mean of members within the median distance of the medoid (denoise)."""
    d = np.linalg.norm(feats - feats[medoid_local], axis=1)
    thr = np.quantile(d, 0.5) if len(d) > 2 else np.inf
    keep = np.where(d <= thr)[0]
    if len(keep) == 0:
        keep = np.array([medoid_local])
    return members[keep].mean(0).clamp(0, 1)


def run_model(name, tau_q=0.35, max_k=128):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    base = first_image_linear(state, g)
    out_dir = U.work_path("candidates_v2", name)
    U.ensure_dir(out_dir)
    if base is None:
        print(f"{name}: no 12288-input linear; skip")
        return None

    gW, gb = g[base + ".weight"], g[base + ".bias"]
    ratio, reliable = ratio_rows(gW, gb)
    rows_all = ratio.shape[0]
    imgs_raw = ratio.reshape(rows_all, U.IMG_C, U.IMG_H, U.IMG_W)
    st = U.image_stats(imgs_raw)

    # keep reliable, finite, mostly in-range rows for clustering
    keep_mask = reliable & st["finite"] & (st["rng_frac"] > 0.6) & (st["std"] > 0.02)
    idx = torch.where(keep_mask)[0]
    n_valid = int(idx.numel())
    if n_valid < 2:
        print(f"{name}: only {n_valid} valid rows; skip")
        return None
    imgs = imgs_raw[idx].clamp(0, 1)             # (n_valid,3,64,64)
    feats = gray_ds(imgs, 24)

    # data-driven agglomerative threshold from nearest-neighbour distances
    from scipy.spatial.distance import cdist
    D = cdist(feats, feats)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    tau = float(np.quantile(nn, tau_q)) * 1.5 + 1e-6
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=tau,
                                  linkage="average", metric="euclidean").fit(feats)
    labels = agg.labels_
    n_clusters = labels.max() + 1

    # per-cluster: medoid, consistency, trimmed-mean rep, quality, confidence
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
        # representative: trimmed mean if tight+supported, else medoid
        if len(loc) >= 3 and consistency > 0.5:
            rep = trimmed_mean(members, fsub, medoid_local)
            rep_kind = "trimmed_mean"
        else:
            rep = medoid_img.clamp(0, 1)
            rep_kind = "medoid"
        rst = U.image_stats(rep.unsqueeze(0))
        rng = float(rst["rng_frac"][0]); std = float(rst["std"][0]); tv = float(rst["tv"][0])
        # mixture rejection
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

    # v2 clean-rep bank (denoised medoids/trimmed-means, cleanest first)
    reps_sorted = sorted(clusters, key=lambda c: c["conf"], reverse=True)
    reps = torch.stack([c["rep"] for c in reps_sorted]).float()   # (n_clusters,3,64,64)

    # MERGE with v1 and pick best-of-both. v1's 128 are in the pool, so the
    # quality+diversity selection can only match or beat v1, and always yields
    # 128 DISTINCT images (fixes the padding/diversity problem).
    import postprocess as PP
    v1p = U.work_path("candidates", name, "selected_128.pt")
    v1 = None
    if os.path.exists(v1p):
        v1 = torch.load(v1p, weights_only=False)["images"].float().clamp(0, 1)
        pool = torch.cat([reps, v1], 0)
    else:
        pool = reps
    sel, chosen_idx = PP.select_128(pool, n=max_k, denoise=False)
    sel = sel.float()
    ci = torch.tensor(chosen_idx)
    n_from_v2 = int((ci < reps.shape[0]).sum())

    # ---- save ----
    torch.save({"images": sel, "n_from_v2": n_from_v2,
                "n_clusters": int(n_clusters), "confident": len(good)},
               os.path.join(out_dir, "mlp_exact_selected_128.pt"))
    torch.save({"clusters": [{"rep": c["rep"], **{k: v for k, v in c.items() if k != "rep"}}
                             for c in clusters]},
               os.path.join(out_dir, "mlp_exact_candidates.pt"))

    U.save_image_grid(sel, os.path.join(out_dir, "mlp_exact_grid_selected.png"), ncol=16)
    U.save_image_grid(reps[:128], os.path.join(out_dir, "mlp_exact_grid_all.png"), ncol=16)

    # compare v1 (top) vs merged v2 (bottom); v1 read-only
    if v1 is not None:
        comp = torch.cat([v1[:32], sel[:32]], 0)
        U.save_image_grid(comp, os.path.join(out_dir, "compare_v1_vs_v2.png"), ncol=16)

    meta = {
        "name": name, "target_layer": base,
        "rows_total": rows_all, "rows_valid": n_valid,
        "tau": tau, "n_clusters": int(n_clusters),
        "confident_clusters": len(good), "rejected_clusters": len(rejected),
        "selected": int(sel.shape[0]),
        "final_from_v2_reps": n_from_v2, "final_kept_from_v1": int(max_k - n_from_v2),
        "trimmed_mean_reps": int(sum(1 for c in clusters if c["rep_kind"] == "trimmed_mean")),
        "cluster_size_hist": np.bincount([c["size"] for c in clusters]).tolist(),
        "top_conf": [round(c["conf"], 3) for c in good[:10]],
    }
    U.save_json(meta, os.path.join(out_dir, "mlp_exact_metadata.json"))
    print(f"{name}: valid={n_valid}/{rows_all} clusters={n_clusters} "
          f"confident={len(good)} | final 128 = {n_from_v2} v2-clean + "
          f"{max_k - n_from_v2} v1 -> {out_dir}")
    return meta


def main():
    names = sys.argv[1:] or [f"model{i}" for i in MLP_DEFAULT]
    summary = {}
    for name in names:
        try:
            summary[name] = run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(summary, U.work_path("candidates_v2", "_summary.json"))


if __name__ == "__main__":
    main()

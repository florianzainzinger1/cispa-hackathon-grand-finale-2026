"""v2 investigation (READ-ONLY): are the 128 private identities shared across models?

Loads the final per-model images from submission.pt (== each model's selected_128
for the 10 real models; == face-bank for ViT 9/11, which is therefore CIRCULAR and
excluded from identity conclusions). Computes, for every model pair, a Hungarian
one-to-one assignment and the matched-pair similarity, plus a within-model
"nearest different image" baseline that discriminates shared-identity from mere
distribution similarity.

Writes ONLY new files under reports/ and reports/v2_grids/. Never touches
submission.pt, submission_v1_bank.pt, or any candidates.
"""
import os
import json
import itertools
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim

import utils as U

REAL = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]     # have independent reconstructions
BANK_CIRCULAR = [9, 11]                       # submission imgs are the strong bank
STRONG = [1, 4, 5, 6, 10, 12]
WEAK_TESTABLE = [2, 3, 7, 8]                   # blurry/noisy but independent


def load_all():
    sub = torch.load(U.work_path("outputs", "submission.pt"),
                     weights_only=False, map_location="cpu")
    return {i: sub[f"model{i}"].float().clamp(0, 1) for i in range(1, 13)}


def flat(x):
    return x.reshape(x.shape[0], -1).numpy().astype(np.float64)


def gray_ds(x, size=16):
    g = x.mean(1, keepdim=True)
    g = torch.nn.functional.interpolate(g, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return g.reshape(g.shape[0], -1).numpy().astype(np.float64)


def ssim_pair(a, b):
    # a,b: (3,64,64) tensors in [0,1]
    A = a.permute(1, 2, 0).numpy()
    B = b.permute(1, 2, 0).numpy()
    return float(ssim(A, B, channel_axis=2, data_range=1.0))


def match(imgsA, imgsB, cost):
    r, c = linear_sum_assignment(cost)
    return r, c


def pair_metrics(A, B):
    """Hungarian match on pixel MSE; report matched-pair MSE + SSIM (full & 16x16)."""
    fa, fb = flat(A), flat(B)
    D = fa.shape[1]
    cost = cdist(fa, fb, "sqeuclidean") / D          # 128x128 pixel MSE
    r, c = match(A, B, cost)
    matched_mse = float(cost[r, c].mean())
    ssims = np.array([ssim_pair(A[i], B[j]) for i, j in zip(r, c)])
    # coarse SSIM (blur-robust): downsample to 16x16 then SSIM
    A16 = torch.nn.functional.interpolate(A, size=(16, 16), mode="bilinear", align_corners=False)
    B16 = torch.nn.functional.interpolate(B, size=(16, 16), mode="bilinear", align_corners=False)
    ssims16 = np.array([ssim_pair(A16[i].clamp(0, 1), B16[j].clamp(0, 1)) for i, j in zip(r, c)])
    return {
        "matched_mse": matched_mse,
        "ssim_mean": float(ssims.mean()), "ssim_median": float(np.median(ssims)),
        "ssim16_mean": float(ssims16.mean()),
        "conf_ssim_gt05": int((ssims > 0.5).sum()),
        "conf_ssim_gt07": int((ssims > 0.7).sum()),
        "conf_mse_lt002": int((cost[r, c] < 0.002).sum()),
        "r": r.tolist(), "c": c.tolist(),
        "ssims": ssims.tolist(),
    }


def within_baseline(A):
    """Each image's SSIM to its NEAREST DIFFERENT image in the same set.
    This is the 'closest different person, same distribution' reference."""
    fa = flat(A)
    D = fa.shape[1]
    cost = cdist(fa, fa, "sqeuclidean") / D
    np.fill_diagonal(cost, np.inf)
    nn = cost.argmin(1)
    ss = np.array([ssim_pair(A[i], A[j]) for i, j in enumerate(nn)])
    mse = np.array([cost[i, nn[i]] for i in range(len(nn))])
    return {"ssim_mean": float(ss.mean()), "mse_mean": float(mse.mean())}


def save_match_grid(A, B, r, c, ssims, path, k=16, mode="top"):
    idx = np.argsort(ssims)
    sel = idx[::-1][:k] if mode == "top" else idx[:k]
    rows = []
    for s in sel:
        rows.append(A[r[s]]); rows.append(B[c[s]])
    grid = torch.stack(rows)  # [a1,b1,a2,b2,...]  ncol=8 -> 4 pairs/row
    U.save_image_grid(grid, path, ncol=8)


def main():
    imgs = load_all()
    outdir = U.work_path("reports")
    grids = U.work_path("reports", "v2_grids")
    U.ensure_dir(grids)

    # within-model baselines (distribution reference)
    baseline = {i: within_baseline(imgs[i]) for i in range(1, 13)}

    # 12x12 matrices (SSIM, coarse SSIM, MSE, confident counts)
    n = 12
    M_ssim = np.full((n, n), np.nan)
    M_ssim16 = np.full((n, n), np.nan)
    M_mse = np.full((n, n), np.nan)
    M_conf05 = np.zeros((n, n))
    pair_detail = {}
    for a, b in itertools.combinations(range(1, 13), 2):
        pm = pair_metrics(imgs[a], imgs[b])
        M_ssim[a - 1, b - 1] = M_ssim[b - 1, a - 1] = pm["ssim_mean"]
        M_ssim16[a - 1, b - 1] = M_ssim16[b - 1, a - 1] = pm["ssim16_mean"]
        M_mse[a - 1, b - 1] = M_mse[b - 1, a - 1] = pm["matched_mse"]
        M_conf05[a - 1, b - 1] = M_conf05[b - 1, a - 1] = pm["conf_ssim_gt05"]
        pair_detail[f"{a}-{b}"] = {k: v for k, v in pm.items() if k not in ("r", "c", "ssims")}

    # detailed key pairs + grids (both best and random samples, anti-cherry-pick)
    key_pairs = [(1, 4), (1, 5), (4, 5), (1, 6), (1, 10), (6, 10), (1, 8), (1, 2)]
    for a, b in key_pairs:
        pm = pair_metrics(imgs[a], imgs[b])
        r, c, ss = np.array(pm["r"]), np.array(pm["c"]), np.array(pm["ssims"])
        save_match_grid(imgs[a], imgs[b], r, c, ss,
                        os.path.join(grids, f"match_{a}_{b}_top.png"), mode="top")
        save_match_grid(imgs[a], imgs[b], r, c, ss,
                        os.path.join(grids, f"match_{a}_{b}_rand.png"), mode="worst")

    # strong-pool vs weak testable models
    pool = torch.cat([imgs[i] for i in STRONG], 0)      # 768
    fp = flat(pool)
    Dp = fp.shape[1]
    pool_analysis = {}
    for w in WEAK_TESTABLE:
        fw = flat(imgs[w])
        cost = cdist(fw, fp, "sqeuclidean") / Dp        # 128 x 768
        nn = cost.argmin(1)
        ss = np.array([ssim_pair(imgs[w][i], pool[nn[i]]) for i in range(128)])
        # blur-robust coarse ssim
        w16 = torch.nn.functional.interpolate(imgs[w], size=(16, 16), mode="bilinear", align_corners=False).clamp(0, 1)
        p16 = torch.nn.functional.interpolate(pool, size=(16, 16), mode="bilinear", align_corners=False).clamp(0, 1)
        ss16 = np.array([ssim_pair(w16[i], p16[nn[i]]) for i in range(128)])
        pool_analysis[f"model{w}"] = {
            "nn_ssim_mean": float(ss.mean()),
            "nn_ssim16_mean": float(ss16.mean()),
            "nn_mse_mean": float(cost[np.arange(128), nn].mean()),
            "baseline_within_ssim": baseline[w]["ssim_mean"],
            "conf_ssim_gt05": int((ss > 0.5).sum()),
        }
        # grid: weak image beside its nearest strong-pool image (top by coarse ssim)
        order = np.argsort(ss16)[::-1][:16]
        rows = []
        for o in order:
            rows.append(imgs[w][o]); rows.append(pool[nn[o]])
        U.save_image_grid(torch.stack(rows), os.path.join(grids, f"pool_vs_model{w}_top.png"), ncol=8)

    # ---- interpretation helpers ----
    strong_pairs = [(a, b) for a, b in itertools.combinations(STRONG, 2)]
    strong_ssim = np.mean([M_ssim[a - 1, b - 1] for a, b in strong_pairs])
    strong_ssim16 = np.mean([M_ssim16[a - 1, b - 1] for a, b in strong_pairs])
    strong_baseline = np.mean([baseline[i]["ssim_mean"] for i in STRONG])
    strong_conf = np.mean([M_conf05[a - 1, b - 1] for a, b in strong_pairs])

    result = {
        "note": "models 9,11 rows/cols use the strong-derived face bank (CIRCULAR) - not identity evidence",
        "baseline_within_model": baseline,
        "matrix_ssim": M_ssim.tolist(),
        "matrix_ssim16": M_ssim16.tolist(),
        "matrix_matched_mse": M_mse.tolist(),
        "matrix_conf_ssim_gt05": M_conf05.tolist(),
        "pair_detail": pair_detail,
        "strong_pool_vs_weak": pool_analysis,
        "summary": {
            "strong_pair_matched_ssim_mean": float(strong_ssim),
            "strong_pair_matched_ssim16_mean": float(strong_ssim16),
            "strong_within_model_baseline_ssim_mean": float(strong_baseline),
            "strong_pair_confident_matches_mean_gt05": float(strong_conf),
            "discriminator_cross_minus_baseline": float(strong_ssim - strong_baseline),
        },
    }
    U.save_json(result, os.path.join(outdir, "shared_identity_test.json"))

    # markdown
    def row(name, vals):
        return "| " + name + " | " + " | ".join(vals) + " |"
    lines = ["# Shared-Identity Test\n",
             f"- strong-pair matched SSIM (mean over {len(strong_pairs)} pairs): **{strong_ssim:.3f}**",
             f"- strong-pair matched SSIM@16x16 (blur-robust): **{strong_ssim16:.3f}**",
             f"- within-model nearest-DIFFERENT-image baseline SSIM: **{strong_baseline:.3f}**",
             f"- discriminator (cross - baseline): **{strong_ssim - strong_baseline:+.3f}**  "
             f"(large positive => shared identities)",
             f"- avg confident matches per strong pair (SSIM>0.5): **{strong_conf:.1f}** / 128\n",
             "## 12x12 matched-SSIM matrix (rows/cols model1..12; 9,11 = bank/circular)\n",
             "| m | " + " | ".join(str(i) for i in range(1, 13)) + " |",
             "|" + "---|" * 13]
    for i in range(12):
        vals = []
        for j in range(12):
            v = M_ssim[i, j]
            vals.append("-" if np.isnan(v) else f"{v:.2f}")
        lines.append(row(str(i + 1), vals))
    lines += ["\n## strong-pool vs weak (independent recon)\n",
              "| weak | nn_ssim | nn_ssim@16 | within-baseline | conf>0.5 |",
              "|---|---|---|---|---|"]
    for w in WEAK_TESTABLE:
        d = pool_analysis[f"model{w}"]
        lines.append(f"| model{w} | {d['nn_ssim_mean']:.3f} | {d['nn_ssim16_mean']:.3f} | "
                     f"{d['baseline_within_ssim']:.3f} | {d['conf_ssim_gt05']} |")
    with open(os.path.join(outdir, "shared_identity_test.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # matrix png (grayscale heat via PIL, no matplotlib)
    try:
        from PIL import Image
        mm = np.nan_to_num(M_ssim, nan=0.0)
        norm = (mm - mm.min()) / (mm.max() - mm.min() + 1e-9)
        im = Image.fromarray((norm * 255).astype(np.uint8)).resize((360, 360), Image.NEAREST)
        im.save(os.path.join(outdir, "shared_identity_matrix.png"))
    except Exception as e:
        print("matrix png skipped:", e)

    print("=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))
    print("strong within-model baselines:",
          {i: round(baseline[i]["ssim_mean"], 3) for i in STRONG})
    print("wrote reports/shared_identity_test.{json,md}, matrix png, and v2_grids/")


if __name__ == "__main__":
    main()

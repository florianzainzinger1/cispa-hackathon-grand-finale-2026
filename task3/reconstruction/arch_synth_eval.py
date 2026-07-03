"""WS1 secondary evidence — synthetic-oracle SSIM: does SPA vertex selection beat
the OLD quality selection on a controlled task with TRUE known ground truth?

Known CelebA faces X -> real leaky-MLP architecture -> synthetic net.0 gradient
(clean, and 'real-like' with calibrated DP noise) -> ratio rows -> {SPA vs OLD}
selection -> per-channel Hungarian SSIM vs the SAME known X.  Non-circular: the GT
here is genuinely known clean faces, so 'SPA higher' = SPA recovers cleaner
individual faces.  Uses the grader-faithful per-channel scorer (ssim_grader).

Still label real-leaderboard impact UNVERIFIED (needs A/B); this is a proxy.
Login-safe.
"""
import sys
import json
import numpy as np
import torch

import synth_lib as L
import rec_lib as R
import archetype_select as A
import ssim_grader as G
import utils as U

LEAKY = [1, 4, 5, 8]


def find_alpha(mi, X, target=0.77):
    best = (0.0, 9.9)
    for a in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        gW, _, _, _ = L.synth_grad(mi, X, noise_alpha=a, seed=1)
        e = L.E128(gW)
        if abs(e - target) < best[1]:
            best = (a, abs(e - target))
    return best[0]


def eval_one(mi, X, conv="perch7"):
    out = {"model": mi}
    for tag, alpha in [("clean", 0.0), ("realistic", None)]:
        a = 0.0 if alpha == 0.0 else find_alpha(mi, X)
        gW, gb, model, meta = L.synth_grad(mi, X, noise_alpha=a, seed=1)
        ratio, reliable = R.ratio_rows(gW, gb)
        # SPA
        sel_g, idx, Y, sel_l = A.spa_select(ratio, gW, reliable, n=128)
        spa = ratio[sel_g].reshape(-1, 3, 64, 64).clamp(0, 1)
        # OLD quality
        q_g, _, _ = A.quality_select(ratio, reliable, n=128)
        old = ratio[q_g].reshape(-1, 3, 64, 64).clamp(0, 1)
        s_spa = G.hungarian_ssim(spa, X, convention=conv)
        s_old = G.hungarian_ssim(old, X, convention=conv)
        out[tag] = {"SPA": round(s_spa, 4), "OLD": round(s_old, 4),
                    "delta": round(s_spa - s_old, 4),
                    "spa_n": int(spa.shape[0]), "old_n": int(old.shape[0]),
                    "alpha": round(float(a), 2)}
        print(f"  model{mi} {tag:9} (a={a:.2f}): SPA={s_spa:.4f}  OLD={s_old:.4f}  "
              f"delta={s_spa-s_old:+.4f}", flush=True)
    return out


def main():
    conv = sys.argv[1] if len(sys.argv) > 1 else "perch7"
    X = L.faces(128)
    print(f"=== WS1 synthetic-oracle eval (convention={conv}, true GT = known CelebA) ===", flush=True)
    report = {"convention": conv, "models": {}}
    for mi in LEAKY:
        report["models"][f"model{mi}"] = eval_one(mi, X, conv=conv)
    U.save_json(report, U.work_path("reports", "arch", "synth_spa_vs_old.json"))
    print("\nWROTE", U.work_path("reports", "arch", "synth_spa_vs_old.json"), flush=True)


if __name__ == "__main__":
    main()

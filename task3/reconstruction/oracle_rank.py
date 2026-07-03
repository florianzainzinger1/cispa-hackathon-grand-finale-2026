"""Realize WS2's value: re-rank the existing candidate pool (outputs_final/*.pt)
with the CORRECTED per-channel oracle vs the OLD grayscale one, against the
replay-certified ground_truth.pt (5 covered models). Answers: does the fixed
oracle surface a candidate that beats the current board best (pib1), or would the
broken grayscale oracle have picked something different/worse?

Efficiency: a candidate's per-model SSIM is recomputed ONLY for models whose block
differs from pib1's; identical blocks reuse pib1's cached score.
"""
import os
import glob
import json
import numpy as np
import torch

import utils as U
import ssim_grader as G
import ssim_validate as V   # reuse match_gt / build_victim (import is side-effect free)

CONVS = ["gray7", "perch7"]
REF = "test_pib1"


def block(d, m):
    e = d[m]
    return (e["images"] if isinstance(e, dict) else e).float().clamp(0, 1)


def main():
    gt = torch.load(V.GT_PATH, weights_only=False, map_location="cpu")
    print("=== matching challenge models -> GT blocks ===", flush=True)
    matches = V.match_gt(gt)                      # {challenge_model: gt_block_key}
    cov = list(matches)
    gtimg = {m: gt[matches[m]]["images"].float().clamp(0, 1) for m in cov}
    print("covered models:", cov, flush=True)

    files = sorted(glob.glob(U.work_path("outputs_final", "*.pt")))
    cands = {}
    for f in files:
        name = os.path.basename(f)[:-3]
        try:
            cands[name] = torch.load(f, weights_only=False, map_location="cpu")
        except Exception:
            pass
    assert REF in cands, "pib1 reference missing"
    pib = cands[REF]

    # cache pib per-(model,conv) scores + pib block bytes for identity test
    pib_score = {c: {} for c in CONVS}
    pib_bytes = {}
    for m in cov:
        pib_bytes[m] = block(pib, m).reshape(128, -1).numpy().tobytes()
        for c in CONVS:
            pib_score[c][m] = G.hungarian_ssim(block(pib, m), gtimg[m], convention=c)
    print("pib1 per-model:", {c: {m: round(pib_score[c][m], 4) for m in cov} for c in CONVS}, flush=True)

    results = {}   # name -> {conv: mean}
    for name, d in cands.items():
        # only rank full 12-model submissions
        if not all(f"model{i}" in d for i in range(1, 13)):
            continue
        row = {}
        for c in CONVS:
            vals = []
            for m in cov:
                same = block(d, m).reshape(128, -1).numpy().tobytes() == pib_bytes[m]
                vals.append(pib_score[c][m] if same else
                            G.hungarian_ssim(block(d, m), gtimg[m], convention=c))
            row[c] = float(np.mean(vals))
        results[name] = row
        print(f"  {name:26} gray7={row['gray7']:.4f}  perch7={row['perch7']:.4f}", flush=True)

    print("\n=== RANKING (over the 5 covered models) ===", flush=True)
    for c in CONVS:
        top = sorted(results, key=lambda n: -results[n][c])[:8]
        print(f"[{c}] top-8: " + ", ".join(f"{n}={results[n][c]:.4f}" for n in top), flush=True)
        best = top[0]
        print(f"   -> {c} picks '{best}'  (pib1={results[REF][c]:.4f}, "
              f"{'== pib1' if best == REF else 'DIFFERENT from pib1'})", flush=True)

    U.save_json({"results": results, "covered": cov, "ref": REF},
                U.work_path("reports", "arch", "oracle_rank.json"))
    print("\nWROTE", U.work_path("reports", "arch", "oracle_rank.json"), flush=True)


if __name__ == "__main__":
    main()

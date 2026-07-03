"""WS2 — pin the SSIM convention by reproducing the real leaderboard ORDER.

Re-scores the three known submissions (test_pib1=0.3878 > test_lowsig=0.3851 >
test_nlm05=0.3722) against the gradient-replay-CERTIFIED ground_truth.pt, under
several SSIM conventions, and reports which convention reproduces the true order.

ground_truth.pt is keyed by arbitrary source descriptors, NOT challenge-model
index, so we match each challenge model to its GT block by replay-cosine
(reproducing gp_vit_prior.find_gt_for's logic, self-contained, no stylegan import).

Login-safe. Heaviest part = per-model 128x128 SSIM matrices; keep the convention
list short (default: gray7 vs perch7 -- the decisive colour test).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

import utils as U
import models as M
import ssim_grader as G

SUBMISSIONS = {"pib1": 0.3878, "lowsig": 0.3851, "nlm05": 0.3722}
GT_PATH = U.work_path("outputs_retr", "ground_truth.pt")


# ---------------------------------------------------------------- GT matching
def build_victim(name):
    g = U.load_grad(name); state = U.load_state(name)
    model = M.build_model(name, g, state)
    for p in model.parameters():
        p.requires_grad_(True)
    P = dict(model.named_parameters())
    keys = [k for k in g["gradients"].keys() if k in P]
    observed = torch.cat([g["gradients"][k].reshape(-1).float() for k in keys])
    observed = observed / (observed.norm() + 1e-12)
    return model, P, keys, observed


def replay_cos(model, P, keys, observed, imgs, labels):
    logits = model(imgs)
    loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(loss, [P[k] for k in keys], retain_graph=False)
    vec = torch.cat([x.reshape(-1) for x in grads])
    vec = vec / (vec.norm() + 1e-12)
    return float((vec * observed).sum())


def match_gt(gt, cos_thresh=0.9):
    """Return {challenge_model_name: gt_block_key} for models with replay-cos>thresh."""
    matches = {}
    for name in U.MODELS:
        try:
            model, P, keys, observed = build_victim(name)
        except Exception as e:
            print(f"  [skip {name}] build failed: {str(e)[:80]}", flush=True)
            continue
        best_k, best_c = None, -2.0
        for gk, entry in gt.items():
            imgs = entry["images"].float().clamp(0, 1)
            labels = entry["labels"].long()
            try:
                c = replay_cos(model, P, keys, observed, imgs, labels)
            except Exception:
                c = -2.0
            if c > best_c:
                best_c, best_k = c, gk
        flag = "MATCH" if best_c > cos_thresh else "no-match"
        print(f"  {name}: best GT='{best_k}' cos={best_c:.3f} [{flag}]", flush=True)
        if best_c > cos_thresh:
            matches[name] = best_k
    return matches


# ---------------------------------------------------------------- scoring
def load_submission(tag):
    p = U.work_path("outputs_final", f"test_{tag}.pt")
    return torch.load(p, weights_only=False, map_location="cpu")


def main():
    conventions = sys.argv[1:] or ["gray7", "perch7"]
    gt = torch.load(GT_PATH, weights_only=False, map_location="cpu")
    print(f"=== matching {len(U.MODELS)} challenge models to {len(gt)} GT blocks ===", flush=True)
    matches = match_gt(gt)
    if not matches:
        print("NO GT MATCHES -> cannot validate.", flush=True)
        return
    print(f"\nmatched models: {list(matches)}", flush=True)

    subs = {tag: load_submission(tag) for tag in SUBMISSIONS}
    results = {}   # convention -> {tag -> mean_over_matched_models}
    per_model = {} # convention -> tag -> {model: ssim}
    for conv in conventions:
        results[conv] = {}
        per_model[conv] = {}
        for tag in SUBMISSIONS:
            per_model[conv][tag] = {}
            vals = []
            for cm, gk in matches.items():
                A = subs[tag][cm]
                A = (A["images"] if isinstance(A, dict) else A).float().clamp(0, 1)
                B = gt[gk]["images"].float().clamp(0, 1)
                s = G.hungarian_ssim(A, B, convention=conv)
                per_model[conv][tag][cm] = s
                vals.append(s)
            results[conv][tag] = float(np.mean(vals))
            print(f"[{conv}] {tag}: mean={results[conv][tag]:.4f}  "
                  f"(per-model {json.dumps({k: round(v,3) for k,v in per_model[conv][tag].items()})})",
                  flush=True)

    # verdict: which convention reproduces pib1 > lowsig > nlm05
    print("\n=== VERDICT ===", flush=True)
    board = list(SUBMISSIONS)  # already in descending board order
    for conv in conventions:
        order = sorted(SUBMISSIONS, key=lambda t: -results[conv][t])
        reproduces = order == board
        triple = tuple(round(results[conv][t], 4) for t in board)
        print(f"  {conv}: order={order}  reproduces_board={reproduces}  "
              f"triple(pib1,lowsig,nlm05)={triple}", flush=True)

    U.save_json({"results": results, "per_model": per_model, "matches": matches,
                 "board": SUBMISSIONS},
                U.work_path("reports", "arch", "ssim_convention_validation.json"))
    print("\nWROTE", U.work_path("reports", "arch", "ssim_convention_validation.json"), flush=True)


if __name__ == "__main__":
    main()

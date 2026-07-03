"""READ-ONLY empirical verification probe (agent3). Answers the 100%-certainty
questions literature cannot, on our ACTUAL gradients. Touches NO pipeline code and
writes NO submission -- only prints + a small reports/build/verify_probe.json.

Q1 do the 12 models share the same private images?  -> MLP top-128 ROWSPACE overlap
   (rowspace(gW)=span of the images; shared images => overlap~1, disjoint => ~128/12288~0.01)
Q2 are the classifier heads identical (co-trained / shared label space)? -> head weight diff
Q3 is the off-top-128 gradient energy isotropic NOISE (ceiling is a real defense) vs a sharp
   rank-128 cliff (clean single-batch)? -> singular spectrum gap + tail coefficient-of-variation
Q4 what does the sibling ground_truth.pt reveal about provenance/preprocessing?
"""
import os
import sys
import json
import torch

import utils as U
import attack_analytic as AA
import infer_labels as IL

MLP = ["model1", "model4", "model5", "model8"]
ALL = U.MODELS


def first_layer(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    base = AA.first_leaky_linear(name, state, g)
    gW = g[base + ".weight"].float()
    gb = g[base + ".bias"].float()
    return gW, gb, grad, state, base


def main():
    rep = {}

    # --- Q1: MLP cross-model top-128 rowspace (image-space) overlap ---
    print("=== Q1: MLP cross-model top-128 ROWSPACE overlap (image space, d=12288) ===")
    print("     overlap=||Va Vb^T||_F^2/128 in [0,1]; disjoint images ~0.01, shared ~high")
    Vs = {}
    for n in MLP:
        gW, gb, grad, state, base = first_layer(n)
        _, S, Vh = torch.linalg.svd(gW, full_matrices=False)
        Vs[n] = Vh[:128].contiguous()
    ov = {}
    print("        " + " ".join(f"{n:>8}" for n in MLP))
    for a in MLP:
        row = []
        for b in MLP:
            M = Vs[a] @ Vs[b].t()
            row.append(float((M ** 2).sum() / 128.0))
        ov[a] = {b: row[i] for i, b in enumerate(MLP)}
        print(f"{a:>8} " + " ".join(f"{x:8.4f}" for x in row))
    rep["mlp_rowspace_overlap"] = ov

    # --- Q2: classifier head weight identity (grouped by shape; ViT heads differ) ---
    print("\n=== Q2: classifier-head weight cross-model relative L2 diff (0=identical) ===")
    from collections import defaultdict
    heads, hshape = {}, {}
    for n in ALL:
        grad = U.load_grad(n)
        state = U.load_state(n)
        wk, bk = IL.head_keys(state, grad["gradients"])
        w = state[wk].float()
        heads[n] = w.reshape(-1)
        hshape[n] = tuple(w.shape)
    print("  head shapes:", {n: hshape[n] for n in ALL})
    groups = defaultdict(list)
    for n in ALL:
        groups[heads[n].shape].append(n)
    hdiff = {}
    for shp, names in groups.items():
        print(f"  -- group shape {tuple(shp)}: {names} --")
        for a in names:
            hdiff[a] = {b: float((heads[a] - heads[b]).norm() / (heads[a].norm() + 1e-12)) for b in names}
            offs = [hdiff[a][b] for b in names if b != a]
            mn = min(offs) if offs else float("nan")
            print(f"     {a:>8} min_offdiag={mn:.4f}  (identical head if ~0)")
    rep["head_reldiff"] = hdiff
    rep["head_shapes"] = {n: list(hshape[n]) for n in ALL}

    # --- Q3: first-layer spectrum tail: clean rank-128 cliff vs isotropic noise ---
    print("\n=== Q3: first-layer spectrum — rank-128 cliff vs noise tail ===")
    tails = {}
    for n in ALL:
        gW, gb, grad, state, base = first_layer(n)
        _, S, _ = torch.linalg.svd(gW, full_matrices=False)
        e128 = float((S[:128] ** 2).sum() / (S ** 2).sum())
        gap = float(S[127] / (S[128] + 1e-12)) if S.numel() > 128 else float("inf")
        tail = S[128:] if S.numel() > 128 else S[:0]
        cv = float(tail.std() / (tail.mean() + 1e-12)) if tail.numel() > 1 else 0.0
        tails[n] = {"family": grad.get("family"), "E128": round(e128, 4),
                    "gap_127_128": round(gap, 3), "tail_cv": round(cv, 4), "n_sv": int(S.numel())}
        print(f"{n:>8} [{str(grad.get('family')):3}] E128={e128:.3f} gap127/128={gap:9.2f} "
              f"tail_cv={cv:.3f} (low cv+sharp gap=noise floor; huge gap=clean rank-128)")
    rep["spectrum_tail"] = tails

    # --- Q4: ground_truth.pt provenance ---
    print("\n=== Q4: outputs_retr/ground_truth.pt provenance ===")
    gt_path = U.work_path("outputs_retr", "ground_truth.pt")
    if os.path.exists(gt_path):
        gt = torch.load(gt_path, weights_only=False)
        info = {"type": str(type(gt))}
        if isinstance(gt, dict):
            info["keys"] = list(gt.keys())[:50]
            for k, v in gt.items():
                if isinstance(v, torch.Tensor):
                    info[f"{k}__shape"] = list(v.shape)
                elif isinstance(v, (str, int, float, bool)):
                    info[k] = v
                elif isinstance(v, dict):
                    info[f"{k}__subkeys"] = list(v.keys())[:25]
                elif isinstance(v, (list, tuple)):
                    info[f"{k}__len"] = len(v)
                    info[f"{k}__head"] = str(v[:6])
        print(json.dumps(info, indent=2, default=str)[:2500])
        rep["ground_truth_meta"] = info
    else:
        print("(missing)")

    U.save_json(rep, U.work_path("reports", "build", "verify_probe.json"))
    print("\nwrote reports/build/verify_probe.json")


if __name__ == "__main__":
    main()

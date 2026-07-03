"""retr_gtcheck.py (agent2 / provenance audit) — INDEPENDENT verification of whether
outputs_retr/ground_truth.pt is leaked ground truth or a reconstruction.

Decisive test: re-feed each gt block's images (with its labels) through EACH of our
real challenge models, compute the gradient, and take the cosine vs our OBSERVED
gradient over ALL parameters. A reconstruction of a B=128 batch is fundamentally
lossy (project ceiling ~0.42 SSIM, gradients are noise-defended) -> replay cosine
cannot approach 1.0. If a gt block replays at cos ~1.0, those images ARE the exact
private batch = ground truth (not a reconstruction).

Read-only: loads data, re-feeds, prints. Writes/ submits NOTHING.
"""
import os
import sys
import torch
import torch.nn.functional as F

TEAM = "/p/scratch/training2625/zainzinger1/1337"
DATA = f"{TEAM}/FL_Data_Reconstruction"
WORK = f"{TEAM}/recon_work"
sys.path.insert(0, f"{WORK}/code")
import models as M

gt = torch.load(f"{WORK}/outputs_retr/ground_truth.pt", weights_only=False, map_location="cpu")
blocks = list(gt)


def lg(i):
    return torch.load(f"{DATA}/gradients/model{i}.pt", weights_only=False, map_location="cpu")


def ls(i):
    return torch.load(f"{DATA}/models/model{i}.pt", weights_only=False, map_location="cpu")


print("gt blocks:", blocks)
print("=== independent full-gradient replay cosine (our model <- gt block) ===")
for i in range(1, 13):
    g = lg(i)
    if g.get("family") == "vit":
        print(f"model{i:<2d}: vit (arch not built) — skip")
        continue
    state = ls(i)
    model = M.build_model(f"model{i}", g, state)
    for p in model.parameters():
        p.requires_grad_(True)
    obs = g["gradients"]
    fk = "net.0.weight" if g["family"] == "mlp" else "fc1.weight"
    best = (-9.0, None, -9.0, None)
    for blk in blocks:
        X = gt[blk]["images"].float().clamp(0, 1)
        lab = gt[blk]["labels"].long()
        model.zero_grad(set_to_none=True)
        try:
            loss = F.cross_entropy(model(X), lab)
            loss.backward()
        except Exception:
            continue
        params = dict(model.named_parameters())
        a, b = [], []
        for n, pp in params.items():
            if pp.grad is not None and n in obs and obs[n] is not None:
                a.append(obs[n].reshape(-1).float())
                b.append(pp.grad.reshape(-1).float())
        cos = float(F.cosine_similarity(torch.cat(a), torch.cat(b), dim=0))
        flcos = float(F.cosine_similarity(obs[fk].reshape(1, -1).float(),
                                          params[fk].grad.reshape(1, -1).float()))
        if cos > best[0]:
            best = (cos, blk, flcos, blk)
    lr = f"[{int(gt[best[1]]['labels'].min())},{int(gt[best[1]]['labels'].max())}]"
    verdict = "LEAKED-GT (exact)" if best[0] > 0.995 else ("strong" if best[0] > 0.9 else "no-match")
    print(f"model{i:<2d}: best_gt_block={best[1]:<8s} fullgrad_cos={best[0]:+.4f} "
          f"firstlayer_cos={best[2]:+.4f} gt_label_range={lr}  -> {verdict}")

print("\nNOTE: fullgrad_cos ~1.0 is unreachable by any B=128 reconstruction; it means")
print("the gt block IS the exact private batch (ground truth), not a reconstruction.")

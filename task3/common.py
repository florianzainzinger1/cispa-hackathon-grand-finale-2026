"""Shared helpers for the FL gradient-inversion task (task3).

Data layout:
  gradients/model{i}.pt -> dict(gradients={pname:tensor}, family, activation,
                                 feature_shape=(C,H,W), batch_size=128)
  models/model{i}.pt    -> plain state_dict {pname: weight}

Reconstruction target: 128 images per model, (3,64,64) float32 in [0,1].
Scored by Hungarian-matched mean SSIM vs the private ground truth.
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRAD_DIR = os.path.join(HERE, "gradients")
MODEL_DIR = os.path.join(HERE, "models")
N_MODELS = 12
BATCH = 128
IMG_SHAPE = (3, 64, 64)


def load_grad(i):
    return torch.load(os.path.join(GRAD_DIR, f"model{i}.pt"),
                      map_location="cpu", weights_only=False)


def load_state(i):
    return torch.load(os.path.join(MODEL_DIR, f"model{i}.pt"),
                      map_location="cpu", weights_only=False)


def imageness(x):
    """Higher = more image-like. Penalises pixel mass outside [0,1].
    x: (...,3,64,64) or (...,N) tensor. Returns per-sample score in (-inf,0]."""
    flat = x.reshape(x.shape[0], -1)
    below = torch.clamp(-flat, min=0)          # how far below 0
    above = torch.clamp(flat - 1, min=0)        # how far above 1
    return -(below + above).mean(dim=1)


def dedup(cands, scores, k=BATCH, sim_thresh=0.02):
    """Greedy: take highest-scoring candidates, skip near-duplicates.
    cands: (M,3,64,64). Returns indices (<=k) into cands, best first."""
    order = torch.argsort(scores, descending=True)
    chosen = []
    flat = cands.reshape(cands.shape[0], -1)
    for idx in order.tolist():
        v = flat[idx]
        dup = False
        for c in chosen:
            if ((flat[c] - v) ** 2).mean() < sim_thresh:
                dup = True
                break
        if not dup:
            chosen.append(idx)
        if len(chosen) >= k:
            break
    return chosen


def finalize(images):
    """Clamp/reshape a batch to the exact submission spec for one model."""
    x = images.reshape(-1, *IMG_SHAPE).float().clamp(0, 1)
    assert x.shape == (BATCH, *IMG_SHAPE), x.shape
    return x.contiguous()

"""Objective, label-free v1-vs-v2 fidelity for MLP models.

The first-layer weight gradient satisfies gW = C @ X_true (X_true = the 128 true
images), so every row of gW lies in the row-space of the true images. A candidate
set X therefore explains gW well only if span(X) aligns with span(X_true).
We report the fraction of gW's energy captured by projecting its rows onto
span(X):  higher = more faithful. Reconstruction noise tilts span(X) off the true
subspace and lowers capture, so this rewards genuine denoising.

READ-ONLY. Prints a table; writes nothing that touches v1.
"""
import os
import sys
import numpy as np
import torch
import utils as U

MLPS = [1, 4, 5, 8]


def first_image_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2 and w.shape[1] == U.IMG_NUMEL:
            base = pname[:-7]
            if base + ".bias" in g:
                return base
    return None


def captured(gW, X, tol=1e-3):
    """Fraction of gW energy in span(rows of X). gW:(1024,12288) np, X:(128,12288) np."""
    Xc = X - 0.0
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # Vt:(128,12288)
    r = int((S > S[0] * tol).sum())
    V = Vt[:r]                                           # (r,12288) orthonormal
    proj = gW @ V.T                                      # (1024,r)
    return float((proj ** 2).sum() / (gW ** 2).sum()), r


def load_imgs(path, key="images"):
    return torch.load(path, weights_only=False)[key].float().clamp(0, 1).reshape(128, -1).numpy()


def main():
    print(f"{'model':7} {'v1_cap':>8} {'v2_cap':>8} {'rand_cap':>8} {'winner':>8}")
    rng = np.random.RandomState(0)
    rows = []
    for m in MLPS:
        name = f"model{m}"
        grad = U.load_grad(name)
        state = U.load_state(name)
        base = first_image_linear(state, grad["gradients"])
        gW = grad["gradients"][base + ".weight"].float().numpy()

        v1 = load_imgs(U.work_path("candidates", name, "selected_128.pt"))
        v2 = load_imgs(U.work_path("candidates_v2", name, "mlp_exact_selected_128.pt"))
        rnd = rng.rand(128, U.IMG_NUMEL)

        c1, r1 = captured(gW, v1)
        c2, r2 = captured(gW, v2)
        cr, _ = captured(gW, rnd)
        win = "v2" if c2 > c1 + 1e-4 else ("v1" if c1 > c2 + 1e-4 else "tie")
        print(f"{name:7} {c1:8.4f} {c2:8.4f} {cr:8.4f} {win:>8}   (rank v1={r1} v2={r2})")
        rows.append({"model": name, "v1_captured": c1, "v2_captured": c2,
                     "random_captured": cr, "winner": win, "rank_v1": r1, "rank_v2": r2})
    U.save_json(rows, U.work_path("reports", "v2_fidelity_capture.json"))
    print("\nHigher captured = the 128-image set better explains the first-layer gradient.")


if __name__ == "__main__":
    main()

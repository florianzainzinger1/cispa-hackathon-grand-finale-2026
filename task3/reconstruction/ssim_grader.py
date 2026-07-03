"""WS2 — grader-faithful Hungarian SSIM.  NEW MODULE (drop-in; does NOT overwrite
synth_lib.py / gp_vit_prior.py).

The existing oracle scores in CHANNEL-MEAN GRAYSCALE (A.mean(1)) and, in synth_lib,
matches pairs by pixel-MSE.  Both diverge from the server grader, which (per the
Task-3 PDF) pairs each submitted image to the closest training image *by SSIM*
one-to-one and averages.  Grayscale is blind to per-channel colour/brightness — the
reason a photometric candidate (pib1) mis-ranks vs grain fixes (lowsig/nlm05).

This module: per-channel skimage SSIM, Hungarian assignment computed *on the SSIM
cost matrix*, window selectable (7x7 uniform = skimage default, or 11x11 gaussian
sigma=1.5 = Wang standard), no shift alignment (the grader does none).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as sk_ssim


def _ssim_kwargs(convention):
    """convention: '{gray|perch}{7|11}'  -> (skimage kwargs, per_channel bool)."""
    per_channel = convention.startswith("perch")
    gaussian = convention.endswith("11")
    kw = dict(data_range=1.0)
    if gaussian:
        kw.update(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)  # win_size->11
    else:
        kw.update(win_size=7)
    return kw, per_channel


def _prep(x):
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.clip(np.asarray(x, dtype=np.float32), 0, 1)   # (N,3,64,64)


def ssim_cost_matrix(A, B, convention="perch7"):
    """(N,N) SSIM cost matrix. A,B: (N,3,64,64)."""
    A = _prep(A); B = _prep(B)
    n = A.shape[0]
    kw, per_channel = _ssim_kwargs(convention)
    if per_channel:
        C = np.empty((n, n), np.float64)
        for i in range(n):
            ai = A[i]
            for j in range(n):
                C[i, j] = sk_ssim(ai, B[j], channel_axis=0, **kw)
    else:
        Ag = A.mean(1); Bg = B.mean(1)
        C = np.empty((n, n), np.float64)
        for i in range(n):
            ai = Ag[i]
            for j in range(n):
                C[i, j] = sk_ssim(ai, Bg[j], **kw)
    return C


def hungarian_ssim(A, B, convention="perch7"):
    """Grader-faithful one-to-one Hungarian SSIM between two (N,3,64,64) sets.
    A = submission[model], B = ground_truth[model]. Returns mean matched SSIM."""
    C = ssim_cost_matrix(A, B, convention)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())

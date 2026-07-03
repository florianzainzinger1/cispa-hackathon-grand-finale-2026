"""Agent R shared helpers (retrieval / membership). No heavy deps beyond torch,
numpy, skimage. Efficient top-k right-singular subspace via the small-side gram
trick (gW is 1024 x 12288, so eigh on the 1024x1024 gram is ~1s vs minutes for
full SVD)."""
import os
import torch
import numpy as np


def load_grad_firstlayer(data_root, name):
    """Return (gW, gb, key) first-layer weight/bias gradient for a model."""
    g = torch.load(os.path.join(data_root, "gradients", f"{name}.pt"),
                   weights_only=False, map_location="cpu")["gradients"]
    if "net.0.weight" in g:                 # MLP: input IS the pixel image
        return g["net.0.weight"].float(), g.get("net.0.bias"), "net.0.weight"
    if "conv.weight" in g:                   # CNN first conv (pixel space, but conv)
        return g["conv.weight"].float(), g.get("conv.bias"), "conv.weight"
    k = list(g)[0]
    return g[k].float(), None, k


def topk_right_subspace(gW, k=128):
    """Top-k right-singular vectors of gW (m,n), m<<n. Returns V (k,n) rows
    orthonormal, and singular values s (k,)."""
    gW = gW.float()
    m, n = gW.shape
    k = min(k, m)
    M = gW @ gW.t()                          # (m,m) small
    evals, evecs = torch.linalg.eigh(M)      # ascending
    order = torch.argsort(evals, descending=True)[:k]
    U = evecs[:, order]                      # (m,k)
    s = evals[order].clamp_min(0).sqrt()     # (k,)
    V = (gW.t() @ U) / s.clamp_min(1e-8)     # (n,k)
    # renormalise rows (guard against tiny s)
    V = V.t().contiguous()                   # (k,n)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
    return V, s


def augment_dc(V):
    """Append 3 per-channel DC (constant-within-channel) basis vectors to V and
    re-orthonormalise. Makes the residual test invariant to per-channel
    brightness/mean shifts (absorbs a ToTensor-vs-Normalize ambiguity). V:(k,n)
    with n=3*64*64."""
    k, n = V.shape
    hw = n // 3
    dc = torch.zeros(3, n)
    for c in range(3):
        dc[c, c * hw:(c + 1) * hw] = 1.0
    dc = dc / dc.norm(dim=1, keepdim=True)
    B = torch.cat([V, dc], dim=0)              # (k+3, n)
    Q, _ = torch.linalg.qr(B.t())              # (n, k+3) orthonormal cols
    return Q.t().contiguous()                  # (k+3, n)


def subspace_residual(imgs, V, center=None):
    """Fractional residual of each image after projecting onto rowspace(V).
    imgs (N,C,H,W) or (N,D). center: optional (D,) vector subtracted first."""
    X = imgs.reshape(imgs.shape[0], -1).float()
    if center is not None:
        X = X - center.reshape(1, -1)
    proj = (X @ V.t()) @ V
    return ((X - proj).norm(dim=1) / (X.norm(dim=1) + 1e-9))


def _ssim_pair(a, b):
    from skimage.metrics import structural_similarity as ssim
    a = a.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    b = b.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return ssim(a, b, channel_axis=2, data_range=1.0)


def best_ssim_to_gallery(query, gallery, topn=1):
    """For a single query image (C,H,W), return best SSIM + index in gallery
    (N,C,H,W). Coarse: use a fast MSE prefilter then SSIM on the top candidates."""
    q = query.reshape(-1).float()
    G = gallery.reshape(gallery.shape[0], -1).float()
    d = ((G - q.unsqueeze(0)) ** 2).mean(dim=1)     # MSE prefilter
    cand = torch.argsort(d)[:max(topn, 32)]
    best_s, best_i = -1.0, -1
    for i in cand.tolist():
        s = _ssim_pair(query, gallery[i])
        if s > best_s:
            best_s, best_i = s, i
    return best_s, best_i

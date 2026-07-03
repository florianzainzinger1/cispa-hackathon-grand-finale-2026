"""BSS (agent3 / lever L3) shared utilities.

Namespace-local so we never edit the shared modules (utils/models/attack_*).
Provides: layer loading, the first-layer decomposition gW = C X, CPA-style
whitening, and a LABEL-FREE measurement harness to rank reconstructions without
ground truth (subspace residual + gW/gb self-consistency + distinctness/grain).

Math anchor: for a biased first linear layer y = W x + b,
    gW = sum_n c_n x_n^T = C X   (C: h x 128, cols c_n = dL/dz_n ; X: 128 x d, rows x_n)
    gb = sum_n c_n     = C 1_128
so every image x_n lies in rowspace(gW), and a correct X must ALSO satisfy
gb = C_hat 1 with C_hat = gW pinv(X) -- the one constraint X was never fit to.
"""
import os
import torch
import torch.nn.functional as F

import utils as U
import models as M

IMG_C, IMG_H, IMG_W, D_IMG, N = U.IMG_C, U.IMG_H, U.IMG_W, U.IMG_NUMEL, U.N_IMAGES

MLP_MODELS = ["model1", "model4", "model5", "model8"]
CNN_MODELS = ["model6", "model10", "model12", "model2", "model7", "model3"]


# ---------------------------------------------------------------------------
# Loading / layer selection
# ---------------------------------------------------------------------------
def first_biased_linear(state, g):
    """Earliest biased 2-D linear (net.0 for MLP, fc1 for CNN). Same rule as
    attack_analytic.first_leaky_linear (copied to keep our namespace isolated)."""
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def load_layer(name, device="cpu"):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    family = grad.get("family")
    activation = grad.get("activation")
    feat_shape = tuple(grad.get("feature_shape")) if grad.get("feature_shape") else None
    base = first_biased_linear(state, g)
    gW = g[base + ".weight"].float().to(device)     # (h, d_in)
    gb = g[base + ".bias"].float().to(device)        # (h,)
    d_in = gW.shape[1]
    return {
        "name": name, "grad": grad, "state": state, "g": g,
        "family": family, "activation": activation, "feature_shape": feat_shape,
        "base": base, "gW": gW, "gb": gb, "d_in": d_in, "h": gW.shape[0],
        "is_image_layer": d_in == D_IMG,
    }


def build_cnn(lay):
    meta = {"family": "cnn", "activation": lay["activation"],
            "feature_shape": lay["feature_shape"]}
    return M.build_model(lay["name"], meta, lay["state"])


def conv_feature_of(lay, X):
    """CNN: model.conv_feature(X) flattened -> (N, d_in) = the fc1 input."""
    model = build_cnn(lay).to(X.device)
    with torch.no_grad():
        f = model.conv_feature(X)
    return f.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# CPA whitening: center per-row, cov over the h-observation axis, top-B eig.
# Returns whitened CENTERED mixtures Xw (B,d), whitening K (B,h), row-mean mu (h,1)
# ---------------------------------------------------------------------------
def whiten_topk(gW, B=N, denoise_rank=None):
    """gW: (h,d). If denoise_rank is set, first project gW onto its own top-r
    right-singular subspace (removes DP noise energy) before whitening."""
    X = gW
    if denoise_rank is not None and denoise_rank < min(X.shape):
        Uu, S, Vh = torch.linalg.svd(X, full_matrices=False)
        r = denoise_rank
        X = (Uu[:, :r] * S[:r]) @ Vh[:r]
    mu = X.mean(dim=1, keepdim=True)                 # (h,1) per-row mean over d
    Xzc = X - mu
    d = Xzc.shape[1]
    cov = (Xzc @ Xzc.t()) / (d - 1)                  # (h,h) symmetric PSD
    evals, evecs = torch.linalg.eigh(cov)            # ascending
    k = min(B, evals.numel())
    top_vals = evals[-k:].clamp_min(1e-12)           # (k,)
    top_vecs = evecs[:, -k:]                          # (h,k)
    K = (top_vals.rsqrt().unsqueeze(1) * top_vecs.t())   # (k,h) = D^-1/2 E^T
    Xw = K @ Xzc                                     # (k,d) whitened centered mixtures
    return Xw, K, mu


# ---------------------------------------------------------------------------
# Sign / scale mapping
# ---------------------------------------------------------------------------
def minmax_rows(x):
    """Per-row min-max to [0,1]. x: (N, ...)."""
    f = x.reshape(x.shape[0], -1)
    lo = f.min(dim=1, keepdim=True).values
    hi = f.max(dim=1, keepdim=True).values
    return ((f - lo) / (hi - lo + 1e-9)).reshape_as(x)


def resolve_sign_lowTV(S):
    """S: (N,C,H,W) raw sources. Pick per-image orientation (s or -s) whose
    min-max image has lower total variation (real faces < their photo-negatives)."""
    pos = minmax_rows(S)
    neg = minmax_rows(-S)
    tvp = U.total_variation(pos)
    tvn = U.total_variation(neg)
    use_neg = (tvn < tvp).view(-1, 1, 1, 1)
    return torch.where(use_neg, neg, pos)


def biasscale_fix(gW, gb, S, min_scale=1e-6):
    """SPEAR++ Thm-4.1-style per-source scale+sign fix, applied to any recovered
    source set S (B,d) that is correct up to per-row scale/sign. With S = diag(d) X,
    C_hat = gW pinv(S) = C diag(1/d), and gb = C 1 = C_hat diag(d) 1 = C_hat d, so
    d = lstsq(C_hat, gb); true sources X = S / d. Fixes magnitude AND sign at once.
    Returns (X, d, ok) where ok flags rows with a reliable (non-degenerate) scale."""
    Spinv = torch.linalg.pinv(S)                     # (d,B)
    C_hat = gW @ Spinv                               # (h,B)
    d = torch.linalg.lstsq(C_hat, gb.unsqueeze(1)).solution.squeeze(1)  # (B,)
    ok = d.abs() > min_scale
    safe = torch.where(ok, d, torch.ones_like(d))
    X = S / safe.unsqueeze(1)
    return X, d, ok


# ---------------------------------------------------------------------------
# Measurement WITHOUT ground truth
# ---------------------------------------------------------------------------
def subspace_basis(gW, k=N):
    """Top-k right-singular vectors of gW -> (d,k) orthonormal columns."""
    _, S, Vh = torch.linalg.svd(gW, full_matrices=False)
    k = min(k, Vh.shape[0])
    return Vh[:k].t().contiguous(), S


def subspace_residual(V, Xin, dc=True):
    """Per-row residual of Xin (N,d) against subspace V (d,k). Lower=better.
    dc=True removes the constant/DC direction (handles the min-max affine shift)."""
    Xf = Xin.reshape(Xin.shape[0], -1).clone()
    Vc = V
    if dc:
        Xf = Xf - Xf.mean(dim=1, keepdim=True)
        ones = torch.ones(Vc.shape[0], 1, device=Vc.device) / (Vc.shape[0] ** 0.5)
        Vc = Vc - ones @ (ones.t() @ Vc)
        Vc, _ = torch.linalg.qr(Vc)
    coeff = Xf @ Vc
    cap = (coeff ** 2).sum(dim=1)
    tot = (Xf ** 2).sum(dim=1).clamp_min(1e-12)
    return torch.sqrt((1.0 - cap / tot).clamp_min(0.0))


def selfconsistency_labelfree(gW, gb, Xin):
    """C_hat = gW pinv(Xin); report first-layer reconstruction + the decisive
    bias check gb ?= C_hat 1 (the constraint Xin was never fit to)."""
    Xin = Xin.reshape(Xin.shape[0], -1)
    Xpinv = torch.linalg.pinv(Xin)                   # (d,N)
    C_hat = gW @ Xpinv                               # (h,N)
    resid_F = float((gW - C_hat @ Xin).norm() / gW.norm().clamp_min(1e-12))
    gb_hat = C_hat @ torch.ones(Xin.shape[0], device=Xin.device)
    bias_cos = float(F.cosine_similarity(gb, gb_hat, dim=0))
    bias_relL2 = float((gb - gb_hat).norm() / gb.norm().clamp_min(1e-12))
    # rank health of Xin (near-duplicate rows collapse pinv)
    sv = torch.linalg.svdvals(F.normalize(Xin, dim=1))
    rank = int((sv > 1e-2 * sv[0]).sum())
    return {"resid_F": resid_F, "bias_cos": bias_cos,
            "bias_relL2": bias_relL2, "rank_Xin": rank}


def distinctness(X):
    """X: (N,3,64,64) in [0,1]. Coverage / duplicate stats."""
    Xf = X.reshape(X.shape[0], -1)
    Xn = F.normalize(Xf - Xf.mean(dim=1, keepdim=True), dim=1)
    Sm = Xn @ Xn.t()
    off = Sm[~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)]
    sv = torch.linalg.svdvals(F.normalize(Xf, dim=1))
    eff = int((sv > 1e-2 * sv[0]).sum())
    # off-diagonal pairs with cosine > 0.97 (Sm.gt counts N diagonal ones + 2x pairs)
    n_dupe = (int(Sm.gt(0.97).sum().item()) - X.shape[0]) // 2
    return {"mean_offdiag": float(off.mean()), "max_offdiag": float(off.max()),
            "n_dupe_pairs": n_dupe, "effective_unique": eff}


def grain(X):
    tv = U.total_variation(X)
    flat = X.reshape(X.shape[0], -1)
    return {"tv_median": float(tv.median()), "tv_mean": float(tv.mean()),
            "std_mean": float(flat.std(dim=1).mean()),
            "frac_flat": float((flat.std(dim=1) < 0.03).float().mean())}


def measure(lay, X, V=None, S=None):
    """Full label-free scoreboard for a reconstruction X (N,3,64,64) in [0,1].
    For CNN, subspace/self-consistency are done in feature space."""
    gW, gb = lay["gW"].cpu(), lay["gb"].cpu()
    X = X.detach().float().cpu().clamp(0, 1)
    if lay["is_image_layer"]:
        Xin = X.reshape(X.shape[0], -1)
    else:
        Xin = conv_feature_of(lay, X).cpu()
    if V is None:
        V, S = subspace_basis(gW)
    # noise floor: energy beyond top-128
    e = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    floor = float((1 - e[min(N, len(e)) - 1]).clamp_min(0) ** 0.5)
    resid = subspace_residual(V, Xin, dc=True)
    sc = selfconsistency_labelfree(gW, gb, Xin)
    ds = distinctness(X)
    gr = grain(X)
    out = {"resid_median": float(resid.median()), "resid_p25": float(resid.quantile(0.25)),
           "noise_floor_resid": floor, **sc, **ds, **gr}
    return out, V, S


def combined_score(m):
    """Heuristic label-free score for ranking methods per model (higher=better).
    Coverage-weighted (Hungarian SSIM rewards distinct faces), gated by realism."""
    s_cov = m["effective_unique"] / float(N)
    s_bias = max(0.0, m["bias_cos"])
    s_sub = max(0.0, (m["noise_floor_resid"] + 0.35 - m["resid_median"])) / 0.35
    s_sub = min(1.0, max(0.0, s_sub))
    s_real = 1.0 - min(1.0, m["frac_flat"])
    return (0.45 * s_cov + 0.30 * s_sub + 0.25 * s_bias) * s_real

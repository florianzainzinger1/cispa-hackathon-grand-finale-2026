"""Agent1 (reconstruction) shared helpers: analytic ratio, intra-cluster
differencing denoise, collision rejection, clustering, certification.

Math (biased linear y=Wx+b, batch B):
  gW_i = sum_b w_{i,b} x_b ,  gb_i = sum_b w_{i,b}
  ratio_i = gW_i/gb_i = affine combo of the layer inputs x_b (=image for MLP
  net.0, =conv feature for CNN fc1). Single-sample-dominant rows -> clean x_b.

Intra-cluster differencing (denoise): for two neurons i,j dominated by the SAME
sample b, (gW_i-gW_j)/(gb_i-gb_j) ~= x_b + (noise_i-noise_j)/(w_{i,b}-w_{j,b}).
Choosing the pair with the LARGEST |gb_i-gb_j| minimises noise amplification and
cancels any shared leakage/DP component. Falls back to the best single ratio row.
"""
import numpy as np
import torch


def ratio_rows(gW, gb, eps=1e-8):
    gb = gb.float(); gW = gW.float()
    reliable = gb.abs() > eps
    denom = torch.where(reliable, gb, torch.full_like(gb, float("nan")))
    ratio = gW / denom.unsqueeze(1)
    return ratio, reliable


def per_image_minmax(x, eps=1e-6):
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo + eps)).reshape_as(x)


def cluster_labels(vecs, k, seed=0):
    """KMeans on L2-normalised rows. vecs (n,d) -> labels (n,)."""
    from sklearn.cluster import KMeans
    X = vecs.detach().cpu().numpy().astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    k = int(min(k, len(X)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=4).fit(X)
    return km.labels_, k


def diff_denoise_rep(gW, gb, members):
    """Best-pairwise-difference representative for a cluster of neuron indices.
    Returns (vec, method, gbdiff). Falls back to best-|gb| single ratio row."""
    members = list(members)
    if len(members) == 1:
        i = members[0]
        return gW[i] / gb[i], "single", float(gb[i].abs())
    gbm = gb[members]
    D = (gbm[:, None] - gbm[None, :]).abs()
    D.fill_diagonal_(-1)
    a, b = np.unravel_index(int(torch.argmax(D)), D.shape)
    gbdiff = float(D[a, b])
    if gbdiff < 1e-6:                                   # collision -> best single
        i = members[int(torch.argmax(gbm.abs()))]
        return gW[i] / gb[i], "single-collide", float(gbm.abs().max())
    i, j = members[a], members[b]
    return (gW[i] - gW[j]) / (gb[i] - gb[j]), "diff", gbdiff


def total_variation(x):
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-3, -2, -1))
    return dh + dw


def collision_ok(img01):
    """Reject blends/noise. img01 (3,64,64) already in ~[0,1]."""
    x = img01.reshape(-1)
    if not torch.isfinite(x).all():
        return False
    tv = total_variation(img01.unsqueeze(0)).item()
    std = x.std().item()
    inrange = ((x >= -0.05) & (x <= 1.05)).float().mean().item()
    return (tv < 0.30) and (std > 0.04) and (inrange > 0.85)


def quality_score(img01):
    """Higher = cleaner face-like (in-range * contrast / grain)."""
    x = img01.reshape(-1)
    tv = total_variation(img01.unsqueeze(0)).item()
    std = x.std().item()
    inrange = ((x >= -0.02) & (x <= 1.02)).float().mean().item()
    return inrange * min(std / 0.15, 1.0) / (1.0 + 4 * max(tv - 0.08, 0.0))

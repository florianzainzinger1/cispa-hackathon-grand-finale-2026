"""Stage 1-2: per-model analytic reconstruction -> candidate image pool.

Ratio readout r_i = gradW[i]/gradB[i] on the first biased linear (Boenisch/teammate),
then family-specific decode. Integrates OUR two recon-stage levers:
  * MLP  : low-rank denoise of gradW at the per-model noise-floor k BEFORE the ratio
           (config.MLP_LOWRANK_K) -- vs teammate's k=128.
  * model2/tanh CNN: featureness scored against the tanh output range [-1,1].
"""
import torch
import torch.nn.functional as F

import common as U
import config as C
from models import build_cnn


def first_biased_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            bkey = pname[:-7] + ".bias"
            if bkey in g and g[bkey] is not None:
                return pname[:-7]
    return None


def lowrank_denoise(W, k):
    """OUR lever #1: project W onto its top-k left/right singular subspace, killing
    the additive-noise floor before the analytic ratio. k=0 -> identity."""
    if not k or k >= min(W.shape):
        return W
    Uu, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (Uu[:, :k] * S[:k]) @ Vh[:k]


def ratio_rows(gW, gb, eps=1e-8):
    gb = gb.float(); gW = gW.float()
    reliable = gb.abs() > eps
    denom = torch.where(reliable, gb, torch.full_like(gb, float("nan")))
    return gW / denom.unsqueeze(1), reliable


def featureness(feat, act):
    """Rows more consistent with the activation OUTPUT range score higher.
    OUR lever #2: tanh features live in [-1,1] (negatives legitimate); the teammate's
    relu-style score (penalising <0) flagged every tanh feature bad -> model2 dead."""
    f = feat.reshape(feat.shape[0], -1)
    if act == "sigmoid":
        bad = torch.clamp(-f, min=0) + torch.clamp(f - 1, min=0)
    elif act == "tanh":
        bad = torch.clamp(-f - 1, min=0) + torch.clamp(f - 1, min=0)
    else:  # relu
        bad = torch.clamp(-f, min=0)
    return -bad.mean(dim=1)


def clamp_activation(feat, act):
    if act == "relu":
        return feat.clamp_min(0.0)
    if act == "sigmoid":
        return feat.clamp(0.0, 1.0)
    if act == "tanh":
        return feat.clamp(-1.0, 1.0)
    return feat


def invert_conv(model, feat_targets, steps=500, lr=0.05, tv=1e-4, base_res=64,
                dev="cpu", upsample_init=True):
    """Recover x in [0,1] s.t. act(conv(x)) matches feat_targets. Strided convs
    optimise a smooth low-res latent (base_res<64) upsampled to 64 (teammate's).
    OUR lever: seed z from the feature channel-mean (a low-freq face guess) instead
    of random noise (round-trip-validated +0.05..0.12 on the pooled models)."""
    model = model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    f_t = feat_targets.to(dev)
    n = f_t.shape[0]
    up = (lambda z: z) if base_res == 64 else \
        (lambda z: F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False))
    if upsample_init:
        gm = f_t.mean(1, keepdim=True)                 # (n,1,H,W) channel-mean guess
        gm = F.interpolate(gm, size=(base_res, base_res), mode="bilinear",
                           align_corners=False)
        gm = U.per_image_minmax(gm)                    # per-image [0,1]
        z = (gm.repeat(1, 3, 1, 1) * 0.6 + 0.2).clone().to(dev).requires_grad_(True)
    else:
        z = torch.rand(n, 3, base_res, base_res, device=dev, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = up(z).clamp(0, 1)
        loss = (model.conv_feature(x) - f_t).pow(2).mean() + tv * U.total_variation(x).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(0, 1)
    with torch.no_grad():
        return up(z).clamp(0, 1).detach().cpu()


def reconstruct(name, topk=None, steps=None, dev="cpu"):
    """Return (pool_images (M,3,64,64) in [0,1], meta) or (None, meta) for ViT."""
    topk = C.TOPK if topk is None else topk
    steps = C.CONV_STEPS if steps is None else steps
    idx = int(name.replace("model", ""))
    grad = U.load_grad(name); state = U.load_state(name); g = grad["gradients"]
    fam, act = grad["family"], grad["activation"]
    fshape = tuple(grad["feature_shape"])
    meta = {"name": name, "family": fam, "activation": act, "feature_shape": fshape}
    if fam == "vit":
        meta["note"] = "vit: analytic skipped, bank-fill only"
        return None, meta

    target = first_biased_linear(state, g)
    gW = g[target + ".weight"].float(); gb = g[target + ".bias"].float()
    in_dim = gW.shape[1]
    is_img = (in_dim == U.IMG_NUMEL)

    if is_img:  # ---- MLP: ratio rows ARE images ----
        k = C.MLP_LOWRANK_K.get(idx, 0)
        gW = lowrank_denoise(gW, k)                         # <-- OUR lever #1
        meta["lowrank_k"] = k
        ratio, reliable = ratio_rows(gW, gb)
        cand = ratio.reshape(-1, 3, 64, 64)                # RAW ratio
        # BUGFIX: score on the RAW ratio so rng_frac discriminates noisy (out-of-range)
        # rows -- per_image_minmax forces [0,1] and makes rng_frac==1 for ALL rows,
        # collapsing the noise filter (matches teammate attack_analytic: score raw,
        # normalise only the kept rows).
        st = U.image_stats(cand)
        score = st["rng_frac"] * torch.clamp(st["std"] / 0.15, max=1.0)
        score = torch.where(reliable & st["finite"], score, torch.zeros_like(score))
        keep = torch.argsort(score, descending=True)[:topk]
        return U.per_image_minmax(cand[keep]).clamp(0, 1).float(), meta   # normalise AFTER

    # ---- CNN: ratio rows are FEATURES -> score -> invert conv ----
    C_, H, W = fshape
    ratio, reliable = ratio_rows(gW, gb)
    feat = torch.nan_to_num(ratio).reshape(-1, C_, H, W)
    fscore = featureness(feat, act)                        # <-- OUR lever #2 (tanh)
    fscore = torch.where(reliable, fscore, torch.full_like(fscore, -1e9))
    keep = torch.argsort(fscore, descending=True)[:topk]
    targ = clamp_activation(feat[keep], act)
    model = build_cnn(name, meta, state)
    strided = model.stride > 1
    base_res = H if strided else 64
    meta.update(stride=model.stride, strided=strided, base_res=base_res)
    imgs = invert_conv(model, targ, steps=steps,
                       lr=0.03 if strided else 0.05,
                       tv=5e-3 if strided else 1e-4,
                       base_res=base_res, dev=dev, upsample_init=C.UPSAMPLE_INIT)
    return imgs.clamp(0, 1).float(), meta

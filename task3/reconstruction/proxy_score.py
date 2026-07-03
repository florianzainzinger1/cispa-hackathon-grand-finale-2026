"""Agent B — label-free-first PROXY scorer for Task 3 submissions/candidates.

We cannot see ground truth (per-model Hungarian-SSIM is leaderboard-only, 5-min
cooldown, 30% split). This ranks a candidate 128-image set per model by proxies
that correlate with true reconstruction quality, so we choose sources rationally
and never burn a submission on an unverified change.

Signals (per model, per source):
 1. CAPTURE  (label-free, PRIMARY fidelity signal): the first leaky layer's weight
    gradient gW = C @ X_true lies in row-space(X_true). We report the fraction of
    gW energy captured by span(rows of the candidate set X). Higher = the set better
    explains the true batch. Random 128 faces ~0.01-0.3; the true/analytic set ~0.77.
      MLP  : X = flattened images        vs  net.0.weight-grad (1024 x 12288)
      CNN  : X = conv_feature(images)     vs  fc1.weight-grad   (1024 x C*H*W)
      ViT  : no usable linear leak -> capture is None (bank is the provable floor).
 2. GRAD-COS (secondary, CONFOUNDED): re-feed the 128 imgs with inferred labels,
    cosine of produced-vs-observed gradient (first-layer keys + full). The image<->
    label pairing is unknown, so this is noisier than capture; use it only relatively.
 3. DIVERSITY: mean nearest-neighbour distance (16x16 gray) + #distinct after dedup.
 4. GRAIN   : mean TV + fraction that pass the face/noise (is_real) filter. SSIM
    punishes grain (c5 regressed by shipping grainier real faces) -> grain is a COST.

Decision rule Agent B enforces (GT-free, avoids the c5 mistake): only prefer a new
source over the incumbent if it PARETO-DOMINATES (capture >=, grain <=, diversity >=)
OR is a verified verbatim retrieval. `proxy_scalar` is only a tie-break hint.

Login-node safe: CPU, single forward/backward per model, numpy SVD. No optimisation.

CLI:
  python proxy_score.py                      # score outputs_final/submission_current_best.pt
  python proxy_score.py <a.pt> <b.pt> ...    # score & compare several submissions
Writes reports/build/proxy_scoreboard.json (+ prints a table).
"""
import os
import sys
import json
import numpy as np
import torch

import utils as U
import infer_labels as IL
from attack_opt import build_any
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# first leaky layer per family
# ---------------------------------------------------------------------------
def leaky_layer(name, grad, state):
    """Return (family, weight_key, bias_key, in_dim) of the first analytically
    leaky layer. MLP -> net.0 (image space); CNN -> fc1 (conv-feature space);
    ViT -> patch_embed.proj (weak/None)."""
    fam = grad["family"]
    g = grad["gradients"]
    if fam == "mlp":
        return fam, "net.0.weight", "net.0.bias", U.IMG_NUMEL
    if fam == "cnn":
        return fam, "fc1.weight", "fc1.bias", None
    return fam, "patch_embed.proj.weight", "patch_embed.proj.bias", None


# ---------------------------------------------------------------------------
# capture: fraction of gW energy inside span(rows of X)
# ---------------------------------------------------------------------------
def _captured(gW, X, tol=1e-3):
    """gW:(m,d) np, X:(n,d) np -> (fraction, effective_rank_of_X)."""
    _, S, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return 0.0, 0
    r = int((S > S[0] * tol).sum())
    V = Vt[:r]                                    # (r,d) orthonormal
    proj = gW.astype(np.float64) @ V.T            # (m,r)
    denom = float((gW.astype(np.float64) ** 2).sum()) + 1e-30
    return float((proj ** 2).sum() / denom), r


def capture_metric(name, imgs, grad, model):
    """Family-aware capture. Returns (capture, rank) or (None, None) for ViT."""
    fam, wkey, bkey, _ = leaky_layer(name, grad, None)
    g = grad["gradients"]
    if wkey not in g:
        return None, None
    gW = g[wkey].float().numpy()
    if fam == "mlp":
        X = imgs.reshape(imgs.shape[0], -1).float().numpy()
        return _captured(gW, X)
    if fam == "cnn":
        with torch.no_grad():
            feat = model.conv_feature(imgs.float())          # (128,C,H,W)
        X = feat.reshape(feat.shape[0], -1).numpy()
        return _captured(gW, X)
    return None, None                                        # vit: no real leak


# ---------------------------------------------------------------------------
# gradient self-consistency (secondary; label pairing unknown)
# ---------------------------------------------------------------------------
def grad_cosine(name, imgs, model, grad):
    labels = IL.assign_labels(name, imgs.shape[0]).long()
    name2param = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in name2param]
    if not keys:
        return None, None
    params = [name2param[k] for k in keys]
    for p in model.parameters():
        p.requires_grad_(True)
    logits = model(imgs.float())
    loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    real = grad["gradients"]

    def cos(sel_keys):
        dv, rv = [], []
        for k, gd in zip(keys, grads):
            if k not in sel_keys or gd is None:
                continue
            dv.append(gd.reshape(-1).float())
            rv.append(real[k].reshape(-1).float())
        if not dv:
            return None
        d = torch.cat(dv); r = torch.cat(rv)
        return float((d @ r) / (d.norm() * r.norm() + 1e-12))

    fam, wkey, bkey, _ = leaky_layer(name, grad, None)
    first_keys = {wkey, bkey}
    return cos(first_keys), cos(set(keys))


# ---------------------------------------------------------------------------
# diversity + grain
# ---------------------------------------------------------------------------
def _feat16(imgs):
    g = imgs.float().mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(16, 16), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1)


def diversity_metric(imgs, dedup_eps=0.03):
    f = _feat16(imgs)
    D = torch.cdist(f, f)
    D.fill_diagonal_(1e9)
    mean_nn = float(D.min(dim=1).values.mean())
    # greedy distinct count at eps (RMS distance)
    order = list(range(imgs.shape[0]))
    chosen = []
    for i in order:
        ok = True
        for c in chosen:
            if float((f[i] - f[c]).pow(2).mean().sqrt()) < dedup_eps:
                ok = False
                break
        if ok:
            chosen.append(i)
    return mean_nn, len(chosen)


def grain_metric(imgs):
    st = U.image_stats(imgs.float())
    real = st["finite"] & (st["tv"] < 0.20) & (st["rng_frac"] > 0.55)
    return float(st["tv"].mean()), float(real.float().mean())


# ---------------------------------------------------------------------------
# per-model scoring (cacheable model/grad build)
# ---------------------------------------------------------------------------
_CACHE = {}


def _model_grad(name):
    if name not in _CACHE:
        model, grad = build_any(name)
        _CACHE[name] = (model, grad)
    return _CACHE[name]


def score_images(name, imgs, want_gradcos=True):
    """Return the full proxy dict for one model's 128 images."""
    imgs = imgs.float().clamp(0, 1)
    model, grad = _model_grad(name)
    fam = grad["family"]
    cap, rank = capture_metric(name, imgs, grad, model)
    gc_first = gc_full = None
    if want_gradcos:
        try:
            gc_first, gc_full = grad_cosine(name, imgs, model, grad)
        except Exception as e:
            gc_first = gc_full = None
    mean_nn, n_distinct = diversity_metric(imgs)
    tv, frac_real = grain_metric(imgs)
    return {
        "family": fam,
        "capture": cap, "capture_rank": rank,
        "grad_cos_first": gc_first, "grad_cos_full": gc_full,
        "mean_nn_dist": mean_nn, "n_distinct": n_distinct,
        "mean_tv": tv, "frac_real": frac_real,
        "proxy_scalar": proxy_scalar(fam, cap, tv, mean_nn, frac_real),
    }


def proxy_scalar(fam, cap, tv, mean_nn, frac_real):
    """A single hint number (NOT the decision — Pareto rule decides). Rewards
    capture (fidelity) and diversity, penalises grain. For ViT (no capture) it
    falls back to diversity minus grain (bank-like sets score well)."""
    grain_pen = 1.5 * max(0.0, tv - 0.10)          # only penalise above face-TV
    if cap is None:
        return round(mean_nn - grain_pen, 4)
    return round(cap + 0.25 * mean_nn - grain_pen, 4)


def dominates(a, b, eps=1e-3):
    """Does source metrics `a` Pareto-dominate `b`? (higher capture, lower grain,
    >= diversity, without being worse on any). Retrieval override handled by caller."""
    ca = a["capture"] if a["capture"] is not None else -1
    cb = b["capture"] if b["capture"] is not None else -1
    better_cap = ca >= cb - eps
    better_grain = a["mean_tv"] <= b["mean_tv"] + eps
    better_div = a["mean_nn_dist"] >= b["mean_nn_dist"] - eps
    strictly = (ca > cb + eps) or (a["mean_tv"] < b["mean_tv"] - eps) or \
               (a["mean_nn_dist"] > b["mean_nn_dist"] + eps)
    return better_cap and better_grain and better_div and strictly


# ---------------------------------------------------------------------------
# CLI: score submission files
# ---------------------------------------------------------------------------
def load_sub(path):
    return torch.load(path, weights_only=False)


def main():
    paths = sys.argv[1:] or [U.work_path("outputs_final", "submission_current_best.pt")]
    boards = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        tag = os.path.basename(path)
        sub = load_sub(path)
        board = {}
        print(f"\n=== {tag} ===")
        hdr = f"{'model':8} {'fam':4} {'capture':>8} {'rank':>5} {'gcos1':>7} " \
              f"{'nn':>6} {'dist':>4} {'tv':>6} {'real':>5} {'scalar':>7}"
        print(hdr)
        for name in U.MODELS:
            if name not in sub:
                continue
            m = score_images(name, sub[name])
            board[name] = m
            capf = f"{m['capture']:.4f}" if m["capture"] is not None else "  -   "
            gcf = f"{m['grad_cos_first']:.3f}" if m["grad_cos_first"] is not None else "  -  "
            print(f"{name:8} {m['family']:4} {capf:>8} {str(m['capture_rank'] or '-'):>5} "
                  f"{gcf:>7} {m['mean_nn_dist']:6.3f} {m['n_distinct']:>4} "
                  f"{m['mean_tv']:6.3f} {m['frac_real']:5.2f} {m['proxy_scalar']:7.3f}")
        boards[tag] = board
    out = U.work_path("reports", "build", "proxy_scoreboard.json")
    U.save_json(boards, out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()

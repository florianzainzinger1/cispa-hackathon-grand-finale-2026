"""Is there headroom for 'fundamentally better low-frequency reconstruction' on the
signal-starved CNN models? Measured OFFLINE on the corrected per-channel oracle,
against the replay-certified GT (models 10 & 12 are covered).

For each covered signal-starved model, report:
  - current best block's oracle SSIM (where we are)
  - blur_s(GT) vs GT: the ceiling a SMOOTH recon reaches IF it had the true low-freq
  - low-rank conv-feature recon vs GT (what the gradient actually permits)
Headroom = (best smooth ceiling) - (current). If ~0, kill the lever; if large, build it.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import utils as U
import models as M
import ssim_grader as G
import ssim_validate as V
import rec_lib as R
import retr_lib as RL

CONV = "perch7"
SIGNAL_STARVED = ["model10", "model12", "model6", "model2", "model3", "model7"]


def block(d, m):
    e = d[m]
    return (e["images"] if isinstance(e, dict) else e).float().clamp(0, 1)


def blur(x, s):
    if s <= 0:
        return x
    out = x.clone().numpy()
    for i in range(out.shape[0]):
        for c in range(3):
            out[i, c] = gaussian_filter(out[i, c], s)
    return torch.tensor(out)


def lowrank_feature_recon(name, k=None):
    """ratio rows of fc1 -> rank-k subspace denoise -> conv-invert -> smooth RGB."""
    g = U.load_grad(name); state = U.load_state(name)
    meta = {"family": "cnn", "activation": g["activation"], "feature_shape": g["feature_shape"]}
    model = M.build_model(name, meta, state)
    for p in model.parameters():
        p.requires_grad_(False)
    grads = g["gradients"]
    gW = grads["fc1.weight"].float(); gb = grads["fc1.bias"].float()
    fs = tuple(int(s) for s in g["feature_shape"])
    ratio, reliable = R.ratio_rows(gW, gb)
    idx = torch.nonzero(reliable & torch.isfinite(ratio).all(1)).flatten()
    rows = ratio[idx]
    # rank-k right-singular subspace denoise of gW, applied to the rows
    if k is None:
        s = torch.linalg.svdvals(gW.float())
        k = int((s > 0.02 * s[0]).sum())            # effective rank
    V_, _ = RL.topk_right_subspace(gW, k=max(4, min(k, gW.shape[0])))
    rows_d = (rows @ V_.t()) @ V_                    # denoised features
    feat = rows_d.reshape(-1, *fs)
    # conv-invert (optimization) -> RGB, then heavy smooth
    from synth_lib import conv_invert_opt
    n = min(feat.shape[0], 128)
    rgb = conv_invert_opt(feat[:n], model, iters=120, lr=0.05)
    return rgb.clamp(0, 1), k


def main():
    gt = torch.load(V.GT_PATH, weights_only=False, map_location="cpu")
    matches = V.match_gt(gt)
    best = torch.load(U.work_path("outputs_final", "submission_current_best.pt"),
                      weights_only=False, map_location="cpu")
    for m in SIGNAL_STARVED:
        if m not in matches:
            print(f"{m}: no GT match (can't measure offline)", flush=True)
            continue
        GTb = gt[matches[m]]["images"].float().clamp(0, 1)
        cur = block(best, m)
        s_cur = G.hungarian_ssim(cur, GTb, CONV)
        print(f"\n=== {m} (GT='{matches[m]}') current_block={s_cur:.4f} ===", flush=True)
        for s in [0.5, 1.0, 1.5, 2.0]:
            sc = G.hungarian_ssim(blur(GTb, s), GTb, CONV)
            print(f"   ceiling blur{s}(GT)vsGT = {sc:.4f}  headroom={sc - s_cur:+.4f}", flush=True)
        try:
            rec, k = lowrank_feature_recon(m)
            if rec.shape[0] < 128:
                rec = torch.cat([rec, cur[rec.shape[0]:128]], 0)   # fill from current
            for s in [0.0, 1.0, 1.5]:
                sc = G.hungarian_ssim(blur(rec, s), GTb, CONV)
                print(f"   lowrank(k={k}) recon blur{s} = {sc:.4f}  vs current={s_cur:.4f} "
                      f"({'WINS' if sc > s_cur else 'loses'} {sc - s_cur:+.4f})", flush=True)
        except Exception as e:
            print(f"   lowrank recon failed: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()

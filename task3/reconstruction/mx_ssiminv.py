"""mx_ssiminv.py — SSIM-optimal (max-smooth) conv-inversion for strided CNNs model3/model7.

The existing conv-invert (tv=5e-3, base_res=feature_res) still yields heavy RGB
checkerboard speckle — the SSIM killer. Here we push for the SMOOTHEST image
consistent with the exact fc1 features:
  min_z || act(conv(up(z))) - f ||^2 + TV_STRONG * TV(up(z))
optimising a low-res latent z (base_res = feature resolution), then a Gaussian
low-pass finish. This is low-freq-only recovery (no detail chasing = the c4 trap).

Then build candidates = vithi45 base with ONLY model3 / model7 swapped, dedup to
128 distinct rows by quality, colour/brightness-match to CelebA (~0.44).
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
sys.path.insert(0, f"{WORK}/code")
import utils as U
import models as M
import rec_lib as R

REF_MEAN = np.array([0.5146, 0.4229, 0.3754], dtype=np.float32)
REF_STD  = np.array([0.3061, 0.2837, 0.2809], dtype=np.float32)
REF_GMEAN = 0.4376
BASE = f"{WORK}/outputs_final/test_vithi45.pt"


def build_model(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state)
    return model, grad


def load_targets(name, grad, topk=256):
    fd = torch.load(U.work_path("candidates", name, "analytic_features.pt"), weights_only=False)
    feat = fd["feature"]
    rows = torch.load(U.work_path("candidates", name, "analytic_candidates.pt"),
                      weights_only=False)["rows"][:topk]
    targ = feat[rows]
    act = grad["activation"]
    if act == "relu":
        targ = targ.clamp_min(0.0)
    elif act == "sigmoid":
        targ = targ.clamp(0.0, 1.0)
    elif act == "tanh":
        targ = targ.clamp(-1.0, 1.0)
    return targ, rows


def invert_smooth(model, feat_targets, base_res, tv, steps=600, lr=0.03,
                  device="cuda", lp_sigma=0.6):
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    f_t = feat_targets.to(device)
    n = f_t.shape[0]

    def up(z):
        return F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)

    z = torch.rand(n, 3, base_res, base_res, device=device)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        x = up(z).clamp(0, 1)
        f = model.conv_feature(x)
        loss = (f - f_t).pow(2).mean() + tv * U.total_variation(x).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(0, 1)
    with torch.no_grad():
        out = up(z).clamp(0, 1).detach().cpu().numpy()  # (n,3,64,64)
    # Gaussian low-pass finish (kill any residual grain). per img per channel.
    if lp_sigma > 0:
        for i in range(out.shape[0]):
            for c in range(3):
                out[i, c] = gaussian_filter(out[i, c], sigma=lp_sigma, mode="nearest")
    return np.clip(out, 0, 1)


def color_match(imgs, strength=0.8, target_g=REF_GMEAN):
    # imgs: (n,3,64,64) numpy. per-channel mean/std match + brightness lift.
    a = imgs.transpose(0, 2, 3, 1)  # n,H,W,3
    flat = a.reshape(-1, 3)
    cur_m, cur_s = flat.mean(0), flat.std(0) + 1e-6
    tgt_s = cur_s + (REF_STD - cur_s) * strength
    scale = np.minimum(tgt_s / cur_s, 1.3)
    tgt_m = cur_m + (REF_MEAN - cur_m) * strength
    a = (a - cur_m) * scale + tgt_m
    # per-image brightness toward target
    for i in range(a.shape[0]):
        m = a[i].mean()
        if m > 1e-4:
            g = min(1.0 + (target_g / m - 1.0) * 0.5, 1.6)
            a[i] = a[i] * g
    a = np.clip(a, 0, 1)
    return a.transpose(0, 3, 1, 2)


def pick_128(imgs):
    """Pick 128 distinct rows by quality (highest quality_score), guarantee distinct."""
    t = torch.from_numpy(imgs).float()
    q = np.array([R.quality_score(t[i]) for i in range(t.shape[0])])
    order = np.argsort(-q)
    chosen, seen = [], set()
    for idx in order:
        key = tuple(np.round(t[idx].reshape(-1)[:64].numpy(), 3))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(idx)
        if len(chosen) == 128:
            break
    # top up if needed
    i = 0
    while len(chosen) < 128:
        if order[i] not in chosen:
            chosen.append(order[i])
        i += 1
    return t[chosen[:128]]


def run(name, out_name, base_res, tv, lp_sigma, steps, device):
    model, grad = build_model(name)
    targ, rows = load_targets(name, grad, topk=256)
    print(f"{name}: feat_shape={grad['feature_shape']} act={grad['activation']} "
          f"stride={model.stride} base_res={base_res} tv={tv} lp={lp_sigma} n_targ={targ.shape[0]}")
    rec = invert_smooth(model, targ, base_res=base_res, tv=tv, steps=steps,
                        device=device, lp_sigma=lp_sigma)
    rec = color_match(rec, strength=0.8)
    imgs128 = pick_128(rec)  # (128,3,64,64) tensor
    # build candidate
    sub = torch.load(BASE, map_location="cpu", weights_only=False)
    mk = name.replace("model", "model")  # e.g. model3
    sub[mk] = imgs128.float().clamp(0, 1).contiguous()
    outp = f"{WORK}/outputs_mx/{out_name}"
    torch.save(sub, outp)
    print(f"[saved] {outp}  {mk} gmean={imgs128.mean():.4f} "
          f"chan_mean={imgs128.mean(dim=(0,2,3)).numpy().round(3)}")
    # montage
    U.save_image_grid(imgs128[:64], f"{WORK}/reports/inspect_new/{out_name.replace('.pt','')}.png",
                      ncol=8, pad=1, upscale=2)
    print(f"[montage] reports/inspect_new/{out_name.replace('.pt','')}.png")
    return outp


def main():
    device = os.environ.get("FL_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device, "cuda_avail:", torch.cuda.is_available())
    # model3: stride8, feature 8x8 -> most starved. Strong smooth. Also a milder variant.
    run("model3", "cand_m3_ssiminv.pt",  base_res=8,  tv=2e-2, lp_sigma=0.8, steps=700, device=device)
    # model7: stride4, feature 16x16. Moderate.
    run("model7", "cand_m7_ssiminv.pt",  base_res=16, tv=1.5e-2, lp_sigma=0.6, steps=700, device=device)
    # model2: stride4, feature 16x16 — already decent in vithi45, but produce for completeness.
    run("model2", "cand_m2_ssiminv.pt",  base_res=16, tv=1.5e-2, lp_sigma=0.6, steps=700, device=device)
    print("DONE ssiminv")


if __name__ == "__main__":
    main()

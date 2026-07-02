"""Stage 2b: invert the first conv to turn CNN feature candidates into RGB images.

fc1 analytic gives feature candidates f ~= act(conv(x)). We recover x by
projected-gradient optimisation:  min_x || act(conv(x)) - f ||^2 + tv*TV(x),
x in [0,1]. Full-res convs (stride 1) invert sharply; strided convs give a
blurrier estimate constrained only at sampled positions.
"""
import os
import sys
import torch
import torch.nn.functional as F

import utils as U
import models as M


def invert_conv(model, feat_targets, steps=400, lr=0.05, tv=1e-4,
                device="cpu", init=None, base_res=64, log_every=0):
    """feat_targets: (n,C,H,W). Returns recovered images (n,3,64,64) in [0,1].

    For strided convs the image is underdetermined at full 64x64 and direct
    optimisation produces checkerboard artifacts. Setting base_res < 64
    optimises a smooth low-res latent that is bilinearly upsampled to 64 before
    the conv, yielding a blurry-but-clean face (better SSIM than checkerboard).
    """
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    f_t = feat_targets.to(device)
    n = f_t.shape[0]

    def up(z):
        if base_res == 64:
            return z
        return F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)

    if init is None:
        z = torch.rand(n, 3, base_res, base_res, device=device)
    else:
        z = init.to(device).clone()
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
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"    invert step {it:4d} loss={loss.item():.5f}")
    with torch.no_grad():
        out = up(z).clamp(0, 1)
    return out.detach().cpu()


def self_test(model, feature_shape, device="cpu"):
    """Invert a known random image through the model's conv; report recovery."""
    torch.manual_seed(0)
    model = model.to(device)
    true = torch.rand(4, 3, 64, 64)
    with torch.no_grad():
        f = model.conv_feature(true.to(device)).cpu()
    rec = invert_conv(model, f, steps=400, lr=0.05, tv=0.0, device=device)
    err = (rec - true).abs().mean().item()
    return err


def run_model(name, device="cpu", steps=400, topk=256):
    grad = U.load_grad(name)
    if grad.get("family") != "cnn":
        print(f"{name}: not cnn, skip conv-invert")
        return
    state = U.load_state(name)
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state)
    ok, oshape = M.validate_forward(model, grad["feature_shape"])
    st_err = self_test(model, grad["feature_shape"], device)
    print(f"{name}: forward_ok={ok} out={oshape} conv=(s{model.stride},p{model.padding}) "
          f"self_test_L1={st_err:.4f}")

    feat_path = U.work_path("candidates", name, "analytic_features.pt")
    if not os.path.exists(feat_path):
        print(f"{name}: no analytic_features.pt (run attack_analytic first)")
        return
    fd = torch.load(feat_path, weights_only=False)
    feat = fd["feature"]                       # (1024,C,H,W) raw ratio
    reliable = fd["reliable"]
    # rank rows by feature energy/among reliable (mirror analytic scoring)
    cand_path = U.work_path("candidates", name, "analytic_candidates.pt")
    rows = torch.load(cand_path, weights_only=False)["rows"][:topk]
    targ = feat[rows]
    # match activation range of the target to the model's activation
    act = grad["activation"]
    if act == "relu":
        targ = targ.clamp_min(0.0)
    elif act == "sigmoid":
        targ = targ.clamp(0.0, 1.0)
    elif act == "tanh":
        targ = targ.clamp(-1.0, 1.0)

    # strided convs: optimise a smooth low-res latent (= feature resolution)
    base_res = 64 if model.stride == 1 else grad["feature_shape"][1]
    rec = invert_conv(model, targ, steps=steps, lr=0.05, tv=1e-4,
                      device=device, base_res=base_res)
    print(f"{name}: base_res={base_res}")
    out_dir = U.work_path("candidates", name)
    torch.save({"images": rec, "rows": rows, "source": "conv_invert"},
               os.path.join(out_dir, "conv_invert_candidates.pt"))
    U.save_image_grid(rec[:64], os.path.join(out_dir, "grid_conv_invert.png"))
    print(f"{name}: inverted {rec.shape[0]} -> {out_dir}/grid_conv_invert.png")


def main():
    # Default to CPU (login-node safe). sbatch sets FL_DEVICE=cuda for GPU runs.
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device)
    names = sys.argv[1:] or ["model6", "model10", "model12", "model2", "model7", "model3"]
    for name in names:
        try:
            run_model(name, device=device)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

"""BSS (agent3 / lever L3) data probe.

Grounds the source-separation implementation in the ACTUAL gradient tensors:
for each honest MLP/CNN model it reports the first-layer weight/bias gradient
shapes, the bias-gradient magnitude distribution (scale-fix relies on gb), the
singular spectrum of gW (rank / noise floor / top-128 energy), and renders the
top right-singular vectors as "eigenfaces" to confirm the 128 source images
span rowspace(gW) = the BSS source subspace.

Login-node safe: pure SVD + tensor algebra, one short single pass. No GPU.
Namespace: bss_*  ->  reports/bss/inspect/.
"""
import os
import sys
import torch

import utils as U
import attack_analytic as AA


MLP = ["model1", "model4", "model5", "model8"]
CNN = ["model6", "model10", "model12", "model2", "model7", "model3"]


def q(t, ps=(0.0, 0.5, 0.9, 0.99, 1.0)):
    t = t.flatten().float()
    return {f"p{int(p*100)}": float(torch.quantile(t, p)) for p in ps}


def spectrum(gW, B=128):
    gW = gW.float()
    # economy SVD: gW (h,d) -> U(h,k) S(k) Vh(k,d), k=min(h,d)=h=1024
    Uu, S, Vh = torch.linalg.svd(gW, full_matrices=False)
    S = S.float()
    total_e = float((S ** 2).sum())
    e_topB = float((S[:B] ** 2).sum()) / (total_e + 1e-12)
    # participation ratio (effective rank): (sum s^2)^2 / sum s^4
    eff_rank = float((S ** 2).sum() ** 2 / ((S ** 4).sum() + 1e-12))
    idx = [0, 1, 31, 63, 127, 128, 199, 255, 511, min(959, S.numel() - 1)]
    sv = {f"sv[{i}]": float(S[i]) for i in idx if i < S.numel()}
    noise_floor = float(S[B] / (S[0] + 1e-12)) if S.numel() > B else 0.0
    return S, Vh, {
        "sv": sv,
        "energy_top128_frac": e_topB,
        "eff_rank_participation": eff_rank,
        "noise_floor_sv128_over_sv0": noise_floor,
    }


def eigenfaces(Vh, name, is_image_layer, feat_shape, out_dir, k=16):
    """Render top-k right singular vectors as images (MLP: direct 3x64x64;
    CNN: reshape to feature map, show first 3 channels)."""
    top = Vh[:k]
    if is_image_layer:
        imgs = top.reshape(k, U.IMG_C, U.IMG_H, U.IMG_W)
    else:
        C, H, W = feat_shape
        f = top.reshape(k, C, H, W)
        imgs = AA.feature_to_rgb_view(f)
        U.save_image_grid(U.per_image_minmax(imgs), os.path.join(out_dir, f"{name}_eigenfeat.png"), ncol=8, upscale=3)
        return
    U.save_image_grid(U.per_image_minmax(imgs), os.path.join(out_dir, f"{name}_eigenfaces.png"), ncol=8, upscale=3)


def run_model(name, out_dir):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    family = grad.get("family")
    act = grad.get("activation")
    feat_shape = tuple(grad.get("feature_shape")) if grad.get("feature_shape") else None

    target = AA.first_leaky_linear(name, state, g)
    gW = g[target + ".weight"].float()
    gb = g[target + ".bias"].float()
    in_dim = gW.shape[1]
    is_image_layer = in_dim == U.IMG_NUMEL

    S, Vh, spec = spectrum(gW)
    eigenfaces(Vh, name, is_image_layer, feat_shape, out_dir)

    info = {
        "name": name, "family": family, "activation": act,
        "target_layer": target,
        "gW_shape": list(gW.shape), "gb_shape": list(gb.shape),
        "in_dim": int(in_dim), "is_image_layer": bool(is_image_layer),
        "feature_shape": list(feat_shape) if feat_shape else None,
        "feat_prod_eq_indim": (feat_shape is not None and feat_shape[0]*feat_shape[1]*feat_shape[2] == in_dim) if feat_shape else None,
        "gW_fro": float(gW.norm()),
        "gb_abs": q(gb.abs()),
        "gb_reliable_1e-8": int((gb.abs() > 1e-8).sum()),
        "gb_reliable_1e-6": int((gb.abs() > 1e-6).sum()),
        "gb_signed": {"n_pos": int((gb > 0).sum()), "n_neg": int((gb < 0).sum()), "n_zero": int((gb == 0).sum())},
        **spec,
    }
    U.save_json(info, os.path.join(out_dir, f"{name}.json"))
    print(f"{name} [{family}/{act}] target={target} gW={list(gW.shape)} "
          f"img_layer={is_image_layer} feat={feat_shape}\n"
          f"    E@128={spec['energy_top128_frac']:.3f}  eff_rank~{spec['eff_rank_participation']:.0f}  "
          f"noise_floor(sv128/sv0)={spec['noise_floor_sv128_over_sv0']:.3f}\n"
          f"    gb: |gb| p50={info['gb_abs']['p50']:.2e} p99={info['gb_abs']['p99']:.2e} "
          f"reliable(>1e-8)={info['gb_reliable_1e-8']}/1024  pos/neg={info['gb_signed']['n_pos']}/{info['gb_signed']['n_neg']}")
    return info


def main():
    names = sys.argv[1:] or (MLP + CNN)
    out_dir = U.work_path("reports", "bss", "inspect")
    U.ensure_dir(out_dir)
    summary = {}
    for name in names:
        try:
            summary[name] = run_model(name, out_dir)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(summary, os.path.join(out_dir, "_summary.json"))
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()

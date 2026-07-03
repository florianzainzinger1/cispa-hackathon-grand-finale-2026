"""READ-ONLY probe 3 (agent3): TEST B from the literature audit's #1 lever ---
does rank-128 (and joint [gW|gb]) SVD DENOISE of the first-layer gradient BEFORE
the analytic ratio / conv-inversion rescue the noise-defended models (m6, m8, MLP)?
Measures capture + separation. Touches no pipeline/submission.
"""
import os
import numpy as np
import torch

import utils as U
import models as M
import attack_analytic as AA
import attack_conv_invert as CI

DEV = os.environ.get("FL_DEVICE", "cpu")
if DEV == "cuda" and not torch.cuda.is_available():
    DEV = "cpu"


def rank_project(Mt, r):
    Uu, S, Vh = torch.linalg.svd(Mt, full_matrices=False)
    r = min(r, S.numel())
    return (Uu[:, :r] * S[:r]) @ Vh[:r]


def joint_project(gW, gb, r):
    G = torch.cat([gW, gb.unsqueeze(1)], dim=1)
    Gr = rank_project(G, r)
    return Gr[:, :-1], Gr[:, -1]


def captured(gW, X, tol=1e-3):
    gW = gW.double().cpu().numpy(); X = X.double().cpu().numpy()
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return 0.0
    r = int((S > S[0] * tol).sum())
    V = Vt[:r]
    proj = gW @ V.T
    return float((proj ** 2).sum() / ((gW ** 2).sum() + 1e-30))


def eff_unique(X):
    Xf = X.reshape(X.shape[0], -1)
    Xn = torch.nn.functional.normalize(Xf, dim=1)
    sv = torch.linalg.svdvals(Xn)
    return int((sv > 1e-2 * sv[0]).sum())


def mlp_capture(name):
    grad = U.load_grad(name); state = U.load_state(name); g = grad["gradients"]
    base = AA.first_leaky_linear(name, state, g)
    gW = g[base + ".weight"].float(); gb = g[base + ".bias"].float()
    out = {}
    for tag, (W, B) in {
        "raw": (gW, gb),
        "rank128_gW": (rank_project(gW, 128), gb),
        "joint128": joint_project(gW, gb, 128),
    }.items():
        ratio, rel = AA.ratio_candidates(W, B)
        imgs = torch.nan_to_num(ratio).reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W)
        # rank rows by in-range/contrast; take top-128
        st = U.image_stats(imgs)
        score = torch.where(rel & st["finite"], st["rng_frac"] * (st["std"] / 0.15).clamp(max=1), torch.zeros(imgs.shape[0]))
        top = imgs[torch.argsort(score, descending=True)[:128]].clamp(0, 1)
        cap = captured(gW, top.reshape(128, -1))
        out[tag] = {"capture": round(cap, 4), "eff_unique": eff_unique(top)}
    print(f"[MLP {name}] " + "  ".join(f"{t}: cap={v['capture']} uniq={v['eff_unique']}" for t, v in out.items()))
    return out


def cnn_capture(name, out_dir):
    grad = U.load_grad(name); state = U.load_state(name); g = grad["gradients"]
    base = AA.first_leaky_linear(name, state, g)  # fc1
    gW = g[base + ".weight"].float(); gb = g[base + ".bias"].float()
    C, H, Wd = grad["feature_shape"]
    meta = {"family": "cnn", "activation": grad["activation"], "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state).to(DEV)
    strided = model.stride > 1
    out = {}
    for tag, (W, B) in {
        "raw": (gW, gb),
        "rank128_gW": (rank_project(gW, 128), gb),
        "joint128": joint_project(gW, gb, 128),
    }.items():
        ratio, rel = AA.ratio_candidates(W, B)
        feat = torch.nan_to_num(ratio).reshape(-1, C, H, Wd)
        # top-128 reliable feature rows by energy
        en = feat.reshape(feat.shape[0], -1).abs().mean(1)
        en = torch.where(rel, en, torch.zeros_like(en))
        idx = torch.argsort(en, descending=True)[:128]
        targ = feat[idx]
        if grad["activation"] == "relu":
            targ = targ.clamp_min(0)
        rec = CI.invert_conv(model, targ.to(DEV), steps=300, lr=0.05, tv=1e-4,
                             device=DEV, base_res=64 if not strided else H)
        # capture: recon images -> m's conv feature -> fraction of RAW gW spanned
        with torch.no_grad():
            f = model.conv_feature(rec.to(DEV)).reshape(128, -1)
        cap = captured(gW, f)
        out[tag] = {"capture": round(cap, 4), "eff_unique": eff_unique(rec)}
        U.save_image_grid(rec[:64], os.path.join(out_dir, f"{name}_{tag}.png"))
        print(f"[CNN {name}] {tag}: cap={cap:.4f} uniq={out[tag]['eff_unique']}")
    return out


def main():
    out_dir = U.work_path("reports", "build", "probe3")
    U.ensure_dir(out_dir)
    rep = {}
    for n in ["model1", "model4", "model5", "model8"]:
        rep[n] = mlp_capture(n)
    for n in ["model6"]:
        rep[n] = cnn_capture(n, out_dir)
    U.save_json(rep, U.work_path("reports", "build", "verify_probe3.json"))
    print("\nInterpretation: if rank128/joint capture >> raw AND eff_unique stays ~128 (not collapsed),")
    print("gradient-domain denoise is a real lever. If capture ~flat or unique collapses, it is not.")
    print("wrote reports/build/verify_probe3.json + grids in probe3/")


if __name__ == "__main__":
    main()

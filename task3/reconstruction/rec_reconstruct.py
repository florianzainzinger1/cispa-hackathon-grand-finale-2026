"""Agent1 — analytic reconstruction with intra-cluster differencing denoise.

Per model (arg): ratio on the first biased linear (net.0 MLP -> image; fc1 CNN ->
feature), KMeans-cluster the reliable rows into <=128 sample groups, take a
differencing-denoised representative per cluster, collision-reject, (CNN) invert
the conv to RGB, assemble exactly 128 (fill shortfalls with best remaining ratio
candidates, never noise), certify by subspace residual, save
outputs_trap/rec_{model}.pt + a grid in reports/trap/{model}/.

Env: FL_DEVICE, REC_STEPS (conv-invert steps), REC_K (max clusters=128).
"""
import os, sys, json, time
import numpy as np
import torch

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
import utils as U
import models as M
from attack_conv_invert import invert_conv
from retr_lib import topk_right_subspace, subspace_residual
import rec_lib as R

DEVICE = os.environ.get("FL_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
STEPS = int(os.environ.get("REC_STEPS", "500"))
KMAX = int(os.environ.get("REC_K", "128"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def first_biased_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            bkey = pname[:-7] + ".bias"
            if bkey in g and g[bkey] is not None:
                return pname[:-7]
    return None


def clamp_activation(feat, act):
    if act == "relu":
        return feat.clamp_min(0.0)
    if act == "sigmoid":
        return feat.clamp(0.0, 1.0)
    if act == "tanh":
        return feat.clamp(-1.0, 1.0)
    return feat


def reconstruct(name):
    grad = U.load_grad(name); state = U.load_state(name); g = grad["gradients"]
    fam = grad["family"]; act = grad["activation"]; fshape = tuple(grad["feature_shape"])
    target = first_biased_linear(state, g)
    gW = g[target + ".weight"].float(); gb = g[target + ".bias"].float()
    if os.environ.get("REC_DENOISE") == "1":
        # rank-128 truncation of gW: drop the out-of-signal noise subspace (helps
        # heavily-noise-defended models 6/8; research L2/v2 finding).
        Vd, _ = topk_right_subspace(gW, 128)
        gW = (gW @ Vd.t()) @ Vd
        log(f"{name}: applied rank-128 gW denoise")
    ratio, reliable = R.ratio_rows(gW, gb)
    in_dim = ratio.shape[1]
    is_img = (in_dim == U.IMG_NUMEL)
    finite = torch.isfinite(ratio).all(1)
    relidx = torch.where(reliable & finite)[0]
    log(f"{name}: fam={fam} act={act} target={target} in_dim={in_dim} "
        f"reliable_rows={len(relidx)} is_img={is_img} feat={fshape} dev={DEVICE}")

    # cluster reliable rows (normalise per-row for direction clustering)
    R_rows = ratio[relidx]
    labels, k = R.cluster_labels(R.per_image_minmax(R_rows).reshape(len(relidx), -1), KMAX)
    log(f"{name}: clustered {len(relidx)} rows -> {k} clusters")

    # per-cluster differencing-denoised representative (in layer-input space)
    reps, methods, gbdiffs, csizes = [], [], [], []
    for c in range(k):
        members = relidx[labels == c].tolist()
        vec, meth, gd = R.diff_denoise_rep(gW, gb, members)
        reps.append(vec); methods.append(meth); gbdiffs.append(gd); csizes.append(len(members))
    reps = torch.stack(reps)                                   # (k, in_dim)

    # also keep the plain best-|gb| single-ratio rep per cluster as a fallback pool
    single = []
    for c in range(k):
        members = relidx[labels == c].tolist()
        i = members[int(torch.argmax(gb[members].abs()))]
        single.append(gW[i] / gb[i])
    single = torch.stack(single)

    if is_img:  # ---- MLP: reps ARE images ----
        cand = reps.reshape(k, 3, 64, 64)
        cand_s = single.reshape(k, 3, 64, 64)
        # normalise to [0,1] (net.0 input is the [0,1] image; use per-image minmax
        # for tanh/sigmoid-scaled nets, clamp for relu)
        imgs = R.per_image_minmax(cand)
        imgs_s = R.per_image_minmax(cand_s)
        # pick per-slot the cleaner of {diff, single}
        out = []
        for a in range(k):
            qd = R.quality_score(imgs[a]); qs = R.quality_score(imgs_s[a])
            out.append(imgs[a] if qd >= qs else imgs_s[a])
        imgs = torch.stack(out)
        feat_for_cert = None
    else:  # ---- CNN: reps are FEATURES -> invert conv ----
        feats = clamp_activation(reps.reshape(k, *fshape), act)
        feats_s = clamp_activation(single.reshape(k, *fshape), act)
        model = M.build_model(name, {"family": "cnn", "activation": act,
                                     "feature_shape": fshape}, state)
        strided = model.stride > 1
        base_res = 64 if not strided else fshape[1]
        log(f"{name}: conv stride={model.stride} pad={model.padding} strided={strided} "
            f"base_res={base_res} inverting {k} feats x{STEPS} steps")
        imgs = invert_conv(model, feats, steps=STEPS,
                           lr=0.03 if strided else 0.05,
                           tv=5e-3 if strided else 1e-4,
                           device=DEVICE, base_res=base_res)
        feat_for_cert = feats

    # ---- quality + collision, rank ----
    q = torch.tensor([R.quality_score(im) for im in imgs])
    ok = torch.tensor([R.collision_ok(im) for im in imgs])
    order = torch.argsort(torch.where(ok, q, q - 10.0), descending=True)
    n_clean = int(ok.sum())

    # ---- assemble exactly 128 (fill with best remaining, never noise) ----
    N = U.N_IMAGES
    sel = order[:N].tolist()
    if len(sel) < N:
        sel = (sel * (N // len(sel) + 1))[:N]
    final = imgs[sel].clamp(0, 1)
    fq = q[sel]

    # ---- certify: subspace residual onto first-layer gW top-128 ----
    Vspace = None; resid = None
    if is_img:
        V, _ = topk_right_subspace(gW, 128)
        resid = subspace_residual(final, V).numpy()
    else:
        # feature-space residual: pass images through conv, compare to fc1 rowspace
        V, _ = topk_right_subspace(gW, 128)
        with torch.no_grad():
            fcur = model.conv_feature(final.to(DEVICE)).reshape(N, -1).cpu()
        resid = subspace_residual(fcur, V).numpy()
    rand_resid = float(subspace_residual(
        torch.rand(128, *( (3,64,64) if is_img else fshape)),
        topk_right_subspace(gW,128)[0]).mean())

    # ---- save ----
    odir = os.path.join(WORK, "outputs_trap"); os.makedirs(odir, exist_ok=True)
    rdir = os.path.join(WORK, "reports", "trap", name); os.makedirs(rdir, exist_ok=True)
    torch.save({"images": final.float(),
                "quality": fq.float(),
                "subspace_resid": torch.tensor(resid),
                "n_clean": n_clean, "family": fam, "activation": act,
                "method": "ratio+diff-denoise" + ("+conv-invert" if not is_img else ""),
                "cluster_methods": methods},
               os.path.join(odir, f"rec_{name}.pt"))
    U.save_image_grid(final[:64], os.path.join(rdir, "grid_top64.png"))
    U.save_image_grid(final[torch.argsort(torch.tensor(resid))[:64]],
                      os.path.join(rdir, "grid_lowresid64.png"))
    summ = {"name": name, "family": fam, "activation": act, "n_clean": n_clean,
            "k_clusters": k, "reliable_rows": len(relidx),
            "resid_mean": round(float(np.mean(resid)), 4),
            "resid_p10": round(float(np.percentile(resid, 10)), 4),
            "rand_resid": round(rand_resid, 4),
            "quality_mean": round(float(fq.mean()), 4),
            "diff_used": int(sum(1 for m in methods if m == "diff"))}
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    log(f"{name}: DONE n_clean={n_clean} resid_mean={summ['resid_mean']} "
        f"(rand {summ['rand_resid']}) diff_used={summ['diff_used']} -> rec_{name}.pt")
    return summ


def main():
    names = sys.argv[1:] or ["model10"]
    alls = {}
    for n in names:
        try:
            alls[n] = reconstruct(n)
        except Exception as e:
            import traceback
            log(f"{n}: ERROR {e}"); traceback.print_exc()
    print("SUMMARY:", json.dumps(alls, indent=2))
    print("__DONE__")


if __name__ == "__main__":
    main()

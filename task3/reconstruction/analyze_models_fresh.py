"""Fresh from-scratch analysis of all 12 models' gradients, to find recoverable
signal we have not yet exploited (esp. ViT 9/11, CNN model6, MLP model8).

For each model we report:
  * architecture: family, activation, all grad params (shape, norm, %near-zero),
    dead-bias fraction (trap-weight / single-sample signature).
  * first biased linear/conv layer: SVD of its weight-gradient. Since
    gradW = sum_{n=1..128} c_n x_n^T (128 samples), gradW has rank <= 128 and its
    ROW SPACE spans the 128 true inputs. We report the singular-value decay
    (sharp drop after 128 => clean, separable batch structure) and, for layers
    whose input is image- or patch-shaped, SAVE the top singular vectors AS
    IMAGES -- if they look like faces/face-patches, the subspace is recoverable.
  * label multiset from the classifier-head bias gradient.

Login-node safe: loads .pt + bounded linear algebra (SVD of <=1024x12288), no
optimization loop.
"""
import os
import sys
import json
import torch

import utils as U
import models as M
try:
    import vit as VIT
except Exception:
    VIT = None

EPS = 1e-8


def grad_param_table(g):
    rows = []
    for k, v in g.items():
        if v is None:
            continue
        v = v.float()
        flat = v.reshape(-1)
        near0 = (flat.abs() < 1e-7).float().mean().item()
        rows.append({"name": k, "shape": tuple(v.shape),
                     "norm": round(v.norm().item(), 5),
                     "frac_near0": round(near0, 4)})
    return rows


def first_biased_linear(state, g):
    """earliest 2-D .weight whose matching .bias has a gradient."""
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def dead_bias_fraction(gb, eps=EPS):
    gb = gb.float()
    return round((gb.abs() < eps).float().mean().item(), 4)


def weight_signature(w):
    """trap-weight tell: balance of +/- weights and bimodality of |w|."""
    w = w.float().reshape(-1)
    frac_neg = (w < 0).float().mean().item()
    return {"mean": round(w.mean().item(), 5), "std": round(w.std().item(), 5),
            "frac_neg": round(frac_neg, 4),
            "absmean": round(w.abs().mean().item(), 5)}


def svd_report(G, k_img=64, img_shape=None, grid_path=None, name=""):
    """SVD of weight-gradient G (out,in). Report singular-value decay; if img_shape
    given, save top-k right singular vectors reshaped as images."""
    G = G.float()
    out, inp = G.shape
    # economy SVD; for tall/wide this is fine up to ~1024x12288
    U_, S, Vh = torch.linalg.svd(G, full_matrices=False)
    S = S.cpu()
    s0 = S[0].item()
    def sv(i): return round(S[i].item(), 4) if i < S.numel() else None
    def svr(i): return round((S[i] / (s0 + 1e-12)).item(), 4) if i < S.numel() else None
    eff_rank = int((S > 0.01 * s0).sum().item())
    rep = {"shape": (out, inp), "s0": round(s0, 4),
           "sv_ratio@1": svr(1), "sv_ratio@64": svr(64), "sv_ratio@127": svr(127),
           "sv_ratio@128": svr(128), "sv_ratio@130": svr(130), "sv_ratio@200": svr(200),
           "eff_rank_1pct": eff_rank}
    if img_shape is not None and grid_path is not None:
        C, H, W = img_shape
        vecs = Vh[:k_img]                       # (k, in)
        imgs = vecs.reshape(-1, C, H, W)
        imgs = U.per_image_minmax(imgs)         # each singular vector to [0,1]
        if (H, W) != (64, 64):
            import torch.nn.functional as F
            imgs = F.interpolate(imgs, size=(64, 64), mode="nearest")
        U.save_image_grid(imgs, grid_path, ncol=8)
        rep["singular_vector_grid"] = grid_path
    return rep


def label_multiset(state, g, N=128):
    last = None
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2 and (pname[:-7] + ".bias") in g:
            last = pname[:-7]
    if last is None or g.get(last + ".bias") is None:
        return {}
    gb = g[last + ".bias"].float()
    counts = (-N * gb).clamp(min=0).round().to(torch.int64)
    return {"head": last, "n_classes": gb.numel(),
            "est_total": int(counts.sum()), "present_classes": int((counts > 0).sum())}


def analyze(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    fam = grad.get("family")
    act = grad.get("activation")
    fs = grad.get("feature_shape")
    out_dir = U.work_path("reports", "fresh", name)
    U.ensure_dir(out_dir)

    info = {"name": name, "family": fam, "activation": act, "feature_shape": fs}
    info["params"] = grad_param_table(g)
    info["labels"] = label_multiset(state, g)

    # ---- first biased linear layer analysis ----
    base = first_biased_linear(state, g)
    if base is not None:
        gW = g[base + ".weight"].float()
        gB = g[base + ".bias"].float()
        info["first_linear"] = {"layer": base,
                                "dead_bias_frac": dead_bias_fraction(gB),
                                "weight_grad_sig": weight_signature(gW)}
        # is the input image-shaped (MLP net.0: in==12288)?
        img_shape = None
        if gW.shape[1] == U.IMG_NUMEL:
            img_shape = (U.IMG_C, U.IMG_H, U.IMG_W)
        gp = os.path.join(out_dir, f"{name}_firstlinear_singvecs.png")
        info["first_linear"]["svd"] = svd_report(
            gW, img_shape=img_shape,
            grid_path=gp if img_shape else None, name=name)
        if img_shape is None:
            # still report SVD decay without image grid
            info["first_linear"]["svd"] = svd_report(gW, name=name)

    # ---- ViT patch-embedding: per-patch linear leak ----
    if fam == "vit":
        pw = g.get("patch_embed.proj.weight")
        pb = g.get("patch_embed.proj.bias")
        if pw is not None:
            dim = pw.shape[0]
            patch = pw.shape[-1]
            Gp = pw.float().reshape(dim, -1)      # (dim, 3*patch*patch)
            gp = os.path.join(out_dir, f"{name}_patchembed_singvecs.png")
            rep = svd_report(Gp, k_img=min(64, dim),
                             img_shape=(3, patch, patch), grid_path=gp, name=name)
            info["patch_embed"] = {"dim": dim, "patch": patch,
                                   "dead_bias_frac": dead_bias_fraction(pb) if pb is not None else None,
                                   "svd": rep}
        # pos_embed / cls_token gradient magnitudes (localization signal?)
        for key in ("pos_embed", "cls_token"):
            t = g.get(key)
            if t is not None:
                info.setdefault("vit_extra", {})[key] = {
                    "shape": tuple(t.shape), "norm": round(t.float().norm().item(), 5)}

    return info


def main():
    names = sys.argv[1:] or U.MODELS
    allinfo = {}
    for name in names:
        try:
            allinfo[name] = analyze(name)
            fi = allinfo[name].get("first_linear", {})
            svd = fi.get("svd", {})
            print(f"{name:8} {allinfo[name]['family']:4} act={allinfo[name]['activation']}: "
                  f"first={fi.get('layer')} dead_bias={fi.get('dead_bias_frac')} "
                  f"eff_rank={svd.get('eff_rank_1pct')} "
                  f"sv@128/@130={svd.get('sv_ratio@128')}/{svd.get('sv_ratio@130')} "
                  f"labels~{allinfo[name].get('labels',{}).get('est_total')}", flush=True)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}"); traceback.print_exc()
            allinfo[name] = {"name": name, "error": str(e)}
    U.save_json(allinfo, U.work_path("reports", "fresh", "analysis.json"))
    print("\nwrote", U.work_path("reports", "fresh", "analysis.json"))


if __name__ == "__main__":
    main()

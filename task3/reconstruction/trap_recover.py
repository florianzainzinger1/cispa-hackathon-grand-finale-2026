"""trap_recover.py  (Agent T / lever L2) — exact/verbatim recovery.

Two analytic recovery paths (Fowl 2021 + Boenisch 2021), auto-routed per model
from reports/trap/diag.json:

  RATIO  (Boenisch trap weights): x_k = gW[k] / gb[k].  For a neuron activated by
         a SINGLE sample the unknown scalar dL/dz cancels -> the weight-grad row
         IS that input, verbatim. We take rows DIRECTLY (no cluster/robust-mean
         averaging, which blends faces -> the blur bug in AGENT_SETUP Part II).

  DIFF   (Robbing-the-Fed imprint): sort neurons by state-dict bias (threshold
         order), consecutive-difference  x = (gW[l]-gW[l+1])/(gb[l]-gb[l+1]).
         Each non-colliding bin isolates the one sample whose measurement falls
         between the two thresholds -> verbatim. (Reverse in-place differencing,
         exactly as lhfowl/robbing_the_fed analytic_attack.py.)

Collision/occupancy handling (from the papers):
  * occupancy: keep only |gb| (or |den|) > tol*max|gb|  (empty bins -> divide by 0).
  * multi-sample bins are blends; we cannot split them, so we RANK candidates by a
    confidence score (reliability |gb|, DP-robust |mean gW|, and image faceness),
    dedup, and keep the best up to 128. Empirically single-sample rows are sharp
    faces; blends are low-contrast ghosts -> the faceness score down-weights them.

MLP: rows are (3,64,64) images directly.  CNN: rows are fc1-input features -> we
invert the first conv (attack_conv_invert.invert_conv) to RGB (GPU).  ViT: no clean
linear image-leak -> skip (bank/other levers own 9,11).

Outputs per model: outputs_trap/trap_{model}.pt = {"images":(<=128,3,64,64) [0,1],
"source", "n_ratio","n_diff","meta"}  + candidates_trap/{model}/*.pt and grids.
Compares DIFF vs RATIO vs the existing analytic-ratio bank in the per-model meta.

Run via sbatch job_trap.sh recover  (GPU for CNN conv-inversion).
    python -u trap_recover.py [model10 model6 ...]
"""
import os
import sys
import json
import torch
import torch.nn.functional as F

import utils as U
import models as M
from attack_conv_invert import invert_conv

_dev = os.environ.get("FL_DEVICE", "cpu")
DEVICE = "cuda" if (_dev == "cuda" and torch.cuda.is_available()) else "cpu"
N = U.N_IMAGES  # 128


# ---------------------------------------------------------------------------
def first_biased_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def load_layer(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    base = first_biased_linear(state, g)
    fs = tuple(grad["feature_shape"]) if grad.get("feature_shape") else None
    return {
        "name": name, "grad": grad, "state": state, "g": g,
        "family": grad.get("family"), "activation": grad.get("activation"),
        "feature_shape": fs, "base": base,
        "gW": g[base + ".weight"].float().to(DEVICE),
        "gb": g[base + ".bias"].float().to(DEVICE),
        "b": state[base + ".bias"].float().to(DEVICE) if base + ".bias" in state else None,
        "in_dim": g[base + ".weight"].shape[1],
    }


# ---------------------------------------------------------------------------
# the two analytic recoveries -> rows in input space + a per-row confidence
# ---------------------------------------------------------------------------
def ratio_rows(gW, gb, tol_rel=1e-4):
    tol = tol_rel * float(gb.abs().max())
    keep = gb.abs() > tol
    x = gW[keep] / gb[keep].unsqueeze(1)
    conf = gW[keep].abs().mean(dim=1)        # DP-robust |mean gW| ranking (RtF)
    return x, conf, int(keep.sum())


def diff_rows(gW, gb, b, tol_rel=1e-3):
    if b is None:
        return gW.new_zeros((0, gW.shape[1])), gW.new_zeros((0,)), 0
    order = torch.argsort(b, descending=True)          # threshold order
    gWo, gbo = gW[order], gb[order]
    num = gWo[:-1] - gWo[1:]
    den = gbo[:-1] - gbo[1:]
    tol = tol_rel * float(gb.abs().max())
    keep = den.abs() > tol
    x = num[keep] / den[keep].unsqueeze(1)
    conf = num[keep].abs().mean(dim=1)
    return x, conf, int(keep.sum())


# ---------------------------------------------------------------------------
# rows (input space) -> RGB images  (MLP reshape / CNN conv-inversion)
# ---------------------------------------------------------------------------
def rows_to_images(rows, layer, invert_steps=300):
    """rows: (n, in_dim) on DEVICE. Returns dict of variants: images in [0,1]."""
    n = rows.shape[0]
    if n == 0:
        z = torch.zeros((0, 3, 64, 64))
        return {"clamp": z, "minmax": z}
    if layer["in_dim"] == U.IMG_NUMEL:                 # MLP: direct image
        img = rows.reshape(n, U.IMG_C, U.IMG_H, U.IMG_W)
        return {"clamp": img.clamp(0, 1).cpu(),
                "minmax": U.per_image_minmax(img).cpu()}
    # CNN: rows are fc1-input features -> invert first conv
    C, H, W = layer["feature_shape"]
    if C * H * W != layer["in_dim"]:
        z = torch.zeros((0, 3, 64, 64))
        return {"clamp": z, "minmax": z}
    feat = rows.reshape(n, C, H, W)
    act = layer["activation"]
    if act == "relu":
        feat = feat.clamp_min(0.0)
    elif act == "sigmoid":
        feat = feat.clamp(0.0, 1.0)
    elif act == "tanh":
        feat = feat.clamp(-1.0, 1.0)
    meta = {"family": "cnn", "activation": act, "feature_shape": layer["feature_shape"]}
    model = M.build_model(layer["name"], meta, layer["state"])
    strided = model.stride > 1
    base_res = 64 if not strided else H
    rec = invert_conv(model, feat, steps=invert_steps,
                      lr=0.03 if strided else 0.05,
                      tv=5e-3 if strided else 1e-4,
                      device=DEVICE, base_res=base_res)
    return {"clamp": rec.cpu(), "minmax": U.per_image_minmax(rec).cpu()}


def faceness(imgs):
    """Higher = more face-like / less blend-ghost. in-range * contrast - TV penalty."""
    if imgs.shape[0] == 0:
        return torch.zeros((0,))
    st = U.image_stats(imgs)
    score = st["rng_frac"] * torch.clamp(st["std"] / 0.15, max=1.0)
    score = score - 2.0 * torch.clamp(st["tv"] - 0.15, min=0.0)   # penalize grain/ghost
    score = torch.where(st["finite"], score, torch.full_like(score, -1.0))
    return score


def pick_variant(variants):
    """Choose clamp vs per-image-minmax by median faceness (verbatim scale can be
    off for tanh/sigmoid MLPs -> minmax rescues those)."""
    best, best_s, best_score = None, -1e9, None
    for k, imgs in variants.items():
        if imgs.shape[0] == 0:
            continue
        s = faceness(imgs)
        med = float(s.median()) if s.numel() else -1e9
        if med > best_s:
            best, best_s, best_score = k, med, s
    return best, best_score


# ---------------------------------------------------------------------------
def run_model(name, diag, invert_steps=300, dedup=True):
    layer = load_layer(name)
    if layer["family"] == "vit":
        print(f"{name}: vit -> trap differencing N/A (bank/other levers own it)", flush=True)
        return {"name": name, "family": "vit", "skipped": True}
    if layer["in_dim"] != U.IMG_NUMEL and layer["feature_shape"] is None:
        print(f"{name}: no image/feature layer; skip", flush=True)
        return {"name": name, "skipped": True}

    dinfo = diag.get(name, {})
    imprint = dinfo.get("imprint_like", False)
    trap = dinfo.get("trap_sparse", False)

    # build both candidate pools (rows in input space)
    r_rows, r_conf, r_keep = ratio_rows(layer["gW"], layer["gb"])
    d_rows, d_conf, d_keep = diff_rows(layer["gW"], layer["gb"], layer["b"])

    # cap raw pools by confidence before the (expensive) CNN inversion
    def top(rows, conf, k=384):
        if rows.shape[0] <= k:
            return rows, conf
        idx = torch.argsort(conf, descending=True)[:k]
        return rows[idx], conf[idx]
    r_rows, r_conf = top(r_rows, r_conf)
    d_rows, d_conf = top(d_rows, d_conf)

    r_var = rows_to_images(r_rows, layer, invert_steps)
    d_var = rows_to_images(d_rows, layer, invert_steps)
    r_k, r_score = pick_variant(r_var)
    d_k, d_score = pick_variant(d_var)
    r_imgs = r_var[r_k] if r_k else torch.zeros((0, 3, 64, 64))
    d_imgs = d_var[d_k] if d_k else torch.zeros((0, 3, 64, 64))

    out_dir = U.work_path("candidates_trap", name)
    U.ensure_dir(out_dir)
    if r_imgs.shape[0]:
        U.save_image_grid(r_imgs[:64], os.path.join(out_dir, "grid_ratio.png"))
    if d_imgs.shape[0]:
        U.save_image_grid(d_imgs[:64], os.path.join(out_dir, "grid_diff.png"))
    torch.save({"ratio": r_imgs, "diff": d_imgs, "ratio_variant": r_k,
                "diff_variant": d_k}, os.path.join(out_dir, "pools.pt"))

    # route: primary path first, then the other as filler; dedup+select 128
    if imprint:
        order = [("diff", d_imgs, d_score), ("ratio", r_imgs, r_score)]
    else:  # trap-sparse or honest -> ratio primary (Boenisch), diff as extra
        order = [("ratio", r_imgs, r_score), ("diff", d_imgs, d_score)]

    pool_imgs, pool_score, prov = [], [], []
    for tag, imgs, sc in order:
        if imgs.shape[0]:
            pool_imgs.append(imgs); pool_score.append(sc)
            prov += [tag] * imgs.shape[0]
    if not pool_imgs:
        print(f"{name}: no candidates recovered", flush=True)
        return {"name": name, "n_recovered": 0}
    pool = torch.cat(pool_imgs, 0)
    scores = torch.cat(pool_score, 0)

    # quality gate + greedy dedup to <=128 (reuse shared postprocess.select_128)
    order2 = torch.argsort(scores, descending=True)
    pool, scores = pool[order2], scores[order2]
    prov = [prov[i] for i in order2.tolist()]
    if dedup:
        try:
            import postprocess as PP
            sel, chosen = PP.select_128(pool, n=N, denoise=False)
            chosen = list(chosen)
        except Exception as e:
            print(f"{name}: select_128 fallback ({e})", flush=True)
            sel, chosen = pool[:N], list(range(min(N, pool.shape[0])))
    else:
        sel, chosen = pool[:N], list(range(min(N, pool.shape[0])))
    sel = sel.float().clamp(0, 1)
    n_from_diff = sum(1 for c in chosen if c < len(prov) and prov[c] == "diff")
    n_from_ratio = len(chosen) - n_from_diff

    U.ensure_dir(U.work_path("outputs_trap"))
    out = U.work_path("outputs_trap", f"trap_{name}.pt")
    torch.save({"images": sel, "source": "trap_recover",
                "n_selected": int(sel.shape[0]),
                "n_from_diff": int(n_from_diff), "n_from_ratio": int(n_from_ratio),
                "ratio_pool": int(r_imgs.shape[0]), "diff_pool": int(d_imgs.shape[0]),
                "ratio_keep_rows": r_keep, "diff_keep_bins": d_keep,
                "route": "diff-primary" if imprint else "ratio-primary",
                "ratio_variant": r_k, "diff_variant": d_k}, out)
    U.save_image_grid(sel[:min(128, sel.shape[0])],
                      os.path.join(out_dir, "grid_selected.png"))
    meta = {
        "name": name, "family": layer["family"], "activation": layer["activation"],
        "target_layer": layer["base"], "in_dim": int(layer["in_dim"]),
        "imprint_like": imprint, "trap_sparse": trap,
        "ratio_keep_rows": r_keep, "diff_keep_bins": d_keep,
        "ratio_pool_imgs": int(r_imgs.shape[0]), "diff_pool_imgs": int(d_imgs.shape[0]),
        "ratio_variant": r_k, "diff_variant": d_k,
        "ratio_med_faceness": round(float(r_score.median()), 4) if r_score is not None and r_score.numel() else None,
        "diff_med_faceness": round(float(d_score.median()), 4) if d_score is not None and d_score.numel() else None,
        "n_selected": int(sel.shape[0]),
        "n_from_ratio": int(n_from_ratio), "n_from_diff": int(n_from_diff),
        "route": "diff-primary" if imprint else "ratio-primary",
    }
    U.save_json(meta, os.path.join(out_dir, "meta.json"))
    print(f"{name:8s} route={meta['route']:13s} ratio_pool={r_imgs.shape[0]:3d}"
          f"(med {meta['ratio_med_faceness']}) diff_pool={d_imgs.shape[0]:3d}"
          f"(med {meta['diff_med_faceness']}) -> sel {sel.shape[0]} "
          f"[{n_from_ratio}R/{n_from_diff}D] -> {out}", flush=True)
    return meta


def main():
    diag_path = U.work_path("reports", "trap", "diag.json")
    diag = json.load(open(diag_path)) if os.path.exists(diag_path) else {}
    if not diag:
        print("WARN: no diag.json; routing everything ratio-primary", flush=True)
    steps = int(os.environ.get("FL_ITERS", "300"))
    names = sys.argv[1:] or [m for m in U.MODELS]
    summary = {}
    for name in names:
        try:
            summary[name] = run_model(name, diag, invert_steps=steps)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}"); traceback.print_exc()
            summary[name] = {"name": name, "error": str(e)}
    U.save_json(summary, U.work_path("reports", "trap", "recover_summary.json"))
    print("\nwrote", U.work_path("reports", "trap", "recover_summary.json"))


if __name__ == "__main__":
    main()

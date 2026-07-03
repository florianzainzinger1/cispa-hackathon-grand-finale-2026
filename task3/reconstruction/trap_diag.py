"""trap_diag.py  (Agent T / lever L2) — per-model first-layer TRAP diagnostics.

Goal: for every model, decide whether its first biased linear layer looks like
  (a) a Robbing-the-Fed IMPRINT stack (Fowl 2021): all weight rows share one
      measurement direction h, biases are ordered thresholds -> consecutive-bin
      DIFFERENCING recovers verbatim images;
  (b) a Boenisch TRAP-weight layer (2112.02918): heterogeneous rows, each ~50/50
      signed with positive half = -s*negative half (s<1), sparse single-sample
      activation + many dead ReLU neurons -> per-row RATIO gW[k]/gb[k] is already
      near-verbatim (NO differencing);
  (c) an HONEST layer: dense mixed activation -> ratio rows are blends.

We are the malicious server AFTER THE FACT: handed trained weights + one averaged
first-layer gradient (batch B=128) per model; we must DETECT structure, not design it.

Signatures (first biased 2-D linear: net.0 for MLP, fc1 for CNN; first 2-D linear
for ViT = patch/qkv, differencing N/A there):
  * dead-neuron fraction     : gb (bias-gradient) exactly 0  (ReLU never fired)
  * IMPRINT test             : mean |cos| between WEIGHT rows, sv0-dominance of W
                               (rows colinear => shared measurement h)
  * BOENISCH sign/scale test : per-row pos-fraction (~0.5) and mean|w+|/mean|w-| (~s<1)
  * gW singular spectrum      : sv ratios @ {1,64,127,128,130,200}, eff-rank, E@128
  * nested-bin ordering test  : sort neurons by state-dict bias desc; is |gb| monotone
                               along that order? (imprint => yes)
Plus eyeball grids: top ratio rows by |gb|, and consecutive-bias-order DIFFERENCES
(the trap_recover preview) — MLP in image space, CNN in fc1-feature RGB view.

Runs on GPU if FL_DEVICE=cuda (fast SVD; avoids the contended login node) else CPU.
Outputs: reports/trap/diag.json  and  reports/trap/<model>/*.png
Run (via sbatch job_trap.sh, or):  python -u trap_diag.py [model10 model6 ...]
"""
import os
import sys
import torch
import torch.nn.functional as F

import utils as U

EPS = 1e-12
_dev = os.environ.get("FL_DEVICE", "cpu")
DEVICE = "cuda" if (_dev == "cuda" and torch.cuda.is_available()) else "cpu"


def first_biased_linear(state, g):
    """Earliest biased 2-D linear whose bias-gradient is present."""
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def mean_abs_offdiag_cosine(M, max_rows=1024):
    """Mean |cosine| between rows of M (out,in), off-diagonal. Sampled if large."""
    M = M.float()
    n = M.shape[0]
    if n > max_rows:
        idx = torch.linspace(0, n - 1, max_rows, device=M.device).long()
        M = M[idx]
    norm = M.norm(dim=1, keepdim=True)
    good = (norm.squeeze(1) > 0)
    if int(good.sum()) < 2:
        return {"mean_abscos": 0.0, "mean_cos": 0.0, "n_used": int(good.sum())}
    Mn = M[good] / norm[good]
    G = Mn @ Mn.t()
    m = G.shape[0]
    off = ~torch.eye(m, dtype=torch.bool, device=G.device)
    vals = G[off]
    return {"mean_abscos": float(vals.abs().mean()),
            "mean_cos": float(vals.mean()), "n_used": int(m)}


def sign_scale_stats(W):
    """Boenisch trap fingerprint from the weight matrix alone.
    Trap rows: ~50% positive components; positive half = -s * negative half, so
    mean|w+|/mean|w-| ~ s (a tight constant < 1). Honest rows: pos_frac ~ 0.5 too
    but the scale ratio ~ 1 with LARGE spread. The discriminator is the IQR: trap =
    tight (small IQR) scale ratio clearly != 1; honest = wide IQR around 1."""
    W = W.float()
    pos = (W > 0)
    pos_frac = pos.float().mean(dim=1)
    sum_pos = W.clamp(min=0).sum(dim=1)
    sum_neg = W.clamp(max=0).abs().sum(dim=1)
    npos = pos.sum(dim=1).clamp(min=1)
    nneg = (W < 0).sum(dim=1).clamp(min=1)
    scale = (sum_pos / npos) / (sum_neg / nneg + 1e-30)

    def q(t, p):
        return float(torch.quantile(t, p))
    return {
        "pos_frac_median": round(float(pos_frac.median()), 4),
        "pos_frac_iqr": round(q(pos_frac, 0.75) - q(pos_frac, 0.25), 4),
        "scale_ratio_median": round(float(scale.median()), 4),
        "scale_ratio_iqr": round(q(scale, 0.75) - q(scale, 0.25), 4),
    }


def sv_profile(gW):
    """Economy SVD singular-value profile of gW (out,in)."""
    S = torch.linalg.svdvals(gW.float()).cpu()
    s0 = float(S[0]) + 1e-30
    energy = float((S ** 2).sum()) + 1e-30

    def r(i):
        return round(float(S[i]) / s0, 5) if i < S.numel() else None
    return {
        "sv@1": r(1), "sv@64": r(64), "sv@127": r(127), "sv@128": r(128),
        "sv@130": r(130), "sv@200": r(200),
        "eff_rank_1pct": int((S > 0.01 * S[0]).sum()),
        "energy_top128": round(float((S[:128] ** 2).sum()) / energy, 5),
        "n_sv": int(S.numel()),
    }


def spearman(a, b):
    a = a.float(); b = b.float()
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / ((ra.norm() * rb.norm()) + 1e-30))


def to_view(rows_2d, in_dim, feature_shape):
    """Ratio/difference rows -> (n,3,64,64) [0,1] view. MLP: reshape; CNN: fc1
    feature -> first-3-channel RGB view (upsampled). ViT/other: None."""
    n = rows_2d.shape[0]
    if in_dim == U.IMG_NUMEL:
        return U.per_image_minmax(rows_2d.reshape(n, U.IMG_C, U.IMG_H, U.IMG_W))
    if not feature_shape:
        return None
    C, H, W = feature_shape
    if C * H * W != in_dim:
        return None
    feat = rows_2d.reshape(n, C, H, W)
    v = feat[:, :3] if C >= 3 else feat.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    if (H, W) != (64, 64):
        v = F.interpolate(v, size=(64, 64), mode="bilinear", align_corners=False)
    return U.per_image_minmax(v)


def run_model(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    family = grad.get("family")
    activation = grad.get("activation")
    feature_shape = tuple(grad.get("feature_shape")) if grad.get("feature_shape") else None
    base = first_biased_linear(state, g)
    out_dir = U.work_path("reports", "trap", name)
    U.ensure_dir(out_dir)
    if base is None:
        return {"name": name, "family": family, "note": "no biased linear"}

    W = state[base + ".weight"].float().to(DEVICE)
    gW = g[base + ".weight"].float().to(DEVICE)
    gb = g[base + ".bias"].float().to(DEVICE)
    b = state[base + ".bias"].float().to(DEVICE) if (base + ".bias") in state else None
    out_dim, in_dim = gW.shape
    is_image = in_dim == U.IMG_NUMEL

    absgb = gb.abs()
    max_absgb = float(absgb.max()) + 1e-30
    dead_exact = int((absgb <= EPS).sum())
    near_dead = int((absgb < 1e-6 * max_absgb).sum())
    dead_row = int((gW.norm(dim=1) <= EPS).sum())
    pos = int((gb > 0).sum()); neg = int((gb < 0).sum())
    alive = absgb[absgb > 1e-6 * max_absgb]
    gbq = ([round(float(x), 6) for x in
            torch.quantile(alive / max_absgb, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=DEVICE))]
           if alive.numel() > 0 else None)

    W_cos = mean_abs_offdiag_cosine(W)
    gW_cos = mean_abs_offdiag_cosine(gW)
    Ssv = torch.linalg.svdvals(W)
    W_sv0_dom = round(float(Ssv[0] ** 2 / (Ssv ** 2).sum()), 5)
    W_effrank = int((Ssv > 0.01 * Ssv[0]).sum())
    sgn = sign_scale_stats(W)
    gW_spec = sv_profile(gW)

    order_info = {}
    if b is not None:
        order = torch.argsort(b, descending=True)   # high bias = low threshold = most active
        absgb_ord = absgb[order]
        d = absgb_ord[1:] - absgb_ord[:-1]
        order_info = {
            "spearman_b_vs_absgb": round(spearman(b, absgb), 4),
            "absgb_monotone_dec_frac": round(float((d <= 0).float().mean()), 4),
            "b_neg_frac": round(float((b < 0).float().mean()), 4),
            "b_min": round(float(b.min()), 4), "b_max": round(float(b.max()), 4),
            "b_mean": round(float(b.mean()), 4), "b_std": round(float(b.std()), 4),
        }

    # eyeball grids
    denom = torch.where(absgb > 1e-8, gb, torch.full_like(gb, float("nan")))
    ratio = gW / denom.unsqueeze(1)
    top = torch.argsort(absgb, descending=True)[:64]
    v = to_view(torch.nan_to_num(ratio[top]), in_dim, feature_shape)
    if v is not None:
        U.save_image_grid(v, os.path.join(out_dir, "grid_ratio_by_absgb.png"))
    diff_kept = 0
    if b is not None:
        order = torch.argsort(b, descending=True)
        gWo, gbo = gW[order], gb[order]
        num = gWo[:-1] - gWo[1:]
        den = gbo[:-1] - gbo[1:]
        good = den.abs() > (1e-3 * max_absgb)
        diff = torch.zeros_like(num)
        diff[good] = num[good] / den[good].unsqueeze(1)
        diff_kept = int(good.sum())
        gidx = torch.where(good)[0]
        if gidx.numel() > 0:
            sel = gidx[torch.argsort(den.abs()[gidx], descending=True)[:64]]
            dv = to_view(diff[sel], in_dim, feature_shape)
            if dv is not None:
                U.save_image_grid(dv, os.path.join(out_dir, "grid_diff_bias_order.png"))

    dead_frac = round(dead_exact / out_dim, 4)
    imprint_like = (W_cos["mean_abscos"] > 0.55) or (W_sv0_dom > 0.45)
    strong_nest = order_info.get("absgb_monotone_dec_frac", 0) > 0.9 and imprint_like
    boenisch_sig = (0.4 <= sgn["pos_frac_median"] <= 0.6) and sgn["scale_ratio_iqr"] < 0.25 \
        and abs(sgn["scale_ratio_median"] - 1.0) > 0.1
    trap_sparse = (dead_frac > 0.12 and activation == "relu") or boenisch_sig
    if imprint_like and strong_nest:
        klass = "IMPRINT(RtF) -> differencing (bias order)"
    elif imprint_like:
        klass = "imprint-like(W shared dir) -> try differencing"
    elif trap_sparse:
        klass = "TRAP-sparse(Boenisch) -> ratio near-verbatim + local differencing"
    else:
        klass = "honest -> ratio(blend); differencing only on shared-sample pairs"

    res = {
        "name": name, "family": family, "activation": activation,
        "target_layer": base, "in_dim": int(in_dim), "out_dim": int(out_dim),
        "is_image_layer": bool(is_image), "feature_shape": feature_shape,
        "dead_exact": dead_exact, "dead_frac": dead_frac,
        "near_dead": near_dead, "dead_gW_rows": dead_row,
        "gb_pos": pos, "gb_neg": neg, "max_absgb": round(max_absgb, 8),
        "absgb_alive_quantiles(/max)": gbq,
        "W_row_mean_abscos": round(W_cos["mean_abscos"], 4),
        "W_row_mean_cos": round(W_cos["mean_cos"], 4),
        "W_sv0_dominance": W_sv0_dom, "W_eff_rank_1pct": W_effrank,
        "sign_scale": sgn,
        "gW_row_mean_abscos": round(gW_cos["mean_abscos"], 4),
        "gW_spectrum": gW_spec, "bias_order": order_info,
        "diff_valid_bins": diff_kept, "classification": klass,
        "imprint_like": bool(imprint_like), "trap_sparse": bool(trap_sparse),
        "boenisch_sig": bool(boenisch_sig),
    }
    print(f"{name:8s} {family:4s} {activation:7s} tgt={base:12s} in={in_dim:6d} "
          f"dead={dead_exact:4d}({dead_frac:.2f}) Wcos={res['W_row_mean_abscos']:.3f} "
          f"Wsv0={W_sv0_dom:.3f} sScale={sgn['scale_ratio_median']}"
          f"(iqr{sgn['scale_ratio_iqr']}) nest={order_info.get('absgb_monotone_dec_frac','-')} "
          f"diffbins={diff_kept:4d} :: {klass}", flush=True)
    return res


def main():
    print(f"device={DEVICE}", flush=True)
    names = sys.argv[1:] or U.MODELS
    out = {}
    for name in names:
        try:
            out[name] = run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}"); traceback.print_exc()
            out[name] = {"name": name, "error": str(e)}
    path = U.work_path("reports", "trap", "diag.json")
    U.save_json(out, path)
    print("\nwrote", path)
    print("\n=== TRAP FLAGS ===")
    for n, r in out.items():
        if "classification" in r:
            print(f"  {n:8s} {r['classification']}")


if __name__ == "__main__":
    main()

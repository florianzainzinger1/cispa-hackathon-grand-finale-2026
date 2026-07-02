"""Stage 1: inspect all 12 models/gradients and decide an attack path per model.

Writes reports/model_inspection.json and reports/model_inspection.md.
Pure gradient/state-dict analysis - no forward passes, login-node safe.
"""
import os
import torch

import utils as U


def analyse_linear_leakage(gW, gb, in_dim, eps=1e-8):
    """Diagnostics for a biased linear layer's analytic leakage.

    candidate[row] = gW[row] / gb[row]  is a weighted average of the layer's
    per-sample inputs. Rows dominated by one sample give clean reconstructions.
    """
    rows = gW.shape[0]
    reliable = gb.abs() > eps                     # rows with usable bias grad
    ratio = gW / gb.unsqueeze(1).clamp_min(eps).where(gb.unsqueeze(1) >= 0,
                                                       gb.unsqueeze(1).clamp_max(-eps))
    ratio = torch.where(reliable.unsqueeze(1), ratio, torch.zeros_like(ratio))
    finite = torch.isfinite(ratio).all(dim=1)
    safe = torch.nan_to_num(ratio)
    in_range = ((safe >= -0.05) & (safe <= 1.05)).float().mean(dim=1)
    std = safe.std(dim=1)
    # "image-like" heuristic: values mostly in [0,1] and non-trivial contrast
    image_like = (reliable & finite & (in_range > 0.85) & (std > 0.03))
    dead_bias = int((gb.abs() <= eps).sum())
    return {
        "rows": int(rows),
        "in_dim": int(in_dim),
        "reliable_rows": int(reliable.sum()),
        "dead_bias_rows": dead_bias,
        "image_like_rows": int(image_like.sum()),
        "maps_to_image": bool(in_dim == U.IMG_NUMEL),
        "median_inrange_frac": float(in_range.median()),
        "max_row_std": float(std.max()),
    }


def conv_identity_score(w):
    """How identity-like are the conv filters? w: (out,in,k,k)."""
    out_c, in_c, kh, kw = w.shape
    if kh < 2 or kw < 2:
        return 0.0
    ctr = w[:, :, kh // 2, kw // 2].abs()
    tot = w.abs().sum(dim=(2, 3)) + 1e-8
    center_frac = (ctr / tot)
    return float(center_frac.mean())


def inspect_one(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    family = grad.get("family")
    act = grad.get("activation")
    feat = tuple(grad.get("feature_shape"))
    bs = grad.get("batch_size")

    params = []
    linears, convs, norms = [], [], []
    for pname, w in state.items():
        gg = g.get(pname)
        entry = {
            "name": pname, "shape": tuple(w.shape),
            "grad_norm": float(gg.float().norm()) if gg is not None else None,
            "grad_nnz": int((gg != 0).sum()) if gg is not None else None,
        }
        params.append(entry)
        if pname.endswith(".weight") and w.ndim == 2:
            linears.append(pname[:-7])   # strip '.weight'
        if pname.endswith(".weight") and w.ndim == 4 and "patch_embed" not in pname:
            convs.append(pname[:-7])
        if ("norm" in pname or "bn" in pname or "running_mean" in pname) and pname.endswith(".weight"):
            norms.append(pname[:-7])

    # --- analytic leakage on biased linear layers ---
    linear_diag = {}
    for base in linears:
        wkey, bkey = base + ".weight", base + ".bias"
        if wkey in g and bkey in g and g[bkey] is not None:
            linear_diag[base] = analyse_linear_leakage(
                g[wkey].float(), g[bkey].float(), g[wkey].shape[1])

    # --- conv identity diagnostics ---
    conv_diag = {}
    for base in convs:
        wkey = base + ".weight"
        if wkey in g:
            conv_diag[base] = {
                "identity_score": conv_identity_score(g[wkey].float()),
                "shape": tuple(state[wkey].shape),
            }

    is_vit = family == "vit" or any("attn" in p["name"] for p in params)
    has_bn = any("running_mean" in k for k in g.keys())

    # --- decide attack path ---
    first_linear_maps_image = any(
        d["maps_to_image"] for d in linear_diag.values())
    strong_analytic = any(
        d["image_like_rows"] >= 8 for d in linear_diag.values())

    if is_vit:
        path = "vit_patch_optimization"
    elif family == "mlp" and first_linear_maps_image:
        path = "direct_analytic"
    elif family == "cnn":
        path = "feature_analytic"          # fc1 leaks features -> invert conv
    else:
        path = "cnn_no_bn_optimization"

    return {
        "name": name, "family": family, "activation": act,
        "feature_shape": feat, "batch_size": bs,
        "n_params": len(params),
        "is_vit": is_vit, "has_batchnorm": has_bn,
        "linear_layers": linears, "conv_layers": convs,
        "first_linear_maps_image": first_linear_maps_image,
        "strong_analytic_signal": strong_analytic,
        "attack_path": path,
        "linear_diag": linear_diag,
        "conv_diag": conv_diag,
        "params": params,
    }


def main():
    reports = {}
    for name in U.MODELS:
        reports[name] = inspect_one(name)
        r = reports[name]
        print(f"{name}: family={r['family']} act={r['activation']} "
              f"feat={r['feature_shape']} -> {r['attack_path']} "
              f"| analytic={'Y' if r['strong_analytic_signal'] else '-'}")

    U.save_json(reports, U.work_path("reports", "model_inspection.json"))

    # markdown summary
    lines = ["# Model Inspection Report\n",
             "| model | family | act | feature | path | analytic rows | notes |",
             "|---|---|---|---|---|---|---|"]
    for name in U.MODELS:
        r = reports[name]
        best_rows = 0
        note = []
        for base, d in r["linear_diag"].items():
            best_rows = max(best_rows, d["image_like_rows"])
            if d["maps_to_image"]:
                note.append(f"{base}->image({d['image_like_rows']} img-like)")
            if d["dead_bias_rows"] > 0:
                note.append(f"{base} dead_bias={d['dead_bias_rows']}")
        lines.append(
            f"| {name} | {r['family']} | {r['activation']} | "
            f"{r['feature_shape']} | {r['attack_path']} | {best_rows} | "
            f"{'; '.join(note)} |")
    md = "\n".join(lines) + "\n"
    path = U.work_path("reports", "model_inspection.md")
    U.ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(md)
    print("\n" + md)
    print("wrote", path)


if __name__ == "__main__":
    main()

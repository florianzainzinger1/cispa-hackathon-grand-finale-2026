"""r3: ablation submissions from the StyleGAN-prior candidates.

Builds three named variants into recon_work/runs/r3/ WITHOUT overwriting anything.
v1/v2/v3 and r1/r2 candidate dirs are read-only here; "previous best/base" =
the v3 submission (outputs_v3/submission_v3.pt), the strongest existing base.

Per-model sources:
  r2      -> runs/r2/<model>/selected_128.pt   (W-space+truncation StyleGAN latent)
  r1proj  -> runs/r1/<model>/selected_128.pt   (StyleGAN manifold projection)
  base    -> outputs_v3/submission_v3.pt[model] (previous best)

Variants (anything not listed defaults to base):
  1. r3_model11_reverted            : model2,9=r2 ; model3,7=r1proj ; (model11->base)
  2. r3_model2_model9_only          : model2,9=r2 ; everything else=base
  3. r3_model2_model9_model3_model7 : model2,9=r2 ; model3,7=r1proj ; model11->base
Note: (1) and (3) resolve to the SAME source map (both revert model11 to base and
keep model3/7 from r1proj) -> identical files, produced under both names as asked.
"""
import os
import torch

import utils as U
import build_submission as BS   # validate_submission

R2_DIR = "runs/r2"
R1_DIR = "runs/r1"
BASE_REL = ("outputs_v3", "submission_v3.pt")
OUT_DIR = "runs/r3"

VARIANTS = {
    "r3_model11_reverted":
        {"model2": "r2", "model9": "r2", "model3": "r1proj", "model7": "r1proj"},
    "r3_model2_model9_only":
        {"model2": "r2", "model9": "r2"},
    "r3_model2_model9_model3_model7":
        {"model2": "r2", "model9": "r2", "model3": "r1proj", "model7": "r1proj"},
    # Leaderboard-informed: latent-matched StyleGAN faces (model2/9) HURT SSIM
    # (-0.0093), but faithful manifold projection (model3/7) HELPED (+0.0014).
    # This variant keeps base v3 everywhere and only swaps in the projections.
    "r3_projection_only":
        {"model3": "r1proj", "model7": "r1proj"},
}


def load_sel(subdir, name):
    p = U.work_path(subdir, name, "selected_128.pt")
    if not os.path.exists(p):
        return None
    t = torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
    if tuple(t.shape) == (128, 3, 64, 64) and torch.isfinite(t).all():
        return t.contiguous()
    return None


def load_base():
    p = U.work_path(*BASE_REL)
    return torch.load(p, weights_only=False), p


def build(overrides, base, base_p):
    sub, prov = {}, {}
    for name in U.MODELS:
        src = overrides.get(name, "base")
        imgs = None
        if src == "r2":
            imgs, tag = load_sel(R2_DIR, name), "r2_stylegan"
        elif src == "r1proj":
            imgs, tag = load_sel(R1_DIR, name), "r1_projection"
        if imgs is None:                       # base fallback (also for src==base)
            imgs = base[name].float().clamp(0, 1).contiguous()
            tag = f"base:{os.path.basename(base_p)}"
        sub[name] = imgs.to(torch.float32)
        prov[name] = tag
    return sub, prov


def check(sub):
    """Explicit shape/dtype/range/NaN/uniqueness per model."""
    rows, all_ok = {}, True
    for name in U.MODELS:
        t = sub[name]
        finite = bool(torch.isfinite(t).all())
        mn, mx = float(t.min()), float(t.max())
        rng_ok = mn >= 0.0 and mx <= 1.0
        uniq = int(torch.unique(t.reshape(t.shape[0], -1), dim=0).shape[0])
        shp_ok = tuple(t.shape) == (128, 3, 64, 64)
        dt_ok = t.dtype == torch.float32
        good = finite and rng_ok and uniq == 128 and shp_ok and dt_ok
        all_ok = all_ok and good
        rows[name] = {"shape": tuple(t.shape), "dtype": str(t.dtype),
                      "min": round(mn, 4), "max": round(mx, 4),
                      "finite": finite, "unique": uniq, "ok": good}
    return all_ok, rows


def main():
    import sys
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None   # build only these
    base, base_p = load_base()
    print(f"[r3] base/previous-best = {base_p}")
    U.ensure_dir(U.work_path(OUT_DIR, "grids"))
    summary = {}
    for vname, overrides in VARIANTS.items():
        if only is not None and vname not in only:
            continue
        sub, prov = build(overrides, base, base_p)
        val_errs = BS.validate_submission(sub)
        ok, rows = check(sub)
        out = U.work_path(OUT_DIR, f"{vname}.pt")
        torch.save(sub, out)
        # preview montage: 8 faces/model, models top->bottom
        montage = torch.cat([sub[n][:8] for n in U.MODELS], 0)
        grid = U.work_path(OUT_DIR, "grids", f"{vname}_montage.png")
        U.save_image_grid(montage, grid, ncol=8)
        summary[vname] = {"path": out, "grid": grid, "provenance": prov,
                          "validate_errors": val_errs, "checks_pass": ok,
                          "per_model": rows}
        status = "VALID" if (ok and not val_errs) else f"INVALID {val_errs}"
        print(f"\n=== {vname}: {status} ===")
        print(f"  file: {out}")
        for n in U.MODELS:
            r = rows[n]
            print(f"  {n:8s} src={prov[n]:26s} shape={r['shape']} "
                  f"range=[{r['min']},{r['max']}] uniq={r['unique']}/128 "
                  f"finite={r['finite']} ok={r['ok']}")
    sname = "r3_summary.json" if only is None else \
        "r3_summary_" + "_".join(sorted(only)) + ".json"
    U.save_json(summary, U.work_path(OUT_DIR, sname))
    print(f"\n[r3] wrote {len(summary)} variant(s) + {sname} to {U.work_path(OUT_DIR)}")


if __name__ == "__main__":
    main()

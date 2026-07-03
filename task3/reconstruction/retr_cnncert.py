"""retr_cnncert.py (agent2 / retrieval) — DECISIVE clean-CNN membership certificate.

The prior membership test used the NOISY MLPs (net.0, E128~0.77): their top-128
subspace is contaminated, so a true member and a same-distribution non-member both
sit at residual ~0.2 -> cannot separate. The CLEAN CNNs (E128=1.0: model2,3,7,10,12)
give a sharp certificate: their fc1 gradient IS rank-128 over the conv-feature space,
so a true batch member's conv-feature lies EXACTLY in rowspace(fc1_gW) -> residual~0,
while any non-member is high. We also do a FULL gallery scan (not SSIM-gated), so a
grainy-recon SSIM miss can't hide a real member.

For each clean CNN and each crop gallery:
  feat = conv_feature(gallery_img) -> flatten -> residual onto top-128 rowspace(fc1_gW).
Controls (calibrate the scale): random [0,1] images (expect HIGH ~non-member) and our
own recon for that model (expect LOW ~member end). A verbatim member shows up as a
cluster of gallery images with residual near the recon-control floor and far below the
same-distribution mass.

Verdict: if min gallery residual ~ recon-control floor (and << same-dist mass) -> SOURCE
FOUND (retrieve). Else -> no verbatim member -> retrieval dead.

Run: sbatch job_retr.sh cnncert   (GPU).   python -u retr_cnncert.py [model10 model2 ...]
"""
import os
import sys
import time
import torch
import numpy as np

import utils as U
import models as M
from retr_lib import topk_right_subspace, subspace_residual
from skimage.metrics import structural_similarity as ssim

_dev = os.environ.get("FL_DEVICE", "cpu")
DEVICE = "cuda" if (_dev == "cuda" and torch.cuda.is_available()) else "cpu"
CROPS = os.environ.get("RETR_CROPS", "cc178,cc148,cc128,r64").split(",")
GALDIR = U.work_path("gallery")
REP = U.work_path("reports", "retr")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def conv_features(model, imgs, bs=512):
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], bs):
            x = imgs[i:i + bs].to(DEVICE)
            f = model.conv_feature(x).reshape(x.shape[0], -1).cpu()
            outs.append(f)
    return torch.cat(outs, 0)


def load_recon(name):
    for p in [U.work_path("outputs_trap", f"trap_{name}.pt"),
              U.work_path("candidates", name, "conv_invert_candidates.pt")]:
        if os.path.exists(p):
            o = torch.load(p, weights_only=False, map_location="cpu")
            x = o["images"] if isinstance(o, dict) and "images" in o else o
            if torch.is_tensor(x) and x.ndim == 4:
                return x.float().clamp(0, 1)
    return None


def run(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    if grad.get("family") != "cnn" or "fc1.weight" not in g:
        log(f"{name}: not a clean-CNN fc1 target; skip"); return None
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state).to(DEVICE)
    gW = g["fc1.weight"].float()                       # (1024, C*H*W)
    V, S = topk_right_subspace(gW, 128)                # feature-space top-128
    V = V.to(DEVICE) if DEVICE == "cuda" else V

    def resid(feat_flat):
        return subspace_residual(feat_flat.to(V.device), V).cpu()

    # controls
    rnd = torch.rand(256, 3, 64, 64)
    r_rand = resid(conv_features(model, rnd))
    rec = load_recon(name)
    r_rec = resid(conv_features(model, rec)) if rec is not None else None
    log(f"\n=== {name} (feat={grad['feature_shape']}) ===")
    log(f"  CONTROL random residual: mean={r_rand.mean():.4f} min={r_rand.min():.4f}")
    if r_rec is not None:
        log(f"  CONTROL recon  residual: mean={r_rec.mean():.4f} min={r_rec.min():.4f}  "
            f"(this is the 'member-end' floor achievable)")

    out = {"name": name, "feature_shape": list(grad["feature_shape"]),
           "rand_resid_mean": round(float(r_rand.mean()), 4),
           "recon_resid_mean": round(float(r_rec.mean()), 4) if r_rec is not None else None,
           "recon_resid_min": round(float(r_rec.min()), 4) if r_rec is not None else None,
           "crops": {}}
    hit_any = False
    for c in CROPS:
        p = os.path.join(GALDIR, f"celeba_top200_{c}.pt")
        if not os.path.exists(p):
            log(f"  {c}: MISSING"); continue
        gal = torch.load(p, weights_only=False, map_location="cpu")
        imgs = gal["images"].float() / 255.0
        rr = resid(conv_features(model, imgs))
        rmin = float(rr.min()); amin = int(rr.argmin())
        n_lt = {t: int((rr < t).sum()) for t in (0.05, 0.10, 0.15, 0.20)}
        # a "hit" = residual near recon floor AND clearly below same-dist mass
        floor = float(r_rec.min()) if r_rec is not None else 0.05
        member_thr = max(1.5 * floor, 0.08)
        hits = int((rr < member_thr).sum())
        hit_any = hit_any or hits > 0
        out["crops"][c] = {"resid_min": round(rmin, 4), "argmin": amin,
                           "resid_p1": round(float(np.percentile(rr, 1)), 4),
                           "resid_median": round(float(rr.median()), 4),
                           "n<thr": n_lt, "member_thr": round(member_thr, 4),
                           "hits": hits}
        log(f"  {c}: min={rmin:.4f} p1={np.percentile(rr,1):.4f} med={rr.median():.4f} "
            f"n<0.05={n_lt[0.05]} n<0.10={n_lt[0.10]} n<0.20={n_lt[0.20]} hits(<{member_thr:.3f})={hits}")
        # save the 16 lowest-residual gallery imgs (+ our recon nearest) to eyeball
        low = torch.argsort(rr)[:16]
        U.ensure_dir(os.path.join(REP, name))
        U.save_image_grid(imgs[low], os.path.join(REP, name, f"lowresid_{c}.png"), ncol=8)
    out["hit_any"] = hit_any
    return out


def main():
    names = sys.argv[1:] or ["model10", "model2", "model3", "model7", "model12"]
    summary = {}
    for n in names:
        try:
            r = run(n)
            if r:
                summary[n] = r
        except Exception as e:
            import traceback
            print(f"{n}: ERROR {e}"); traceback.print_exc()
            summary[n] = {"name": n, "error": str(e)}
    U.save_json(summary, os.path.join(REP, "cnncert.json"))
    print("\n===== CLEAN-CNN MEMBERSHIP VERDICT =====")
    any_hit = False
    for n, r in summary.items():
        if "crops" not in r:
            continue
        gmin = min((cc["resid_min"] for cc in r["crops"].values()), default=1.0)
        hits = sum(cc["hits"] for cc in r["crops"].values())
        any_hit = any_hit or hits > 0
        print(f"  {n}: gallery_min_resid={gmin:.4f} recon_floor={r.get('recon_resid_min')} "
              f"rand={r.get('rand_resid_mean')} total_hits={hits}")
    print(f"\nANY CLEAN-CNN VERBATIM MEMBER: {any_hit}")
    print("wrote", os.path.join(REP, "cnncert.json"))


if __name__ == "__main__":
    main()

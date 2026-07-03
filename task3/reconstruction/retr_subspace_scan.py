"""Agent R / STEP 2b — GRAIN-INDEPENDENT membership certificate.

SSIM is defeated by recon grain, so it cannot confirm/deny verbatim membership on
noisy queries. This test does NOT use any reconstruction. For each MLP model it
projects the ENTIRE top-200 CelebA gallery (at each crop, + flips) onto the top-128
right-singular subspace of net.0's weight-gradient and reads the residual
DISTRIBUTION:

  * a verbatim member (at the correct crop/normalisation) lies in rowspace(gW) ->
    residual near the gradient noise-floor, well BELOW generic CelebA faces;
  * if ~128 gallery images form a distinct low-residual cluster at ONE crop and NOT
    under a wrong-model subspace (permutation control) -> verbatim membership CONFIRMED.

Controls: (a) same gallery vs a DIFFERENT model's subspace; (b) vs a random 128-dim
subspace. Reports percentiles + counts below thresholds, saves the lowest-residual
gallery grid (the putative retrieved members). Read-only except reports/retr.
"""
import os, sys, json, time
import numpy as np
import torch

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
from utils import save_image_grid
from retr_lib import load_grad_firstlayer, topk_right_subspace, subspace_residual, augment_dc

MODELS = os.environ.get("RETR_MODELS", "model1,model4,model5,model8").split(",")
CROPS = os.environ.get("RETR_CROPS", "cc148,cc128,cc178,r64").split(",")
GALDIR = os.path.join(WORK, "gallery")
REP = os.path.join(WORK, "reports", "retr")
os.makedirs(REP, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pctl(x):
    return {k: round(float(np.percentile(x, k)), 4) for k in (0, 1, 2, 5, 10, 50)}


def main():
    galleries = {}
    meta = {}
    for c in CROPS:
        p = os.path.join(GALDIR, f"celeba_top200_{c}.pt")
        if os.path.exists(p):
            g = torch.load(p, weights_only=False, map_location="cpu")
            galleries[c] = g["images"].float() / 255.0
            meta[c] = g
    log(f"galleries: {list(galleries)}")

    # subspaces per model
    subs = {}
    for m in MODELS:
        gW, gb, key = load_grad_firstlayer(DATA, m)
        if key != "net.0.weight":
            log(f"{m}: key={key} skip"); continue
        V, S = topk_right_subspace(gW, 128)
        subs[m] = (V, augment_dc(V))
    # random control subspace (orthonormal 128 x 12288)
    Rrand = torch.linalg.qr(torch.randn(12288, 128))[0].t().contiguous()

    report = {"models": {}}
    for m in MODELS:
        if m not in subs:
            continue
        V, Vaug = subs[m]
        mrep = {"crops": {}}
        log(f"\n=== {m} ===")
        for c, G in galleries.items():
            r = subspace_residual(G, V).numpy()
            ra = subspace_residual(G, Vaug).numpy()
            # counts below thresholds (relative to the bulk median)
            med = float(np.median(ra))
            cnt = {f"<{t}": int((ra < t).sum()) for t in (0.05, 0.1, 0.15, 0.2, 0.3)}
            mrep["crops"][c] = {"resid_pctl": pctl(r), "resid_aug_pctl": pctl(ra),
                                "aug_median": round(med, 4), "n_below": cnt}
            log(f"  {c}: aug pctl {pctl(ra)}  median={med:.3f}  below={cnt}")
        # permutation control: this gallery (best crop = min p1) vs OTHER models' subspaces
        report["models"][m] = mrep

    # cross-model control: gallery cc148 residual under each model's subspace + random
    if "cc148" in galleries:
        G = galleries["cc148"]
        ctrl = {}
        for m in MODELS:
            if m in subs:
                ctrl[m] = round(float(np.percentile(subspace_residual(G, subs[m][1]).numpy(), 1)), 4)
        ctrl["RANDOM_subspace"] = round(float(np.percentile(subspace_residual(G, augment_dc(Rrand)).numpy(), 1)), 4)
        report["cc148_p1_by_subspace"] = ctrl
        log(f"\ncc148 p1 residual by subspace (control): {ctrl}")

    # For the strongest (model,crop) = lowest p1 aug residual, dump the lowest-residual grid
    best = None
    for m, mr in report["models"].items():
        for c, cr in mr["crops"].items():
            p1 = cr["resid_aug_pctl"]["1"]
            if best is None or p1 < best[2]:
                best = (m, c, p1)
    if best:
        m, c, p1 = best
        V, Vaug = subs[m]
        G = galleries[c]
        ra = subspace_residual(G, Vaug).numpy()
        order = np.argsort(ra)[:64]
        save_image_grid(G[torch.from_numpy(order)], os.path.join(REP, f"lowresid_{m}_{c}.png"), ncol=8)
        report["strongest"] = {"model": m, "crop": c, "p1_aug": p1,
                               "low_ids": meta[c]["celeb_id"][order].tolist()[:32]}
        log(f"\nstrongest: {m}/{c} p1_aug={p1:.4f}; saved lowresid grid")

    with open(os.path.join(REP, "subspace_scan.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("WROTE reports/retr/subspace_scan.json")
    print("__DONE__")


if __name__ == "__main__":
    main()

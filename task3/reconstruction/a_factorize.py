"""runs/a* — Direction A: rank-denoised, bias-anchored, box-constrained GATED
SEMI-NMF source separation (method recommended by the factorization derivation).

gW = A X, A=(out,128) ReLU-gated backprop signals, X=(128,in) images in [0,1].
1. rank-r denoise: gWr = gW Vr Vrᵀ (Vr = top-r right singular subspace).
2. warm start X from ratio-cluster centroids (the proven single-sample leaks).
3. alternate: A-step closed form with bias anchor + soft-threshold sparsity;
   X-step FISTA projected to [0,1].
4. composite N_clean gate: in-rowspace AND TV-band AND box-native AND distinct AND
   NOT-a-nonnegative-blend-of-others (the only test that attacks the rowspace trap).

Compares N_clean vs the ratio baseline. Writes runs/<RUN>/ only. Read-only elsewhere.
Env: A_RUN(a1) A_MODEL(model8) A_R(128) A_OUTER(50) A_INNER(30) FL_DEVICE.
"""
import os, sys, json, time
import numpy as np, torch

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
import utils as U
from retr_lib import topk_right_subspace
from rec_lib import total_variation, ratio_rows, cluster_labels, per_image_minmax

RUN = os.environ.get("A_RUN", "a1"); MODEL = os.environ.get("A_MODEL", "model8")
R = int(os.environ.get("A_R", "128"))
OUTER = int(os.environ.get("A_OUTER", "50")); INNER = int(os.environ.get("A_INNER", "30"))
DEV = os.environ.get("FL_DEVICE", "cpu")
if DEV == "cuda" and not torch.cuda.is_available():
    DEV = "cpu"
RUNDIR = os.path.join(WORK, "runs", RUN)
for s in ("candidates", "grids", "reports", "logs"):
    os.makedirs(os.path.join(RUNDIR, s), exist_ok=True)


def log(m): print(m, flush=True)


def composite_clean(X, Vr, imgshape, tv_hi=0.16):
    """Return (n_clean, keep_mask, imgs) per the composite gate."""
    imgs = per_image_minmax(X.reshape(-1, *imgshape))
    flat = X.reshape(X.shape[0], -1)
    resid = ((flat - (flat @ Vr) @ Vr.t()).norm(dim=1) / (flat.norm(dim=1) + 1e-9))
    tv = total_variation(imgs)
    boxnat = ((flat >= -0.02) & (flat <= 1.02)).float().mean(1)
    g16 = torch.nn.functional.interpolate(imgs.mean(1, keepdim=True), size=(16, 16),
              mode="bilinear", align_corners=False).reshape(X.shape[0], -1)
    base = (resid < 0.15) & (tv < tv_hi) & (boxnat > 0.88)
    # distinct: greedy dedup on 16x16
    order = torch.argsort(tv)  # cleanest first
    kept, keptvec = [], []
    for i in order.tolist():
        if not base[i]:
            continue
        if all(((g16[i]-g16[j])**2).mean().sqrt() > 0.045 for j in keptvec):
            kept.append(i); keptvec.append(i)
    # not-a-blend: reject a kept source well-fit by a nonneg combo of 3 nearest kept
    keep = torch.zeros(X.shape[0], dtype=torch.bool)
    kv = torch.stack([g16[i] for i in kept]) if kept else g16[:0]
    final = []
    for a, i in enumerate(kept):
        others = [kept[b] for b in range(len(kept)) if b != a]
        if len(others) >= 3:
            D = ((g16[others] - g16[i]).pow(2).mean(1))
            nn = [others[b] for b in torch.argsort(D)[:3].tolist()]
            Bm = torch.stack([g16[j] for j in nn]).t()   # (256,3)
            w = torch.linalg.lstsq(Bm, g16[i]).solution.clamp_min(0)
            fit = 1 - ((Bm @ w - g16[i]).norm() / (g16[i].norm() + 1e-9))
            if fit > 0.92:            # well-explained as blend of neighbours -> drop
                continue
        keep[i] = True; final.append(i)
    return len(final), keep, imgs


def main():
    g = U.load_grad(MODEL)["gradients"]
    key = "net.0.weight" if "net.0.weight" in g else "fc1.weight"
    gW = g[key].float(); gb = g[key.replace("weight", "bias")].float()
    imgshape = (3, 64, 64) if key == "net.0.weight" else tuple(U.load_grad(MODEL)["feature_shape"])
    out, inn = gW.shape
    log(f"{MODEL}: {key} {tuple(gW.shape)} dev={DEV} img={imgshape}")

    # ratio baseline
    ratio, reliable = ratio_rows(gW, gb)
    rel = torch.where(reliable & torch.isfinite(ratio).all(1))[0]
    lab, k = cluster_labels(per_image_minmax(ratio[rel]).reshape(len(rel), -1), 128)
    Xb = torch.stack([ratio[rel[lab == c][int(torch.argmax(gb[rel[lab == c]].abs()))]] for c in range(k)])

    # rank-r denoise
    Vr_rows, S = topk_right_subspace(gW, R)          # (R,in)
    Vr = Vr_rows.t().contiguous()                     # (in,R)
    gWr = (gW @ Vr) @ Vr.t()

    nb, _, _ = composite_clean(Xb, Vr, imgshape)
    log(f"  RATIO baseline: composite N_clean={nb}")

    # warm start X (pad to 128)
    X = torch.zeros(128, inn); X[:Xb.shape[0]] = Xb.clamp(0, 1)
    X, gWr, gb2, Vr = X.to(DEV), gWr.to(DEV), gb.to(DEV), Vr.to(DEV)
    ones = torch.ones(128, 1, device=DEV); lam = 1.0
    t0 = time.time()
    for it in range(OUTER):
        XXt = X @ X.t() + lam * (ones @ ones.t())
        RHS = gWr @ X.t() + lam * (gb2.unsqueeze(1) @ ones.t())
        A = torch.linalg.solve(XXt.t(), RHS.t()).t()      # (out,128)
        thr = 0.05 * A.abs().max()
        A = torch.sign(A) * (A.abs() - thr).clamp_min(0)
        L = torch.linalg.matrix_norm(A, 2) ** 2 + 1e-6
        Z = X.clone(); tt = 1.0
        for _ in range(INNER):
            grad = A.t() @ (A @ Z - gWr)
            Xn = (Z - grad / L).clamp(0, 1)
            tn = 0.5 * (1 + (1 + 4 * tt * tt) ** 0.5)
            Z = Xn + ((tt - 1) / tn) * (Xn - X); X, tt = Xn, tn
        if it % 10 == 0 or it == OUTER - 1:
            rr = float((gWr - A @ X).norm() / gWr.norm())
            log(f"  outer{it}: recon_res={rr:.3f} ({time.time()-t0:.0f}s)")
    X = X.cpu(); Vr = Vr.cpu()
    nclean, keep, imgs = composite_clean(X, Vr, imgshape)
    rr = float((gWr.cpu() - (A.cpu() @ X)).norm() / gWr.cpu().norm())
    log(f"  SEMI-NMF: composite N_clean={nclean} (ratio {nb}) recon_res={rr:.3f}")
    torch.save({"images": imgs.float(), "keep": keep, "n_clean": nclean,
                "source": f"seminmf:{MODEL}:R{R}"},
               os.path.join(RUNDIR, "candidates", f"{MODEL}_seminmf.pt"))
    U.save_image_grid(imgs[torch.argsort(total_variation(imgs))[:64]],
                      os.path.join(RUNDIR, "grids", f"{MODEL}_seminmf_cleanest.png"))
    json.dump({"model": MODEL, "ratio_Nclean": nb, "seminmf_Nclean": nclean,
               "recon_res": round(rr, 4), "R": R},
              open(os.path.join(RUNDIR, "reports", f"seminmf_{MODEL}.json"), "w"), indent=2)
    print("__DONE__")


if __name__ == "__main__":
    main()

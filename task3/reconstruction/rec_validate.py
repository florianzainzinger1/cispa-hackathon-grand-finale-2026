import os, sys, torch
WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
OUT = os.path.join(WORK, "outputs_trap")
mods = ["model1","model2","model3","model4","model5","model6","model7","model8","model10","model12"]
allok = True
for m in mods:
    p = os.path.join(OUT, f"rec_{m}.pt")
    if not os.path.exists(p):
        print(f"{m}: MISSING"); allok=False; continue
    d = torch.load(p, weights_only=False, map_location="cpu")
    x = d["images"].float()
    n_uni = len({hash(x[i].numpy().tobytes()) for i in range(x.shape[0])})
    resid = d.get("subspace_resid")
    nlow = int((resid < 0.15).sum()) if resid is not None else -1
    ok = (tuple(x.shape)==(128,3,64,64) and torch.isfinite(x).all()
          and float(x.min())>=-1e-4 and float(x.max())<=1+1e-4 and n_uni==128)
    allok = allok and ok
    print(f"{m}: shape={tuple(x.shape)} range[{x.min():.3f},{x.max():.3f}] "
          f"uniq={n_uni} n_clean={d.get('n_clean')} resid<0.15={nlow} OK={ok}")
print("ALL_OK" if allok else "SOME_FAIL")

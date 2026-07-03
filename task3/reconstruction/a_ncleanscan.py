"""runs/a* — grounding scan: count CLEAN, DISTINCT, NON-BLEND faces (the quantity
that actually scores under Hungarian one-to-one SSIM) for the c3 baseline vs my
trap reconstructions, per model. Identifies where a real shippable gain exists.
Read-only. Writes runs/a1/reports/nclean_scan.json."""
import os, sys, json
import numpy as np, torch
TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
sys.path.insert(0, os.path.join(WORK, "code"))
import utils as U
from rec_lib import total_variation, per_image_minmax

MODS = ["model1", "model4", "model5", "model8", "model6", "model10", "model12",
        "model2", "model3", "model7"]


def clean_distinct(imgs, tv_hi=0.16):
    imgs = per_image_minmax(imgs.float())
    tv = total_variation(imgs)
    g16 = torch.nn.functional.interpolate(imgs.mean(1, keepdim=True), size=(16, 16),
              mode="bilinear", align_corners=False).reshape(imgs.shape[0], -1)
    order = torch.argsort(tv)
    kept = []
    for i in order.tolist():
        if tv[i] >= tv_hi:
            continue
        if all(((g16[i]-g16[j])**2).mean().sqrt() > 0.045 for j in kept):
            kept.append(i)
    # not-a-blend
    final = 0
    for a, i in enumerate(kept):
        others = [kept[b] for b in range(len(kept)) if b != a]
        if len(others) >= 3:
            D = ((g16[others]-g16[i]).pow(2).mean(1))
            nn = [others[b] for b in torch.argsort(D)[:3].tolist()]
            Bm = torch.stack([g16[j] for j in nn]).t()
            w = torch.linalg.lstsq(Bm, g16[i]).solution.clamp_min(0)
            if 1 - ((Bm @ w - g16[i]).norm()/(g16[i].norm()+1e-9)) > 0.92:
                continue
        final += 1
    return len(kept), final


c3 = torch.load(os.path.join(WORK, "outputs_v3", "submission_v3.pt"),
                weights_only=False, map_location="cpu")
out = {}
for m in MODS:
    row = {}
    d, f = clean_distinct(c3[m])
    row["c3_distinct"] = d; row["c3_clean_nonblend"] = f
    rp = os.path.join(WORK, "outputs_trap", f"rec_{m}.pt")
    if os.path.exists(rp):
        r = torch.load(rp, weights_only=False, map_location="cpu")
        im = r["images"] if isinstance(r, dict) and "images" in r else r
        d2, f2 = clean_distinct(im)
        row["rec_distinct"] = d2; row["rec_clean_nonblend"] = f2
    out[m] = row
    print(f"{m}: c3 clean_nonblend={row['c3_clean_nonblend']}/128 "
          f"| rec={row.get('rec_clean_nonblend','-')}  (c3 distinct {row['c3_distinct']})", flush=True)
json.dump(out, open(os.path.join(WORK, "runs", "a1", "reports", "nclean_scan.json"), "w"), indent=2)
print("__DONE__")

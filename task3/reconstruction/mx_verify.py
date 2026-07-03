"""mx_verify.py — confirm each candidate is a clean single-variable swap off vithi45:
exactly ONE model differs; the other 11 are byte-identical."""
import os, sys, torch
WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
BASE = f"{WORK}/outputs_final/test_vithi45.pt"

CANDS = {
    "cand_m12_color1.pt": "model12",
    "cand_m12_color2.pt": "model12",
    "cand_m12_color3.pt": "model12",
    "cand_m10_clean1.pt": "model10",
    "cand_m10_clean2.pt": "model10",
    "cand_m3_ssiminv.pt": "model3",
    "cand_m7_ssiminv.pt": "model7",
    "cand_m2_ssiminv.pt": "model2",
}

base = torch.load(BASE, map_location="cpu", weights_only=False)
allok = True
for fn, expect in CANDS.items():
    p = f"{WORK}/outputs_mx/{fn}"
    if not os.path.exists(p):
        print(f"{fn:24s} MISSING"); allok = False; continue
    sub = torch.load(p, map_location="cpu", weights_only=False)
    diffs = []
    for mk in base:
        if not torch.equal(base[mk].float(), sub[mk].float()):
            diffs.append(mk)
    ok = (diffs == [expect])
    allok &= ok
    print(f"{fn:24s} changed={diffs}  expect=[{expect}]  {'OK' if ok else 'BAD'}")
print("ALL_SINGLE_VAR_OK" if allok else "SOME_BAD")

"""mx_compare.py — stack the SAME faces from base vs candidates for direct A/B.
Rows: vithi45 base / color1 / color2  (model12), then a model10 base/clean1 block.
Uses first 16 imgs, each as a row-block."""
import os, sys, torch
WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
sys.path.insert(0, f"{WORK}/code")
import utils as U

base = torch.load(f"{WORK}/outputs_final/test_vithi45.pt", map_location="cpu", weights_only=False)

def load(fn): return torch.load(f"{WORK}/outputs_mx/{fn}", map_location="cpu", weights_only=False)

N = 16
# model12: base, color1, color2
b12 = base["model12"][:N]
c1  = load("cand_m12_color1.pt")["model12"][:N]
c2  = load("cand_m12_color2.pt")["model12"][:N]
stack12 = torch.cat([b12, c1, c2], dim=0)  # 3 blocks of 16
U.save_image_grid(stack12, f"{WORK}/reports/inspect_new/cmp_m12.png", ncol=N, pad=1, upscale=3)
print("[cmp] reports/inspect_new/cmp_m12.png  rows: base / color1 / color2")

# model10: base, clean1, clean2
b10 = base["model10"][:N]
d1  = load("cand_m10_clean1.pt")["model10"][:N]
d2  = load("cand_m10_clean2.pt")["model10"][:N]
stack10 = torch.cat([b10, d1, d2], dim=0)
U.save_image_grid(stack10, f"{WORK}/reports/inspect_new/cmp_m10.png", ncol=N, pad=1, upscale=3)
print("[cmp] reports/inspect_new/cmp_m10.png  rows: base / clean1 / clean2")

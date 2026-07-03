"""mx_montage.py — render an 8x8 montage (first 64 imgs) of a given model from a candidate .pt."""
import os, sys
import torch
sys.path.insert(0, os.environ.get("FL_WORK_ROOT", ".") + "/code")
import utils as U

WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")

def main():
    cand_path = sys.argv[1]       # e.g. outputs_mx/cand_m12_color1.pt
    model_key = sys.argv[2]       # e.g. model12
    out_png = sys.argv[3]         # e.g. reports/inspect_new/cand_m12_color1.png
    sub = torch.load(f"{WORK}/{cand_path}", map_location="cpu", weights_only=False)
    imgs = sub[model_key][:64]
    U.save_image_grid(imgs, f"{WORK}/{out_png}", ncol=8, pad=1, upscale=2)
    print(f"[montage] {out_png}  gmean={imgs.mean():.4f}")

if __name__ == "__main__":
    main()

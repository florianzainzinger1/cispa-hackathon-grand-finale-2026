"""Agent R / STEP 1 — build the CelebA TOP-200-IDENTITY gallery at 4 crops + hflip.

Per RESEARCH_FINDINGS.md §2: source is (hypothesised) CelebA restricted to the 200
identities with the most images (Fantomas, arXiv 2210.10651). Transforms to try, in
order: CenterCrop 148 -> 128 -> 178 -> plain Resize(64), ToTensor only ([0,1], no
normalize). We also add each image's horizontal flip (Fantomas lead).

Writes one file per crop: gallery/celeba_top200_{crop}.pt =
  {images: uint8 (M,3,64,64), celeb_id:(M,), src_index:(M,), is_flip:(M,), crop, src}
where M = (#top200 images) * 2 (orig + hflip).

Login node = download (needs internet). Progress -> gallery/build_log.txt, and a
gallery/BUILD_DONE.json marker on success so a poller can detect completion.
"""
import os, sys, io, time, json, traceback
from collections import Counter
import numpy as np
import torch
from PIL import Image

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
sys.path.insert(0, os.path.join(WORK, "runs", "pylibs"))
os.environ.setdefault("HF_HOME", os.path.join(WORK, "runs", "hf_cache"))

SRC = os.environ.get("RETR_SRC", "flwrlabs/celeba")
SPLIT = os.environ.get("RETR_SPLIT", "train")
TOPK = int(os.environ.get("RETR_TOPID", "200"))
CROPS = os.environ.get("RETR_CROPS", "cc148,cc128,cc178,r64").split(",")
OUT = os.path.join(WORK, "gallery")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "build_log.txt")
open(LOG, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def variant(img, crop):
    """PIL RGB -> np.uint8 (64,64,3). crop in {ccNNN, r64}."""
    W, H = img.size
    if crop == "r64":
        cc = img.resize((64, 64), Image.BICUBIC)
    else:
        c = int(crop[2:])
        s = min(c, W, H)
        left, top = (W - s) // 2, (H - s) // 2
        cc = img.crop((left, top, left + s, top + s)).resize((64, 64), Image.BICUBIC)
    return np.asarray(cc.convert("RGB"), dtype=np.uint8)


def main():
    from datasets import load_dataset
    log(f"loading {SRC} split={SPLIT}  HF_HOME={os.environ['HF_HOME']}")
    ds = load_dataset(SRC, split=SPLIT)
    cols = ds.column_names
    log(f"columns={cols}  n={len(ds)}")
    img_col = next((c for c in ("image", "img", "jpg", "picture") if c in cols), cols[0])
    id_col = next((c for c in ("celeb_id", "identity", "id", "label", "celebId") if c in cols), None)
    if id_col is None:
        log("!! NO identity column -> cannot build top-200; aborting"); return

    log(f"img_col={img_col} id_col={id_col}; reading identities...")
    ids_all = np.asarray(ds[id_col])                     # fast column read (no decode)
    cnt = Counter(ids_all.tolist())
    top = [i for i, _ in cnt.most_common(TOPK)]
    topset = set(top)
    sel = np.where(np.isin(ids_all, top))[0]
    log(f"top-{TOPK} identities cover {len(sel)} images "
        f"(min/median/max per-id: {min(cnt[i] for i in top)}/"
        f"{int(np.median([cnt[i] for i in top]))}/{max(cnt[i] for i in top)})")

    # native size probe
    def to_pil(x):
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, dict) and "bytes" in x:
            return Image.open(io.BytesIO(x["bytes"]))
        return Image.fromarray(np.asarray(x))
    ex0 = to_pil(ds[int(sel[0])][img_col]).convert("RGB")
    log(f"native image size = {ex0.size} (W,H)")

    M = len(sel) * 2
    buffers = {c: np.empty((M, 64, 64, 3), dtype=np.uint8) for c in CROPS}
    cid = np.empty(M, dtype=np.int64)
    sidx = np.empty(M, dtype=np.int64)
    flip = np.zeros(M, dtype=np.uint8)
    t0 = time.time()
    for j, idx in enumerate(sel.tolist()):
        im = to_pil(ds[idx][img_col]).convert("RGB")
        imf = im.transpose(Image.FLIP_LEFT_RIGHT)
        for c in CROPS:
            buffers[c][2 * j] = variant(im, c)
            buffers[c][2 * j + 1] = variant(imf, c)
        cid[2 * j] = cid[2 * j + 1] = int(ids_all[idx])
        sidx[2 * j] = sidx[2 * j + 1] = idx
        flip[2 * j + 1] = 1
        if (j + 1) % 500 == 0:
            dt = time.time() - t0
            log(f"  {j+1}/{len(sel)} imgs  {dt:.0f}s ({(j+1)/dt:.0f}/s)")

    for c in CROPS:
        arr = torch.from_numpy(buffers[c]).permute(0, 3, 1, 2).contiguous()  # (M,3,64,64)
        path = os.path.join(OUT, f"celeba_top200_{c}.pt")
        torch.save({"images": arr, "celeb_id": torch.from_numpy(cid),
                    "src_index": torch.from_numpy(sidx), "is_flip": torch.from_numpy(flip),
                    "crop": c, "src": SRC, "n_ids": TOPK}, path)
        log(f"saved {path}  {tuple(arr.shape)}  {arr.numel()/1e9:.2f}GB")

    with open(os.path.join(OUT, "BUILD_DONE.json"), "w") as f:
        json.dump({"src": SRC, "n_ids": TOPK, "n_images": int(len(sel)),
                   "M_with_flip": int(M), "crops": CROPS,
                   "native_size": list(ex0.size)}, f, indent=2)
    log("BUILD DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        raise

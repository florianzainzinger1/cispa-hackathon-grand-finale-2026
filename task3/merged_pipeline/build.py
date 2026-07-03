"""End-to-end merged pipeline -> submission.pt.

  reconstruct (OUR low-rank MLP + tanh model2)  ->  face bank
   ->  v3 diverse selection (bank-fill ViT)  ->  global mean face
   ->  metric denoise + mean-blend + pib (+ OUR chroma)  ->  validate -> save

Run:
  python build.py                 # full build (caches recon pools in work/)
  python build.py --no-cache      # force re-reconstruction
  python build.py --submit        # build then upload (reads ../../.env API_KEY)
"""
import os
import sys
import json
import torch

import common as U
import reconstruct as R
import selection as S
import postproc as P

CACHE = U.work_path("pools")
OUT = os.path.join(U.HERE, "submission.pt")


def get_pools(use_cache=True, dev="cpu"):
    U.ensure_dir(CACHE)
    pools, metas = {}, {}
    for name in U.MODELS:
        cp = os.path.join(CACHE, f"{name}.pt")
        if use_cache and os.path.exists(cp):
            blob = torch.load(cp, weights_only=False)
            pools[name], metas[name] = blob["pool"], blob["meta"]
            print(f"  {name}: cached pool "
                  f"{tuple(pools[name].shape) if pools[name] is not None else None}")
            continue
        pool, meta = R.reconstruct(name, dev=dev)
        pools[name], metas[name] = pool, meta
        torch.save({"pool": pool, "meta": meta}, cp)
        print(f"  {name}: {meta.get('family')} "
              f"{'(vit->bank)' if pool is None else tuple(pool.shape)} "
              f"lowrank_k={meta.get('lowrank_k','-')}")
    return pools, metas


def validate(sub):
    errs = []
    for m in U.MODELS:
        x = sub[m]
        if tuple(x.shape) != (128, 3, 64, 64):
            errs.append(f"{m}: shape {tuple(x.shape)}")
        if x.dtype != torch.float32:
            errs.append(f"{m}: dtype {x.dtype}")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            errs.append(f"{m}: range/finite")
        if torch.unique(x.reshape(128, -1), dim=0).shape[0] < 128:
            errs.append(f"{m}: <128 unique")
    return errs


def main():
    use_cache = "--no-cache" not in sys.argv
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("== Stage 1-2: reconstruct ==")
    pools, metas = get_pools(use_cache, dev)

    print("== face bank (from clean recoveries 1/4/5/10/12) ==")
    bank = S.build_face_bank(pools)
    print(f"  bank: {None if bank is None else tuple(bank.shape)}")

    print("== Stage 3: select 128 diverse ==")
    sel = {}
    for name in U.MODELS:
        s, info = S.select_model(name, pools[name], bank)
        sel[name] = s
        print(f"  {json.dumps(info)}")

    # global mean face = mean over all selected images (smooth SSIM-safe prior)
    mean_face = torch.stack([sel[m] for m in U.MODELS]).reshape(-1, 3, 64, 64).mean(0)

    print("== Stage 4-6: metric denoise + mean-blend + pib (+ chroma) ==")
    sub = {}
    for name in U.MODELS:
        sub[name] = P.postprocess_model(name, sel[name], mean_face).to(torch.float32).contiguous()
        print(f"  {name}: done  tv={float(U.total_variation(sub[name]).mean()):.4f}")

    errs = validate(sub)
    if errs:
        print("VALIDATION ERRORS:", *errs, sep="\n  - "); raise SystemExit(1)
    torch.save(sub, OUT)
    print(f"\nsaved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)  VALID")

    # preview grids
    viz = U.ensure_dir(os.path.join(U.HERE, "viz"))
    for name in U.MODELS:
        U.save_image_grid(sub[name], os.path.join(viz, f"{name}.png"), ncol=16)
    U.save_image_grid(torch.cat([sub[m][:16] for m in U.MODELS], 0),
                      os.path.join(viz, "ALL_first16.png"), ncol=16)
    print(f"wrote grids -> {viz}")

    if "--submit" in sys.argv:
        submit(OUT)


def submit(path):
    import requests
    env = os.path.join(os.path.dirname(U.TASK3), ".env")
    key = None
    for line in open(env):
        if line.strip().startswith("API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    r = requests.post("http://<SUBMIT_SERVER>/submit/21-fl-audit",
                      headers={"X-API-Key": key},
                      files={"file": (os.path.basename(path), open(path, "rb"),
                                      "application/octet-stream")})
    print("submit:", r.status_code, r.text)


if __name__ == "__main__":
    main()

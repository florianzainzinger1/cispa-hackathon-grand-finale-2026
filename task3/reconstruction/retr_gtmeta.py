import os, torch
WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
gt = torch.load(os.path.join(WORK, "outputs_retr", "ground_truth.pt"),
                weights_only=False, map_location="cpu")
print("TOP KEYS:", list(gt))
for m in list(gt):
    v = gt[m]
    if isinstance(v, dict):
        info = {}
        for k, x in v.items():
            if torch.is_tensor(x):
                info[k] = f"tensor{tuple(x.shape)}:{x.dtype}"
            elif isinstance(x, (list, tuple)):
                info[k] = f"{type(x).__name__}[{len(x)}]" + (f" e0={x[0]}" if len(x) and not hasattr(x[0], '__len__') else "")
            else:
                info[k] = x
        print(f"\n{m}: {info}")
    elif torch.is_tensor(v):
        print(f"\n{m}: tensor{tuple(v.shape)}:{v.dtype} range[{v.min():.3f},{v.max():.3f}]")
    else:
        print(f"\n{m}: {type(v)} = {v}")
print("\n__DONE__")

"""Agent R / STEP 0 — data provenance.

Inspect all 12 challenge models (architecture + classification-head dim = the
200-class fingerprint), the gradient metadata, and query the HuggingFace repos
`SprintML/FL_Data_Reconstruction` + `SprintML/hackathon` for any generation /
metadata / preprocessing script. Login node has internet.

Read-only w.r.t. challenge data. Writes nothing except stdout (captured to a log).
"""
import os
import sys
import json
import urllib.request
import torch

DATA_ROOT = os.environ.get(
    "FL_DATA_ROOT",
    "/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction",
)
MODELS = [f"model{i}" for i in range(1, 13)]


def hr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74, flush=True)


def load(sub, name):
    return torch.load(
        os.path.join(DATA_ROOT, sub, f"{name}.pt"),
        weights_only=False, map_location="cpu",
    )


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- models
hr("MODEL ARCHITECTURES + CLASSIFICATION HEADS")
head_dims = {}
for m in MODELS:
    try:
        sd = load("models", m)
    except Exception as e:
        print(f"{m}: FAILED to load: {e}")
        continue
    if not isinstance(sd, dict):
        print(f"{m}: state is {type(sd)}")
        continue
    print(f"\n--- {m} --- ({len(sd)} tensors)")
    last_out = None
    for k, v in sd.items():
        sh = tuple(v.shape) if hasattr(v, "shape") else None
        print(f"    {k:44s} {sh}")
        if hasattr(v, "shape") and v.ndim == 2:      # linear weight (out,in)
            last_out = v.shape[0]
    head_dims[m] = last_out
    print(f"    >>> final linear out-dim (num classes?) = {last_out}")

# ---------------------------------------------------------------- gradients
hr("GRADIENT METADATA")
for m in MODELS:
    try:
        g = load("gradients", m)
    except Exception as e:
        print(f"{m}: FAILED: {e}")
        continue
    meta = {k: g.get(k) for k in ("family", "activation", "feature_shape", "batch_size")}
    gk = list(g.get("gradients", {}).keys())
    head_g = None
    for k in gk:
        t = g["gradients"][k]
        if hasattr(t, "ndim") and t.ndim == 2:
            head_g = tuple(t.shape)
    print(f"{m}: {meta} | n_grad={len(gk)} | last2d_grad={head_g}")

hr("HEAD-DIM SUMMARY (the 200-class fingerprint)")
print(json.dumps(head_dims, indent=2))

# ---------------------------------------------------------------- HF repos
hr("HUGGINGFACE REPO PROVENANCE")
REPOS = [
    ("datasets", "SprintML/FL_Data_Reconstruction"),
    ("datasets", "SprintML/hackathon"),
    ("models",   "SprintML/FL_Data_Reconstruction"),
]
TEXT_EXT = (".py", ".md", ".json", ".yaml", ".yml", ".txt", ".cfg",
            ".sh", ".ipynb", ".toml", ".csv")

for repo_type, rid in REPOS:
    base = f"https://huggingface.co/api/{repo_type}/{rid}"
    tree_url = f"{base}/tree/main?recursive=true"
    print(f"\n### {repo_type}/{rid}")
    try:
        tree = json.loads(fetch(tree_url))
    except Exception as e:
        print(f"  tree ERR: {e}")
        continue
    files = [e for e in tree if e.get("type") == "file"]
    print(f"  {len(files)} files:")
    for e in files:
        print(f"    {e['path']:50s} {e.get('size', '?')} bytes")
    # auto-fetch small text files (generation/metadata scripts)
    raw_base = (f"https://huggingface.co/datasets/{rid}/raw/main"
                if repo_type == "datasets"
                else f"https://huggingface.co/{rid}/raw/main")
    for e in files:
        p = e["path"]
        if p.lower().endswith(TEXT_EXT) and e.get("size", 1e9) < 200_000:
            try:
                body = fetch(f"{raw_base}/{p}")
            except Exception as ex:
                print(f"\n  --- {p}: raw ERR {ex}")
                continue
            print(f"\n  ===== {p} ({len(body)} chars) =====")
            print(body[:8000])

# ---------------------------------------------------------------- org listing
hr("SprintML ORG LISTING")
for url in [
    "https://huggingface.co/api/datasets?author=SprintML&limit=200&full=true",
    "https://huggingface.co/api/models?author=SprintML&limit=200",
]:
    try:
        print(f"\n# {url}")
        print(fetch(url)[:4000])
    except Exception as e:
        print(f"  ERR {e}")

print("\nDONE.", flush=True)

"""Agent R env probe: what's importable, disk space, where our best recons live."""
import os, sys, shutil, importlib, json

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
PYLIBS = f"{WORK}/runs/pylibs"
if os.path.isdir(PYLIBS):
    sys.path.insert(0, PYLIBS)

print("PY:", sys.version.split()[0])
for mod in ["torch", "numpy", "sklearn", "skimage", "scipy", "PIL",
            "huggingface_hub", "datasets", "faiss", "insightface",
            "facenet_pytorch", "cv2", "onnxruntime", "requests", "tqdm"]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK  {mod:16s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  --  {mod:16s} MISSING ({type(e).__name__})")

print("\nDISK (scratch):")
for p in [TEAM, WORK, f"{WORK}/gallery"]:
    try:
        t, u, f = shutil.disk_usage(p)
        print(f"  {p}: free {f/1e9:.1f} GB / total {t/1e9:.1f} GB")
        break
    except Exception as e:
        print(f"  {p}: {e}")

print("\nRECON SOURCES (look for MLP recons model1/4/5):")
for d in ["outputs_c3", "outputs_v3", "candidates_v3", "candidates",
          "candidates_v2", "outputs", "gallery", "outputs_retr"]:
    dp = os.path.join(WORK, d)
    if os.path.isdir(dp):
        entries = sorted(os.listdir(dp))
        print(f"  {d}/  -> {entries[:12]}{' ...' if len(entries) > 12 else ''}")
    else:
        print(f"  {d}/  (absent)")

# Peek into candidates_v3 per-model layout
cv3 = os.path.join(WORK, "candidates_v3")
if os.path.isdir(cv3):
    for m in ["model1", "model4", "model5"]:
        mp = os.path.join(cv3, m)
        if os.path.isdir(mp):
            print(f"    candidates_v3/{m}/ -> {sorted(os.listdir(mp))[:15]}")

# internet check
try:
    import urllib.request
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                 headers={"User-Agent": "curl/8"})
    urllib.request.urlopen(req, timeout=10)
    print("\nINTERNET: reachable (login node)")
except Exception as e:
    print(f"\nINTERNET: {type(e).__name__} {e}")
print("DONE.")

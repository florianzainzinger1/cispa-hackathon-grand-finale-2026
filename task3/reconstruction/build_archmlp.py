"""Build a submission = current best (test_pib1.pt) with ONLY the 4 leaky-MLP
blocks {1,4,5,8} replaced by the DENOISED archetypal reconstruction. Everything
else (CNN + ViT) is byte-identical to pib1 -> a clean one-variable A/B.
Dedup any exact-duplicate row in a swapped block using the corresponding pib1 row
(a real face from the current best) so every model has 128 distinct images."""
import torch
import utils as U

PIB = U.work_path("outputs_final", "test_pib1.pt")
ARCH = U.work_path("outputs_final", "base_archetypal_denoised.pt")
OUT = U.work_path("outputs_final", "test_archmlp.pt")


def asblock(v):
    return (v["images"] if isinstance(v, dict) else v).float().clamp(0, 1)


def dedup(block, fallback):
    out = block.clone()
    seen = set()
    for i in range(out.shape[0]):
        key = out[i].reshape(-1).numpy().tobytes()
        if key in seen:
            out[i] = fallback[i].clone()
            key = out[i].reshape(-1).numpy().tobytes()
        seen.add(key)
    return out


pib = torch.load(PIB, weights_only=False, map_location="cpu")
arch = torch.load(ARCH, weights_only=False, map_location="cpu")

out = {}
swapped = []
for i in range(1, 13):
    m = f"model{i}"
    pb = asblock(pib[m])
    if m in arch:
        out[m] = dedup(arch[m].float().clamp(0, 1), pb)
        swapped.append(m)
    else:
        out[m] = pb

# hard validation (submission format)
for m, v in out.items():
    assert v.shape == (128, 3, 64, 64), (m, v.shape)
    assert torch.isfinite(v).all() and float(v.min()) >= 0 and float(v.max()) <= 1, m
    u = torch.unique(v.reshape(128, -1), dim=0).shape[0]
    assert u == 128, (m, "unique", u)

torch.save(out, OUT)
print("swapped MLP blocks:", swapped)
print("WROTE", OUT)
print("all 12 models: shape (128,3,64,64), [0,1], finite, 128 unique -> VALID")

"""Diagnostic: singular spectrum of the first-layer weight gradient gW = C@X_true.
Its effective rank bounds how many of the 128 images are linearly recoverable
from net.0. READ-ONLY."""
import numpy as np
import utils as U
import compare_fidelity as CF

print(f"{'model':7} {'E@64':>6} {'E@128':>6} {'E@256':>6} {'rank90':>7} {'v1_cap':>7}")
import json
res = []
for m in [1, 4, 5, 8]:
    name = f"model{m}"
    grad = U.load_grad(name)
    state = U.load_state(name)
    base = CF.first_image_linear(state, grad["gradients"])
    gW = grad["gradients"][base + ".weight"].float().numpy()
    s = np.linalg.svd(gW, compute_uv=False)
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    e64, e128, e256 = float(e[63]), float(e[127]), float(e[255])
    r90 = int(np.searchsorted(e, 0.90) + 1)
    v1 = CF.load_imgs(U.work_path("candidates", name, "selected_128.pt"))
    v1cap, _ = CF.captured(gW, v1)
    print(f"{name:7} {e64:6.3f} {e128:6.3f} {e256:6.3f} {r90:7} {v1cap:7.3f}")
    res.append({"model": name, "energy_top64": e64, "energy_top128": e128,
                "energy_top256": e256, "rank90": r90, "v1_captured": v1cap})
U.save_json(res, U.work_path("reports", "gw_spectrum.json"))
print("\nE@128 = fraction of gW energy in its top-128 singular directions = the")
print("ceiling any 128-image set can capture. If v1_cap ~ E@128, v1 is near-optimal")
print("and the rest of gW is unspannable noise (perturbed gradients).")

"""v2b: full-batch gradient-matching refinement for MLP models.

The first-layer analytic uses only gW[net.0]; the FULL gradient (net.0, net.2,
net.5) constrains the images further. Starting from the v1 reconstruction, we
optimize all 128 images so their full gradient matches the real one (cosine) +
TV prior. Measures span-capture before/after (objective, label-free).

Init from v1 selected_128 (READ-ONLY). Writes only candidates_v2/. GPU (sbatch).
"""
import os
import sys
import numpy as np
import torch

import utils as U
import attack_opt as AO
import infer_labels as IL
import compare_fidelity as CF

MLPS_DEFAULT = ["model8", "model1", "model4", "model5"]


def refine(name, iters=3000, lr=0.05, tv=1e-3, device="cpu", log_every=500):
    model, grad = AO.build_any(name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    name2param = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in name2param]
    real = AO.real_grad_vector(grad, keys).to(device)
    real_unit = real / (real.norm() + 1e-12)
    labels = IL.assign_labels(name, U.N_IMAGES).to(device)

    v1 = torch.load(U.work_path("candidates", name, "selected_128.pt"),
                    weights_only=False)["images"].float().clamp(0, 1)
    x = v1.clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    # objective fidelity of the starting point
    state = U.load_state(name)
    base = CF.first_image_linear(state, grad["gradients"])
    gW = grad["gradients"][base + ".weight"].float().numpy()
    cap0, _ = CF.captured(gW, v1.reshape(128, -1).numpy())

    for it in range(iters):
        opt.zero_grad()
        dummy = AO.compute_grad_vector(model, x.clamp(0, 1), labels, keys, name2param)
        cos = (dummy / (dummy.norm() + 1e-12) * real_unit).sum()
        loss = (1 - cos) + tv * U.total_variation(x.clamp(0, 1)).mean()
        gx = torch.autograd.grad(loss, x)[0]
        x.grad = gx
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  {name} it{it:5d} cos={cos.item():.4f}")

    ref = x.detach().clamp(0, 1).cpu()
    cap1, _ = CF.captured(gW, ref.reshape(128, -1).numpy())
    out_dir = U.work_path("candidates_v2", name)
    U.ensure_dir(out_dir)
    torch.save({"images": ref, "captured_before": cap0, "captured_after": cap1},
               os.path.join(out_dir, "mlp_refined_128.pt"))
    U.save_image_grid(ref, os.path.join(out_dir, "mlp_refined_grid.png"), ncol=16)
    comp = torch.cat([v1[:32], ref[:32]], 0)
    U.save_image_grid(comp, os.path.join(out_dir, "compare_v1_vs_refined.png"), ncol=16)
    print(f"{name}: captured v1={cap0:.4f} -> refined={cap1:.4f} "
          f"(delta {cap1-cap0:+.4f}) final_cos={cos.item():.4f}")
    return {"name": name, "captured_before": cap0, "captured_after": cap1,
            "final_cos": float(cos.item())}


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    iters = int(os.environ.get("FL_ITERS", "3000"))
    tv = float(os.environ.get("FL_TV", "1e-3"))
    lr = float(os.environ.get("FL_LR", "0.05"))
    print(f"device={device} iters={iters} tv={tv} lr={lr}")
    names = sys.argv[1:] or MLPS_DEFAULT
    res = []
    for name in names:
        try:
            res.append(refine(name, iters=iters, lr=lr, tv=tv, device=device))
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(res, U.work_path("reports", "v2b_refine_capture.json"))


if __name__ == "__main__":
    main()

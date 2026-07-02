"""Stage 4: optimization-based inversion (Inverting-Gradients / GradInversion).

Optimize 128 dummy images so their gradient matches the real one:
    loss = (1 - cos(flat(dummy_grad), flat(real_grad))) + tv * TV(images)
Cosine is scale-invariant, so the unknown loss reduction (sum vs mean) does not
matter. Labels come from infer_labels (fixed multiset). Primary use: ViT models
9/11; can also refine MLP/CNN from an analytic init.

GPU-heavy -> run via sbatch (FL_DEVICE=cuda). Full batch of 128 is required
because the real gradient aggregates all 128 samples.
"""
import os
import sys
import torch

import utils as U
import models as M
import vit as V
import infer_labels as IL


def build_any(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    fam = grad["family"]
    if fam == "vit":
        model, missing, unexpected = V.build_vit(name, state)
        if missing or unexpected:
            print(f"  [warn] {name} vit load missing={list(missing)[:4]} "
                  f"unexpected={list(unexpected)[:4]}")
    else:
        meta = {"family": fam, "activation": grad["activation"],
                "feature_shape": grad["feature_shape"]}
        model = M.build_model(name, meta, state)
    return model, grad


def real_grad_vector(grad, keys):
    return torch.cat([grad["gradients"][k].reshape(-1).float() for k in keys])


def compute_grad_vector(model, images, labels, keys, name2param):
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    params = [name2param[k] for k in keys]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.cat([g.reshape(-1) for g in grads])


def optimize(name, iters=4000, restarts=1, tv=1e-4, lr=0.1, device="cpu",
             init_images=None, seed=0, log_every=500):
    model, grad = build_any(name)
    model = model.to(device)
    # params MUST require grad: the dummy gradient is d(CE)/d(params). We never
    # step them (optimiser holds only the image), so they stay fixed.
    for p in model.parameters():
        p.requires_grad_(True)
    name2param = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in name2param]
    real = real_grad_vector(grad, keys).to(device)
    real_unit = real / (real.norm() + 1e-12)
    labels = IL.assign_labels(name, U.N_IMAGES).to(device)

    best = None
    best_loss = float("inf")
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        if init_images is not None:
            x = init_images.clone().to(device)
        else:
            x = torch.rand(U.N_IMAGES, 3, 64, 64, generator=g).to(device)
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad()
            dummy = compute_grad_vector(model, x.clamp(0, 1), labels, keys, name2param)
            dunit = dummy / (dummy.norm() + 1e-12)
            cos = (dunit * real_unit).sum()
            loss = (1 - cos) + tv * U.total_variation(x.clamp(0, 1)).mean()
            # gradient w.r.t. the image only (leave model params' .grad untouched)
            gx = torch.autograd.grad(loss, x)[0]
            x.grad = gx
            opt.step()
            sched.step()
            with torch.no_grad():
                x.clamp_(0, 1)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"  {name} r{r} it{it:5d} cos={cos.item():.4f} loss={loss.item():.4f}")
        fl = loss.item()
        if fl < best_loss:
            best_loss = fl
            best = x.detach().clamp(0, 1).cpu()
    out_dir = U.work_path("candidates", name)
    U.ensure_dir(out_dir)
    torch.save({"images": best, "source": "optimization", "best_loss": best_loss},
               os.path.join(out_dir, "opt_candidates.pt"))
    U.save_image_grid(best[:64], os.path.join(out_dir, "grid_opt.png"))
    print(f"{name}: opt done best_loss={best_loss:.4f} -> {out_dir}")
    return best


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    iters = int(os.environ.get("FL_ITERS", "4000"))
    restarts = int(os.environ.get("FL_RESTARTS", "1"))
    print("device:", device, "iters:", iters, "restarts:", restarts)
    names = sys.argv[1:] or ["model9", "model11"]
    for name in names:
        try:
            optimize(name, iters=iters, restarts=restarts, device=device)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

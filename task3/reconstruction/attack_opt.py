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


def load_bank(base_res, device):
    p = U.work_path("outputs", "face_bank.pt")
    if not os.path.exists(p):
        return None
    imgs = torch.load(p, weights_only=False)["images"].float()
    if base_res != 64:
        imgs = torch.nn.functional.interpolate(imgs, size=(base_res, base_res),
                                               mode="bilinear", align_corners=False)
    return imgs.clamp(0, 1).to(device)


def optimize(name, iters=4000, restarts=1, tv=1e-4, lr=0.1, device="cpu",
             base_res=64, init_mode="noise", seed=0, log_every=500):
    """Parametrise images at base_res (upsampled to 64 in the forward) to impose
    smoothness; init from noise or the face bank. best kept by cosine (=lowest
    grad-mismatch), independent of the TV term."""
    model, grad = build_any(name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    name2param = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in name2param]
    real = real_grad_vector(grad, keys).to(device)
    real_unit = real / (real.norm() + 1e-12)
    labels = IL.assign_labels(name, U.N_IMAGES).to(device)

    def up(z):
        if base_res == 64:
            return z
        return torch.nn.functional.interpolate(z, size=(64, 64), mode="bilinear",
                                               align_corners=False)

    bank = load_bank(base_res, device) if init_mode == "bank" else None
    best = None
    best_cos = -1.0
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        if bank is not None:
            perm = torch.randperm(bank.shape[0], generator=g)[:U.N_IMAGES]
            z = bank[perm].clone()
        else:
            z = torch.rand(U.N_IMAGES, 3, base_res, base_res, generator=g).to(device)
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad()
            x = up(z).clamp(0, 1)
            dummy = compute_grad_vector(model, x, labels, keys, name2param)
            dunit = dummy / (dummy.norm() + 1e-12)
            cos = (dunit * real_unit).sum()
            loss = (1 - cos) + tv * U.total_variation(x).mean()
            gz = torch.autograd.grad(loss, z)[0]
            z.grad = gz
            opt.step()
            sched.step()
            with torch.no_grad():
                z.clamp_(0, 1)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"  {name} r{r} it{it:5d} cos={cos.item():.4f} loss={loss.item():.4f}")
        with torch.no_grad():
            xf = up(z).clamp(0, 1)
        if cos.item() > best_cos:
            best_cos = cos.item()
            best = xf.detach().cpu()
    best_loss = 1 - best_cos
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
    base_res = int(os.environ.get("FL_BASE_RES", "64"))
    init_mode = os.environ.get("FL_INIT", "noise")
    tv = float(os.environ.get("FL_TV", "1e-4"))
    lr = float(os.environ.get("FL_LR", "0.1"))
    print(f"device={device} iters={iters} restarts={restarts} "
          f"base_res={base_res} init={init_mode} tv={tv} lr={lr}")
    names = sys.argv[1:] or ["model9", "model11"]
    for name in names:
        try:
            optimize(name, iters=iters, restarts=restarts, device=device,
                     base_res=base_res, init_mode=init_mode, tv=tv, lr=lr)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

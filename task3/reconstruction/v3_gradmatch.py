"""v3 — optimization-based reconstruction (gradient inversion), the DLG /
Inverting-Gradients family, warm-started from analytic candidates.

This is a DIFFERENT method family than v1 (analytic gradW/gradB ratio + KMeans
robust-mean) and v2 (confident single-sample clustering + v1 blend). Here we
REFINE a full batch of 128 images so the model gradient they induce matches the
observed client gradient, using Geiping et al. "Inverting Gradients":

    min_{x,y}  1 - cos( grad_theta CE(f(x), y), g_observed )  + tv * TV(x)
    x in [0,1]^{128x3x64x64},  optimized with SIGN(grad) Adam + LR decay.

Key choices (all load-bearing per the papers):
  * COSINE (magnitude-invariant) matching loss, not L2  (Geiping: +5 dB).
  * SIGNED-gradient Adam, not L-BFGS                     (Geiping: +15 dB).
  * WARM START from analytic candidates / face bank, not N(0,1) noise
    (fixes DLG's bad-init failure and the v1 ViT noise collapse).
  * LEARNABLE soft labels (DLG), so we don't need the unknown per-image
    label<->image pairing; initialized from the iDLG head-bias multiset.

Modes:
  diagnose  -> report warm-start gradient cosine (feasibility gate, cheap-ish)
  refine    -> full optimization; writes candidates_v3/<model>/refined.pt

GPU only for `refine` (second-order backprop). Device from FL_DEVICE.
"""
import os
import sys
import json
import time
import torch
import torch.nn.functional as F

import utils as U
import models as M
import infer_labels as IL

try:
    import vit as VIT
except Exception:
    VIT = None


# ----------------------------------------------------------------------------
# Model building (MLP / CNN / ViT) with the observed-gradient key alignment
# ----------------------------------------------------------------------------
def build_any(name):
    """Return (model, family, meta). Model is eval() with params frozen."""
    grad = U.load_grad(name)
    state = U.load_state(name)
    fam = grad.get("family")
    meta = {"family": fam, "activation": grad.get("activation"),
            "feature_shape": grad.get("feature_shape")}
    if fam == "vit":
        if VIT is None:
            raise RuntimeError("vit module unavailable")
        model, missing, unexpected = VIT.build_vit(name, state)
    else:
        model = M.build_model(name, meta, state)
    model.eval()
    # params MUST require grad so we can compute d CE / d params (the inner
    # gradient we match). They are never added to an optimizer, so they are
    # never updated — only the images/labels are optimized.
    for p in model.parameters():
        p.requires_grad_(True)
    return model, fam, meta, grad, state


def aligned_param_keys(model, gdict):
    """Parameter tensors (in model order) whose names exist in the observed
    gradient dict. Returns (names, params, obs_grads) all in the same order."""
    names, params, obs = [], [], []
    for pname, p in model.named_parameters():
        if pname in gdict and gdict[pname] is not None:
            names.append(pname)
            params.append(p)
            obs.append(gdict[pname].detach().float())
    return names, params, obs


def flat(vec_list):
    return torch.cat([v.reshape(-1) for v in vec_list])


# ----------------------------------------------------------------------------
# Gradient computation + cosine loss
# ----------------------------------------------------------------------------
def model_grad(model, params, images, label_logits, create_graph):
    """Mean-reduction CE gradient of the model w.r.t. `params`, evaluated at
    `images` with soft labels = softmax(label_logits). Returns list of grads."""
    logits = model(images)                                  # (N, n_classes)
    logp = F.log_softmax(logits, dim=1)
    y = F.softmax(label_logits, dim=1)                      # soft targets
    # cross-entropy with soft labels, mean over batch (matches observed grad)
    loss = -(y * logp).sum(dim=1).mean()
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return grads, float(loss.detach())


def cosine_distance(g_list, obs_flat, obs_norm):
    g = flat(g_list)
    num = (g * obs_flat).sum()
    den = g.norm() * obs_norm + 1e-12
    return 1.0 - num / den


# ----------------------------------------------------------------------------
# Warm-start loading
# ----------------------------------------------------------------------------
def load_warmstart(name, fam, device):
    """Return (images 128x3x64x64 in [0,1], source_str)."""
    # prefer v1's selected_128 (analytic/conv-invert); else conv-invert; else bank
    cand = U.work_path("candidates", name, "selected_128.pt")
    ci = U.work_path("candidates", name, "conv_invert_candidates.pt")
    ac = U.work_path("candidates", name, "analytic_candidates.pt")
    bank = U.work_path("outputs", "face_bank.pt")
    if fam != "vit" and os.path.exists(cand):
        imgs = torch.load(cand, weights_only=False)["images"].float()
        src = "selected_128"
    elif fam != "vit" and os.path.exists(ci):
        imgs = torch.load(ci, weights_only=False)["images"].float()
        src = "conv_invert"
    elif fam != "vit" and os.path.exists(ac):
        imgs = torch.load(ac, weights_only=False)["images"].float()
        src = "analytic"
    elif os.path.exists(bank):
        imgs = torch.load(bank, weights_only=False)["images"].float()
        src = "face_bank"
    else:
        imgs = torch.rand(U.N_IMAGES, 3, 64, 64)
        src = "random"
    # coerce to exactly 128
    if imgs.shape[0] >= U.N_IMAGES:
        imgs = imgs[:U.N_IMAGES]
    else:
        reps = (U.N_IMAGES + imgs.shape[0] - 1) // imgs.shape[0]
        imgs = imgs.repeat(reps, 1, 1, 1)[:U.N_IMAGES]
    return imgs.clamp(0, 1).to(device), src


def init_label_logits(name, device, scale=5.0):
    """Soft-label logits (128, n_classes) initialized from the iDLG multiset."""
    info = IL.infer(name, N=U.N_IMAGES)
    n_classes = int(info.get("n_classes", 200))
    labels = IL.assign_labels(name, N=U.N_IMAGES).to(device)   # (128,)
    logits = torch.zeros(U.N_IMAGES, n_classes, device=device)
    logits.scatter_(1, labels.view(-1, 1), scale)
    logits += 0.01 * torch.randn_like(logits)
    return logits


# ----------------------------------------------------------------------------
# Diagnose: warm-start cosine (feasibility gate)
# ----------------------------------------------------------------------------
def diagnose(name, device):
    model, fam, meta, grad, state = build_any(name)
    model = model.to(device)
    gdict = grad["gradients"]
    names, params, obs = aligned_param_keys(model, gdict)
    obs_flat = flat([o.to(device) for o in obs])
    obs_norm = obs_flat.norm()
    imgs, src = load_warmstart(name, fam, device)
    lab = init_label_logits(name, device)
    with torch.enable_grad():
        imgs_r = imgs.clone().requires_grad_(False)
        g_list, ce = model_grad(model, params, imgs_r, lab, create_graph=False)
        cos = 1.0 - float(cosine_distance(g_list, obs_flat, obs_norm))
    # per-layer cosine for insight
    per_layer = {}
    off = 0
    for nm, o in zip(names, obs):
        num = (g_list[names.index(nm)].reshape(-1).to(device) *
               o.reshape(-1).to(device)).sum()
        den = g_list[names.index(nm)].norm() * o.norm() + 1e-12
        per_layer[nm] = round(float(num / den), 4)
    info = {"name": name, "family": fam, "warmstart": src,
            "n_matched_params": len(names), "obs_norm": float(obs_norm),
            "ce_loss": round(ce, 4), "warmstart_cosine": round(cos, 4),
            "per_layer_cosine": per_layer}
    print(json.dumps(info, indent=2))
    U.save_json(info, U.work_path("candidates_v3", name, "diagnose.json"))
    return info


# ----------------------------------------------------------------------------
# Refine: full gradient-matching optimization
# ----------------------------------------------------------------------------
def refine(name, device, steps=1200, lr=0.1, tv=1e-2, label_lr=0.1,
           opt_labels=True, sign=True, log_every=100):
    t0 = time.time()
    model, fam, meta, grad, state = build_any(name)
    model = model.to(device)
    gdict = grad["gradients"]
    names, params, obs = aligned_param_keys(model, gdict)
    obs_flat = flat([o.to(device) for o in obs])
    obs_norm = obs_flat.norm()

    imgs0, src = load_warmstart(name, fam, device)
    x = imgs0.clone().requires_grad_(True)
    lab = init_label_logits(name, device)
    lab.requires_grad_(opt_labels)

    par = [x] + ([lab] if opt_labels else [])
    opt = torch.optim.Adam(par, lr=lr)
    # LR decay at 3/8, 5/8, 7/8 (Geiping)
    milestones = [int(steps * m) for m in (3 / 8, 5 / 8, 7 / 8)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)

    def cos_now():
        # enable_grad: autograd.grad needs a live graph from params->CE loss.
        # create_graph=False keeps it first-order (cheap, no retained graph).
        with torch.enable_grad():
            g_list, _ = model_grad(model, params, x.detach(), lab.detach(),
                                   create_graph=False)
            return 1.0 - float(cosine_distance(g_list, obs_flat, obs_norm))

    cos_start = cos_now()
    best_cos, best_x = cos_start, x.detach().clone()
    for it in range(steps):
        opt.zero_grad()
        g_list, ce = model_grad(model, params, x, lab, create_graph=True)
        loss = cosine_distance(g_list, obs_flat, obs_norm) + tv * U.total_variation(x).mean()
        loss.backward()
        # signed-gradient Adam (Geiping): replace grad by its sign. Optional,
        # because from a GOOD warm start signing can erode already-clean pixels.
        if sign:
            with torch.no_grad():
                for p in par:
                    if p.grad is not None:
                        p.grad.sign_()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if (it % log_every == 0) or (it == steps - 1):
            c = cos_now()
            if c > best_cos:
                best_cos, best_x = c, x.detach().clone()
            print(f"  {name} step {it:4d} cos={c:.4f} ce={ce:.3f} "
                  f"lr={sched.get_last_lr()[0]:.3g}", flush=True)
    final_cos = cos_now()
    if final_cos > best_cos:
        best_cos, best_x = final_cos, x.detach().clone()

    out = best_x.clamp(0, 1).cpu()
    out_dir = U.work_path("candidates_v3", name)
    U.ensure_dir(out_dir)
    torch.save({"images": out, "cos_start": cos_start, "cos_final": best_cos,
                "source": src, "steps": steps},
               os.path.join(out_dir, "refined.pt"))
    U.save_image_grid(out[:64], os.path.join(out_dir, "grid_refined.png"))
    meta_out = {"name": name, "family": fam, "warmstart": src,
                "cos_start": round(cos_start, 4), "cos_final": round(best_cos, 4),
                "delta": round(best_cos - cos_start, 4),
                "steps": steps, "lr": lr, "tv": tv,
                "seconds": round(time.time() - t0, 1)}
    U.save_json(meta_out, os.path.join(out_dir, "refined_meta.json"))
    print(json.dumps(meta_out, indent=2), flush=True)
    return meta_out


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    mode = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    names = sys.argv[2:] or U.MODELS
    steps = int(os.environ.get("FL_ITERS", "1200"))
    lr = float(os.environ.get("FL_LR", "0.1"))
    tv = float(os.environ.get("FL_TV", "0.01"))
    sign = os.environ.get("FL_SIGN", "1") != "0"
    print(f"device={device} mode={mode} steps={steps} lr={lr} tv={tv} sign={sign}")
    for name in names:
        try:
            if mode == "diagnose":
                diagnose(name, device)
            elif mode == "refine":
                refine(name, device, steps=steps, lr=lr, tv=tv, sign=sign)
            else:
                print("unknown mode", mode); sys.exit(2)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

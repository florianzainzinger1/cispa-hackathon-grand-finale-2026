"""Agent G — decisive ViT identifiability probe (model9, model11).

Question: does the ViT's observed gradient carry recoverable IDENTITY, or is it
so batch-averaged that any plausible face set explains it equally? If a prior
can't beat 'random faces' by much on replay cosine, no generative prior will
beat the bank on SSIM (it would only produce realistic-but-wrong faces, the
confirmed leaderboard-loser).

For each ViT we compare replay cosine (aggregate gradient vs observed) of:
  * the current-best bank recon,
  * the r2 StyleGAN latent-matched set (if present),
  * K sets of RANDOM CelebA-HQ StyleGAN faces (label multiset from infer_labels).
If random faces already reach a cosine comparable to the optimized sets, the
gradient is identity-poor -> DEFER.

Also reports how sharply the gradient discriminates: cos(true-ish) - cos(random).
Cheap (ViTs are 2.7M / 6.5M params). CPU-friendly.
"""
import os
import torch
import torch.nn.functional as F

import utils as U
import attack_opt as AO
import infer_labels as IL
import gan_prior as GP


def ctx(name, device):
    model, grad = AO.build_any(name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    n2p = dict(model.named_parameters())
    keys = [k for k in grad["gradients"].keys() if k in n2p]
    real = AO.real_grad_vector(grad, keys).to(device)
    ru = real / (real.norm() + 1e-12)
    return model, grad, keys, n2p, ru


def rcos(model, imgs, labels, keys, n2p, ru, device):
    logits = model(imgs.to(device))
    loss = F.cross_entropy(logits, labels.to(device))
    g = torch.autograd.grad(loss, [n2p[k] for k in keys])
    vec = torch.cat([x.reshape(-1) for x in g])
    return float((vec / (vec.norm() + 1e-12) * ru).sum())


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    prior = None
    for name in ["model9", "model11"]:
        print(f"\n===== {name} =====", flush=True)
        model, grad, keys, n2p, ru = ctx(name, device)
        labels = IL.assign_labels(name, U.N_IMAGES).to(device)

        # current best
        cb = torch.load(U.work_path("outputs_final", "submission_current_best.pt"),
                        weights_only=False)[name]
        cb = (cb["images"] if isinstance(cb, dict) else cb).float().clamp(0, 1)
        c_cb = rcos(model, cb, labels, keys, n2p, ru, device)
        print(f"  current_best   replay_cos = {c_cb:+.4f}", flush=True)

        # r2 latent
        for sub in ["r2", "r1"]:
            p = U.work_path("runs", sub, name, "selected_128.pt")
            if os.path.exists(p):
                im = torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
                print(f"  {sub}_latent     replay_cos = {rcos(model, im, labels, keys, n2p, ru, device):+.4f}", flush=True)
                break

        # random StyleGAN faces (3 seeds)
        if prior is None:
            G = GP.load_generator(device=device)
            prior = GP.StyleGANPrior(G, out_size=64, w_plus=True).to(device)
        rand_cos = []
        for s in range(3):
            torch.manual_seed(100 + s)
            faces = prior.sample(U.N_IMAGES, device, truncation=0.7).float().clamp(0, 1)
            rand_cos.append(rcos(model, faces, labels, keys, n2p, ru, device))
        rc = sum(rand_cos) / len(rand_cos)
        print(f"  random_faces   replay_cos = {rc:+.4f}  (seeds {[round(x,3) for x in rand_cos]})", flush=True)
        print(f"  --> discrimination current_best - random = {c_cb - rc:+.4f}", flush=True)


if __name__ == "__main__":
    main()

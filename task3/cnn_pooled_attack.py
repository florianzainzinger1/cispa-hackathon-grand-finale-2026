"""Analytic-readout + conv/pool inversion for the POOLED CNN models (3,7).

conv(3->8,3x3,pad1) -> act -> pool(k) -> flatten -> fc1 -> act -> head, with
pool factor k = 64 / feat_H  (k=8 for model3's (8,8,8); k=4 for model7's (8,16,16)).

fc1 analytic readout (clean for relu) gives the POOLED feature map. We recover the
image by optimizing x in [0,1] so that pool(relu(conv(x))) matches it, with TV +
chroma priors. Heavy compression -> low-res-but-real faces (upsampled implicitly).

Run: ./.venv/bin/python cnn_pooled_attack.py [id]
"""
import sys
import torch
import torch.nn.functional as F
from common import load_grad, load_state, dedup, BATCH, IMG_SHAPE

POOLED_CNN = [3, 7]


def featureness(feats, act):
    f = feats.reshape(feats.shape[0], -1)
    bad = torch.clamp(-f, min=0) if act != "sigmoid" else \
        torch.clamp(-f, min=0) + torch.clamp(f - 1, min=0)
    return -bad.mean(dim=1)


def analytic_features(g, bias_eps=1e-8):
    C, H, W = g["feature_shape"]
    Wt = g["gradients"]["fc1.weight"]
    b = g["gradients"]["fc1.bias"]
    good = b.abs() > bias_eps
    feat = (Wt[good] / b[good].unsqueeze(1)).reshape(-1, C, H, W)
    return feat, featureness(feat, g["activation"])


def invert(feats, w, bias, act, pool_k, pool="avg",
           iters=1500, lr=0.1, tv=8e-3, chroma=0.05, dev="cpu"):
    # Round-trip-validated (2026-07-03): vs the old gray-init/tv1e-3 defaults, a
    # low-frequency UPSAMPLE INIT + stronger TV recover ~+0.12 SSIM on k=4 (model
    # 7/2) and ~+0.05 on k=8 (model3). The pooled->image map is 6-24x under-
    # determined, so the recoverable content is low-freq: seed it and smooth it.
    feats = feats.to(dev)
    poolfn = (F.avg_pool2d if pool == "avg" else F.max_pool2d)

    def fwd(x):
        y = F.conv2d(x, w, bias, padding=1)
        y = F.relu(y) if act == "relu" else (torch.tanh(y) if act == "tanh"
                                             else torch.sigmoid(y))
        return poolfn(y, pool_k)

    # init = per-image bilinear upsample of the pooled channel-mean (a luminance
    # guess), min-max normalized into a mild [0.2,0.8] band, replicated to RGB.
    gm = feats.mean(1, keepdim=True)                                  # (n,1,H,W)
    up = F.interpolate(gm, size=(64, 64), mode="bilinear", align_corners=False)
    up = (up - up.amin()) / (up.amax() - up.amin() + 1e-6)
    x = (up.repeat(1, 3, 1, 1) * 0.6 + 0.2).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for t in range(iters):
        opt.zero_grad()
        xb = x.clamp(0, 1)
        loss = ((fwd(xb) - feats) ** 2).mean()
        loss = loss + tv * ((xb[:, :, 1:] - xb[:, :, :-1]).abs().mean()
                            + (xb[:, :, :, 1:] - xb[:, :, :, :-1]).abs().mean())
        loss = loss + chroma * xb.var(dim=1).mean()
        loss.backward()
        opt.step()
    return x.detach().clamp(0, 1).cpu()


def reconstruct(i, pool="avg", dev="cpu"):
    g = load_grad(i)
    st = load_state(i)
    C, H, W = g["feature_shape"]
    feats, fs = analytic_features(g)
    chosen = dedup(feats, fs, k=BATCH, sim_thresh=0.05)
    feats = feats[chosen]
    imgs = invert(feats, st["conv.weight"].to(dev), st["conv.bias"].to(dev),
                  g["activation"], 64 // H, pool=pool, dev=dev)
    if imgs.shape[0] < BATCH:
        imgs = torch.cat([imgs, imgs[:1].repeat(BATCH - imgs.shape[0], 1, 1, 1)], 0)
    return imgs[:BATCH].float().contiguous()


if __name__ == "__main__":
    import os
    # usage:  cnn_pooled_attack.py [id ...] [--save]
    #   no --save -> just print per-model diagnostics
    #   --save    -> also write the improved base recon to recon/model{i}.pt
    #                (this is the +0.05..0.12-SSIM base the metric/postproc track
    #                 runs on top of; back up the old recon first if you want an A/B)
    args = [a for a in sys.argv[1:] if a != "--save"]
    save = "--save" in sys.argv
    ids = [int(a) for a in args] if args else POOLED_CNN
    from common import HERE
    os.makedirs(os.path.join(HERE, "recon"), exist_ok=True)
    for i in ids:
        g = load_grad(i)
        _, fs = analytic_features(g)
        print(f"model{i}: feat={g['feature_shape']} pool_k={64//g['feature_shape'][1]} "
              f"clean={(fs>-0.01).sum().item()}")
        if save:
            out = reconstruct(i)
            path = os.path.join(HERE, "recon", f"model{i}.pt")
            torch.save(out, path)
            print(f"  saved {path}  {tuple(out.shape)}")

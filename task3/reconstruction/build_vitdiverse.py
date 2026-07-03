"""ViT diverse in-distribution gallery on the pib2 base.

Bet: ViT 9/11 carry no per-sample identity, so their 128 slots are pure filler.
Under Hungarian one-to-one SSIM, a DIVERSE set of distinct in-distribution faces
covers more GT faces than a smoothed blend (pib2's ViT blocks are TV=0.028 -> a
de-diversified blend). This is the opposite bet, aligned with the diversity law.

Single variable isolated: we brightness-match the diverse gallery to pib2's ViT
per-channel mean (so luminance isn't a confound); everything else = pib2.
"""
import torch
import utils as U

GAL = U.work_path("gallery", "celeba_top200_cc178.pt")   # 12048 x 3x64x64, [0,255]
PIB2 = U.work_path("outputs_final", "test_pib2.pt")
OUT = U.work_path("outputs_final", "test_vitdiv.pt")
VIT = ["model9", "model11"]


def load_gallery():
    G = torch.load(GAL, weights_only=False, map_location="cpu")
    if isinstance(G, dict):
        G = next(v for v in G.values() if torch.is_tensor(v) and v.ndim == 4)
    return (G.float() / 255.0).clamp(0, 1)


def gray16(x):
    g = x.mean(1, keepdim=True)
    g = torch.nn.functional.interpolate(g, size=(16, 16), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1)


def farthest_point(feats, k, seed=0):
    n = feats.shape[0]
    g = torch.Generator().manual_seed(seed)
    start = int(torch.randint(0, n, (1,), generator=g))
    sel = [start]
    mind = ((feats - feats[start]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, ((feats - feats[i]) ** 2).sum(1))
    return torch.tensor(sel, dtype=torch.long)


def block(d, m):
    e = d[m]
    return (e["images"] if isinstance(e, dict) else e).float().clamp(0, 1)


def main():
    gal = load_gallery()
    feats = gray16(gal)
    sel = farthest_point(feats, 128, seed=0)
    diverse = gal[sel].clone()                      # (128,3,64,64), diverse real faces

    pib2 = torch.load(PIB2, weights_only=False, map_location="cpu")
    # brightness match: shift diverse per-channel mean to pib2's ViT target
    tgt = block(pib2, "model9").mean(dim=(0, 2, 3))          # (3,)
    cur = diverse.mean(dim=(0, 2, 3))
    diverse = (diverse + (tgt - cur).view(1, 3, 1, 1)).clamp(0, 1)
    print(f"diverse gallery: mean_rgb {[round(x,3) for x in diverse.mean(dim=(0,2,3)).tolist()]} "
          f"(target {[round(x,3) for x in tgt.tolist()]}), unique="
          f"{torch.unique(diverse.reshape(128,-1),dim=0).shape[0]}, "
          f"tv={((diverse[:,:,1:]-diverse[:,:,:-1]).abs().mean()+(diverse[:,:,:,1:]-diverse[:,:,:,:-1]).abs().mean()):.3f}",
          flush=True)

    out = {}
    for i in range(1, 13):
        m = f"model{i}"
        out[m] = diverse.clone() if m in VIT else block(pib2, m)

    for m, v in out.items():
        assert v.shape == (128, 3, 64, 64) and torch.isfinite(v).all()
        assert float(v.min()) >= 0 and float(v.max()) <= 1
        assert torch.unique(v.reshape(128, -1), dim=0).shape[0] == 128, (m, "not 128 unique")
    torch.save(out, OUT)
    U.save_image_grid(diverse, U.work_path("candidates_arch", "vit_diverse_grid.png"), ncol=16)
    print("WROTE", OUT, "(models 9,11 = diverse gallery; other 10 = pib2)", flush=True)


if __name__ == "__main__":
    main()

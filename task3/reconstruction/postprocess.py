"""Stage 5: turn each model's candidate bank into exactly 128 diverse images.

Because each of the ~1024 neurons reconstructs one of the 128 private images,
candidates cluster around the true images. We cluster into 128 groups and take
the best-quality member of each - this maximises diversity (needed for the
Hungarian one-to-one SSIM matching) while keeping quality.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import utils as U


def load_best_candidates(name):
    """Prefer conv-inverted images (CNN), else analytic direct/feature images."""
    d = U.work_path("candidates", name)
    ci = os.path.join(d, "conv_invert_candidates.pt")
    ac = os.path.join(d, "analytic_candidates.pt")
    if os.path.exists(ci):
        blob = torch.load(ci, weights_only=False)
        return blob["images"].float(), "conv_invert"
    if os.path.exists(ac):
        blob = torch.load(ac, weights_only=False)
        return blob["images"].float(), "analytic"
    return None, None


def quality_score(imgs):
    """Higher = cleaner, more face-like. Rewards in-range pixels and moderate
    contrast, strongly penalises high-frequency noise (TV). Raw std is NOT
    rewarded because pixel noise inflates it."""
    st = U.image_stats(imgs)
    contrast = torch.clamp(st["std"] / 0.25, max=1.0)   # saturates; not unbounded
    q = st["rng_frac"] + 0.3 * contrast - 2.0 * st["tv"]
    q = torch.where(st["finite"], q, torch.full_like(q, -1e9))
    return q


def feat_vectors(imgs, size=24):
    """Downsampled grayscale vectors for clustering/dedup."""
    g = imgs.mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1).cpu().numpy()


def select_128(imgs, n=128, seed=0, denoise=True):
    """Cluster into n groups; each cluster yields one representative (robust
    quality-weighted mean when denoise=True, else the best-quality member).
    Falls back to quality-ranked fill if the pool is smaller than n."""
    from sklearn.cluster import KMeans
    q = quality_score(imgs)
    N = imgs.shape[0]
    if N <= n:
        # not enough: take all, then pad with slightly-noised copies of the best
        order = torch.argsort(q, descending=True)
        sel = list(order.tolist())
        i = 0
        while len(sel) < n:
            base = imgs[order[i % order.numel()]]
            sel.append(("noise", base))
            i += 1
        out = []
        for s in sel[:n]:
            if isinstance(s, tuple):
                out.append((s[1] + 0.02 * torch.randn_like(s[1])).clamp(0, 1))
            else:
                out.append(imgs[s])
        return torch.stack(out), list(range(n))

    X = feat_vectors(imgs)
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(X)
    labels = km.labels_
    Xt = torch.from_numpy(X)
    reps, chosen = [], []
    for c in range(n):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        idx_t = torch.from_numpy(idx)
        qc = q[idx_t]
        best_local = int(idx[int(torch.argmax(qc))])
        if denoise and len(idx) >= 2:
            # robust mean of members close to the highest-quality member
            anchor = Xt[best_local]
            d = (Xt[idx_t] - anchor).pow(2).sum(dim=1).sqrt()
            thr = torch.quantile(d, 0.5).clamp_min(1e-6)
            keep = idx_t[d <= thr]
            w = torch.clamp(q[keep], min=0) + 1e-3
            rep = (imgs[keep] * w.view(-1, 1, 1, 1)).sum(0) / w.sum()
            reps.append(rep.clamp(0, 1))
        else:
            reps.append(imgs[best_local].clamp(0, 1))
        chosen.append(best_local)
    # fill empty clusters from highest-quality not-yet-used
    if len(reps) < n:
        order = torch.argsort(q, descending=True).tolist()
        used = set(chosen)
        for i in order:
            if i not in used:
                reps.append(imgs[i].clamp(0, 1)); chosen.append(i); used.add(i)
            if len(reps) == n:
                break
    reps = torch.stack(reps[:n])
    return reps, chosen[:n]


def run_model(name, n=128):
    imgs, src = load_best_candidates(name)
    if imgs is None:
        print(f"{name}: no candidates")
        return None
    sel, idx = select_128(imgs, n=n)
    out_dir = U.work_path("candidates", name)
    torch.save({"images": sel, "source": src},
               os.path.join(out_dir, "selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_selected.png"), ncol=16)
    print(f"{name}: source={src} pool={imgs.shape[0]} -> selected {sel.shape[0]}")
    return sel


def main():
    names = sys.argv[1:] or U.MODELS
    for name in names:
        try:
            run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

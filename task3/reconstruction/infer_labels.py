"""Stage 3: iDLG/GradInversion-style label restoration from the classifier head.

For softmax cross-entropy, the head-bias gradient is
    gb[c] = (1/N) * sum_n (p_{n,c} - 1[y_n = c]).
Classes truly present get a negative push ~ -count_c/N, so the per-class image
count is estimated as  count_c ~= round(-N * gb[c]).  This is approximate for
trained models but gives a strong label multiset prior for optimization.
"""
import os
import sys
import torch

import utils as U


def head_keys(state, g):
    """Return (weight_key, bias_key) of the last linear (classifier head)."""
    last = None
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g:
                last = base
    return (last + ".weight", last + ".bias") if last else (None, None)


def infer(name, N=128):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    wkey, bkey = head_keys(state, g)
    if bkey is None or g.get(bkey) is None:
        return {"name": name, "confidence": "none"}
    gb = g[bkey].float()
    n_classes = gb.numel()
    # counts from negative bias gradient
    raw = -N * gb
    counts = raw.clamp(min=0).round().to(torch.int64)
    # normalise total to N by trimming/greedy-adding on the largest fractionals
    total = int(counts.sum())
    present = (counts > 0).sum().item()
    # label multiset (class ids repeated by count)
    labels = torch.repeat_interleave(torch.arange(n_classes), counts)
    conf = "high" if abs(total - N) <= max(4, N // 20) else "low"
    return {
        "name": name, "n_classes": n_classes,
        "est_total": total, "target": N, "present_classes": int(present),
        "counts_sum": total, "confidence": conf,
        "top_classes": torch.argsort(counts, descending=True)[:20].tolist(),
        "top_counts": torch.sort(counts, descending=True).values[:20].tolist(),
        "labels_multiset": labels.tolist(),
    }


def assign_labels(name, N=128):
    """Return a length-N label tensor for optimization (best-effort)."""
    info = infer(name, N)
    labels = torch.tensor(info.get("labels_multiset", []), dtype=torch.long)
    if labels.numel() >= N:
        return labels[:N]
    # pad with the most frequent present classes (or 0)
    pad_cls = info.get("top_classes", [0]) or [0]
    pad = torch.tensor([pad_cls[0]] * (N - labels.numel()), dtype=torch.long)
    return torch.cat([labels, pad])[:N]


def main():
    names = sys.argv[1:] or U.MODELS
    out = {}
    for name in names:
        info = infer(name)
        out[name] = info
        print(f"{name}: present={info.get('present_classes')} "
              f"est_total={info.get('est_total')}/{info.get('target')} "
              f"conf={info.get('confidence')}")
    U.save_json(out, U.work_path("reports", "label_hints.json"))
    print("wrote", U.work_path("reports", "label_hints.json"))


if __name__ == "__main__":
    main()

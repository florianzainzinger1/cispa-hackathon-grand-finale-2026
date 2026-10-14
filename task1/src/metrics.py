"""The judge: pooled TPR @ target FPR.

Every idea in this project is measured by tpr_at_fpr on validation.
Definition (conservative, no interpolation): the maximum TPR over all score
thresholds whose FPR is <= target_fpr, with all tokens of the evaluated
document set pooled into one ranking.
"""

from __future__ import annotations

import numpy as np


def tpr_at_fpr(
    y_true,
    scores,
    target_fpr: float = 0.001,
) -> float:
    """Pooled TPR at FPR <= target_fpr (no interpolation).

    y_true: array-like of {0,1}
    scores: array-like of floats, higher = more likely watermarked
    """
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both positive and negative tokens")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = np.cumsum(y_sorted, dtype=np.int64)
    fp = np.cumsum(1 - y_sorted, dtype=np.int64)

    # collapse tied scores: only the last index of each distinct score is a
    # valid operating point (a threshold cannot split equal scores)
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tpr = tp[distinct] / n_pos
    fpr = fp[distinct] / n_neg

    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def pool_documents(rows, scores_by_doc) -> tuple[np.ndarray, np.ndarray]:
    """Pool labeled rows + {document_id: scores} into flat (y_true, scores)."""
    ys, ss = [], []
    for row in rows:
        doc_id = row["document_id"]
        if doc_id not in scores_by_doc:
            raise KeyError(f"missing scores for {doc_id}")
        sc = scores_by_doc[doc_id]
        if len(sc) != len(row["labels"]):
            raise ValueError(f"{doc_id}: score length mismatch")
        ys.extend(row["labels"])
        ss.extend(sc)
    return np.asarray(ys, dtype=np.int8), np.asarray(ss, dtype=np.float64)


def _self_test() -> None:
    """Verify against sklearn on random data + hand-checkable cases."""
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(0)

    # hand-checkable: perfect separation
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.001) == 1.0
    # hand-checkable: perfectly wrong -> 0 at tiny FPR
    assert tpr_at_fpr([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.001) == 0.0
    # ties: threshold cannot split equal scores
    assert tpr_at_fpr([0, 1, 1], [0.5, 0.5, 0.9], 0.4) == 0.5

    for trial in range(20):
        n = 200_000
        y = (rng.random(n) < 0.47).astype(np.int8)
        s = rng.normal(0, 1, n) + 2.0 * y
        s = np.round(s, 2)  # force many ties
        for target in (0.001, 0.01, 0.1):
            got = tpr_at_fpr(y, s, target)
            fpr, tpr, _ = roc_curve(y, s)
            want = float(tpr[fpr <= target].max())
            assert abs(got - want) < 1e-12, (trial, target, got, want)
    print("metrics self-test passed (vs sklearn.roc_curve, 20 trials + edge cases)")


if __name__ == "__main__":
    _self_test()

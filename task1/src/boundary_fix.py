"""Boundary-bleed fix: two-sided evidence requirement.

92% of the 0.1%-FPR clean tail sits within 10 tokens of a true span =
posterior smoothing bleed. An in-span token has wm evidence on BOTH sides;
a bleed token only on ONE. Compute forward-filtered and backward-filtered
clean-posteriors separately; combine with min / blends; measure val TPR.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, emission_loglik  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
y = va["label"].to_numpy()

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.02, 45.0)


def filtered_scores(ll: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    """Forward-FILTERED -log10 P(Clean | obs_1..t) per token."""
    T, S = ll.shape
    out = np.zeros(T)
    alpha = logpi + ll[0]
    a = alpha - np.logaddexp.reduce(alpha)
    out[0] = -a[0] / np.log(10)
    for t in range(1, T):
        m = alpha[:, None] + logA
        alpha = ll[t] + np.logaddexp.reduce(m, axis=0)
        a = alpha - np.logaddexp.reduce(alpha)
        out[t] = -a[0] / np.log(10)
    return out


def full_fb_scores(ll, logA, logpi):
    T, S = ll.shape
    alpha = np.zeros((T, S)); beta = np.zeros((T, S))
    alpha[0] = logpi + ll[0]
    for t in range(1, T):
        alpha[t] = ll[t] + np.logaddexp.reduce(alpha[t - 1][:, None] + logA, axis=0)
    for t in range(T - 2, -1, -1):
        beta[t] = np.logaddexp.reduce(logA + (ll[t + 1] + beta[t + 1])[None, :], axis=1)
    post = alpha + beta
    post -= np.logaddexp.reduce(post, axis=1, keepdims=True)
    p_clean = np.clip(np.exp(post[:, 0]), 1e-30, 1.0)
    return -np.log10(p_clean)


ll_all = emission_loglik(va, params)
docs = va["document_id"].to_numpy()

s_full = np.zeros(len(va)); s_fwd = np.zeros(len(va)); s_bwd = np.zeros(len(va))
start = 0
for i in range(1, len(va) + 1):
    if i == len(va) or docs[i] != docs[start]:
        ll = ll_all[start:i]
        s_full[start:i] = full_fb_scores(ll, logA, logpi)
        s_fwd[start:i] = filtered_scores(ll, logA, logpi)
        s_bwd[start:i] = filtered_scores(ll[::-1], logA, logpi)[::-1]
        start = i

variants = {
    "full FB (baseline)": s_full,
    "min(fwd,bwd)": np.minimum(s_fwd, s_bwd),
    "min(full, fwd+bwd)": np.minimum(s_full, s_fwd + s_bwd),
    "geomean fwd,bwd": np.sqrt(s_fwd * s_bwd),
    "min(full, 2*min(f,b)+1)": np.minimum(s_full, 2 * np.minimum(s_fwd, s_bwd) + 1.0),
    "full * sig two-sided": s_full * np.minimum(1.0, (np.minimum(s_fwd, s_bwd) + 0.3)),
    "min(full, max(f,b))": np.minimum(s_full, np.maximum(s_fwd, s_bwd)),
}
print(f"{'variant':28s} {'T@0.1%':>8s} {'T@1%':>8s}")
for name, s in variants.items():
    print(f"{name:28s} {tpr_at_fpr(y, s, 0.001):8.4f} {tpr_at_fpr(y, s, 0.01):8.4f}")

# where does min(f,b) lose/gain? check tail composition after fix
best = np.minimum(s_fwd, s_bwd)
clean = y == 0
tau_new = np.quantile(best[clean], 1 - 0.001)
ti = np.where(clean & (best >= tau_new))[0]
near = 0
W = 10
lab = y
for d in np.unique(docs[ti]):
    m = docs == d
    idx = np.where(m)[0]
    conv = np.convolve(lab[idx], np.ones(2 * W + 1), mode="same") > 0
    for j in idx:
        if best[j] >= tau_new and clean[j] and conv[np.where(idx == j)[0][0]]:
            near += 1
print(f"\nmin(f,b): clean-tail tokens near true spans: {near}/{len(ti)}")

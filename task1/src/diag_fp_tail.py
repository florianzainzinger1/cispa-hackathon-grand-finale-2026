"""Diagnose the FP tail: WHICH clean tokens set the 0.1%-FPR threshold?

The global tau is set by the top ~0.1% of clean tokens (~51 of 51379 on val).
If those are dominated by repeated contexts / few documents / specific token
types, a deterministic tail-cleaner (signature filtering) has real headroom.
If they are diverse ambiguous text, the tail is irreducible (confirms the
earlier tail-veto finding).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
y = va["label"].to_numpy()

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.02, 45.0)
sc = hmm_scores(va, params, logA, logpi)

clean = y == 0
tau = np.quantile(sc[clean], 1 - 0.001)
n_tail = int((clean & (sc >= tau)).sum())
print(f"tau={tau:.3f}, clean tail tokens >= tau: {n_tail}")

tail = clean & (sc >= tau)
ti = np.where(tail)[0]

# 1. document concentration
docs = va["document_id"].to_numpy()
doc_counts = pd.Series(docs[ti]).value_counts()
print(f"\ndocuments involved: {len(doc_counts)} (of {pd.Series(docs[clean]).nunique()} clean-containing docs)")
print("top docs:\n", doc_counts.head(8).to_string())

# 2. contiguity: are tail tokens contiguous runs (few FP 'spans') or isolated?
runs = 1
for a, b in zip(ti[:-1], ti[1:]):
    if b != a + 1 or docs[a] != docs[b]:
        runs += 1
print(f"\ncontiguous runs: {runs} (mean run length {len(ti)/runs:.1f})")

# 3. dedup flags: did these tokens carry gumbel/ts/kgw evidence despite being repeats?
for col in ["gumbel_first", "textseal_first", "kgw_first", "is_repeated_ngram_2",
            "is_repeated_ngram_3", "is_repeated_ngram_4", "is_special_token",
            "is_number_like", "is_newline_like"]:
    v = va[col].to_numpy()[ti]
    print(f"  {col:22s} mean={np.nanmean(v.astype(float)):.3f}   (clean baseline {np.nanmean(va[col].to_numpy()[clean].astype(float)):.3f})")

# 4. signal values on tail tokens: which scheme drives the false evidence?
print("\nsignal means on FP-tail vs all-clean:")
for col in ["gumbel_score", "textseal_score", "kgw_green", "unigram_green", "qwen_entropy", "qwen_logprob"]:
    v = va[col].to_numpy().astype(float)
    print(f"  {col:15s} tail={np.nanmean(v[ti]):.3f}   clean-all={np.nanmean(v[clean]):.3f}")

# 5. label of NEIGHBORING tokens: are these adjacent to true wm spans (boundary bleed)?
near_wm = np.zeros(len(va), bool)
W = 10
lab = y.copy()
for d in np.unique(docs[ti]):
    m = docs == d
    idx = np.where(m)[0]
    ywin = lab[idx]
    conv = np.convolve(ywin, np.ones(2 * W + 1), mode="same") > 0
    near_wm[idx] = conv
frac_near = near_wm[ti].mean()
print(f"\nfraction of FP-tail tokens within {W} tokens of a TRUE wm span: {frac_near:.3f}")

# 6. how much would removing the tail help? simulate perfect removal of top-k FP docs
print("\nceiling if the top-k FP documents' clean tails were fully cleaned:")
for k in (1, 2, 3, 5):
    bad_docs = set(doc_counts.head(k).index)
    sc2 = sc.copy()
    kill = clean & np.isin(docs, list(bad_docs))
    sc2[kill] = 0.0
    print(f"  top-{k} docs cleaned: TPR@0.1% {tpr_at_fpr(y, sc2, 0.001):.4f}  (base {tpr_at_fpr(y, sc, 0.001):.4f})")

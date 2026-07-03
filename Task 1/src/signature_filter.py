"""Signature filtering (Hong et al. 2026, hackathon version) with split-half.

Learn 2-/3-gram 'signature' token patterns overrepresented in clean FP-tail
windows on train half A; mask their emissions on half B; keep ONLY if both
directions transfer positively. Runs on the corrected v2 cache.

Reject rule (pre-registered): if either direction's delta <= 0, REJECT.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
logA, logpi = make_transitions(0.03, 80.0)

docs_all = tr["document_id"].unique()
half_a = set(docs_all[::2]); half_b = set(docs_all[1::2])
mask_a = tr["document_id"].isin(half_a).to_numpy()
dfa, dfb = tr[mask_a].reset_index(drop=True), tr[~mask_a].reset_index(drop=True)
eva, evb = tre[mask_a].reset_index(drop=True), tre[~mask_a].reset_index(drop=True)


def ngrams_at(df, idx, n):
    """n-grams (tuples) covering position idx within its doc."""
    tid = df["token_id"].to_numpy()
    docs = df["document_id"].to_numpy()
    out = []
    for i in idx:
        for off in range(n):
            a = i - off
            b = a + n
            if a < 0 or b > len(df):
                continue
            if docs[a] != docs[i] or docs[b - 1] != docs[i]:
                continue
            out.append(tuple(tid[a:b]))
    return out


def learn_signatures(fit_df, fit_ev, score_df, q=0.995, min_count=2, min_lift=8.0):
    """Fit on fit_df, score score_df, learn overrepresented tail n-grams."""
    params = fit_emissions(fit_df, fit_ev, use_entropy=BEST)
    sc = hmm_scores(score_df, params, logA, logpi)
    y = score_df["label"].to_numpy()
    clean = y == 0
    tau = np.quantile(sc[clean], q)
    tail_idx = np.where(clean & (sc >= tau))[0]
    n_clean = clean.sum()
    sigs = set()
    for n in (2, 3):
        tail_ngrams = Counter(ngrams_at(score_df, tail_idx, n))
        # background frequency over all clean positions
        clean_idx = np.where(clean)[0]
        bg = Counter(ngrams_at(score_df, clean_idx, n))
        for ng, c in tail_ngrams.items():
            if c < min_count:
                continue
            lift = (c / max(len(tail_idx), 1)) / (bg[ng] / n_clean)
            if lift >= min_lift:
                sigs.add(ng)
    return sigs, len(tail_idx)


def apply_mask(df, sigs):
    """Invalidate all scheme emissions on tokens covered by a signature n-gram."""
    d = df.copy()
    tid = d["token_id"].to_numpy()
    docs = d["document_id"].to_numpy()
    kill = np.zeros(len(d), bool)
    by_len: dict[int, set] = {}
    for s in sigs:
        by_len.setdefault(len(s), set()).add(s)
    for n, ss in by_len.items():
        for a in range(len(d) - n + 1):
            if docs[a] != docs[a + n - 1]:
                continue
            if tuple(tid[a:a + n]) in ss:
                kill[a:a + n] = True
    d.loc[kill, "gumbel_first"] = 0
    d.loc[kill, "textseal_first"] = 0
    d.loc[kill, "kgw_first"] = 0
    d.loc[kill, "is_repeated_ngram_2"] = 1
    print(f"    masked {kill.sum()} tokens ({kill.mean()*100:.2f}%)")
    return d


def direction(fit_df, fit_ev, test_df, test_ev, tag):
    sigs, ntail = learn_signatures(fit_df, fit_ev, fit_df)  # learn on FIT half (in-sample tail)
    print(f"  {tag}: {len(sigs)} signatures from {ntail} tail tokens")
    params = fit_emissions(fit_df, fit_ev, use_entropy=BEST)
    y = test_df["label"].to_numpy()
    base = tpr_at_fpr(y, hmm_scores(test_df, params, logA, logpi), 0.001)
    masked = apply_mask(test_df, sigs)
    filt = tpr_at_fpr(y, hmm_scores(masked, params, logA, logpi), 0.001)
    print(f"    base {base:.4f} -> filtered {filt:.4f}  (delta {filt-base:+.4f})")
    return filt - base


print("== split-half signature filtering (pre-registered reject if either <= 0) ==")
d1 = direction(dfa, eva, dfb, evb, "A->B")
d2 = direction(dfb, evb, dfa, eva, "B->A")
print(f"\nVERDICT: {'PASS -> evaluate on validation' if d1 > 0 and d2 > 0 else 'REJECT (no transfer)'}")

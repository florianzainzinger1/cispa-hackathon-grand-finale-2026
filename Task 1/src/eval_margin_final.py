"""FINAL margin/S3 channel verdict run — full ship-gate + recalls + FP-tail.

Candidate: top-48 loser-suppression evidence (build_margin_channel.py) as a
joint emission axis for gumbel/textseal, following the lp_match pattern:
u-bin x margin-level (train-clean tercile edges) x logprob level.

Configs (all fit-train-clean; margin edges recomputed from EACH fit set's
clean tokens, so split-half evals test the full pipeline transfer):
  champion       BEST cond (gumbel/textseal logprob)               [0.4384]
  s3g12+tsmax12  gumbel s3sup, textseal max-key margin, 12 u-bins  [primary]
  s3g12          gumbel s3sup only, 12 u-bins
  s3g+tsmax      same as primary but coarse 6 u-bins (spec literal)
  s3g            gumbel s3sup only, coarse 6 u-bins

Gate per config: full val, val halves A/B (fit train), reverse splits
(fit train+other half), train split-half A->B and B->A. PASS = all 7 >=
champion.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES, set_margin_edges  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"

FULL_BINS = hmm.SCORE_BINS
COARSE_BINS = hmm.SCORE_BINS_COARSE


def load_with_margin(split):
    df = pd.read_parquet(C / f"{split}_signals.parquet").copy()
    m = np.load(C / f"margin_{split}.npz")
    # canonical channel columns consumed by hmm cond modes "margin"/"s3":
    #   margin_ts  = MAX over keys A/B (only one key active per wm token;
    #                mean dilutes — max measured better, see step-2 scan)
    #   s3_*       = suppression-checked S3 (s3sup); the literal spec formula
    #                anti-separates (AUC 0.43) and was discarded at sanity.
    df["margin_gumbel"] = m["margin_gumbel"].astype(np.float64)
    df["margin_ts"] = np.maximum(m["margin_tsa"], m["margin_tsb"]).astype(np.float64)
    df["s3_gumbel"] = m["s3sup_gumbel"].astype(np.float64)
    df["s3_ts"] = m["s3sup_ts_mean"].astype(np.float64)
    ev = pd.read_parquet(C / f"{split}_evidence.parquet")
    return df, ev


tr, tre = load_with_margin("train")
va, vae = load_with_margin("validation")

# validation halves (doc-level, alternating)
vdocs = pd.unique(va["document_id"])
in_a = va["document_id"].isin(set(vdocs[::2])).to_numpy()
va_a, vae_a = va[in_a], vae[in_a]
va_b, vae_b = va[~in_a], vae[~in_a]
trva_a = pd.concat([tr, va_a], ignore_index=True)
treva_a = pd.concat([tre, vae_a], ignore_index=True)
trva_b = pd.concat([tr, va_b], ignore_index=True)
treva_b = pd.concat([tre, vae_b], ignore_index=True)

# train halves (doc-level, alternating)
tdocs = pd.unique(tr["document_id"])
t_in_a = tr["document_id"].isin(set(tdocs[::2])).to_numpy()
tr_a, tre_a = tr[t_in_a].reset_index(drop=True), tre[t_in_a].reset_index(drop=True)
tr_b, tre_b = tr[~t_in_a].reset_index(drop=True), tre[~t_in_a].reset_index(drop=True)

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
logA, logpi = make_transitions(0.03, 80.0)

CONFIGS = [
    # tag, cond, u_bins
    ("champion", BEST, COARSE_BINS),
    ("s3g12+tsmax12", {**BEST, "gumbel": "s3", "textseal": "margin"}, FULL_BINS),
    ("s3g12", {**BEST, "gumbel": "s3"}, FULL_BINS),
    ("s3g+tsmax", {**BEST, "gumbel": "s3", "textseal": "margin"}, COARSE_BINS),
    ("s3g", {**BEST, "gumbel": "s3"}, COARSE_BINS),
]


def run(fit_df, fit_ev, eval_df, cond, ubins):
    hmm.SCORE_BINS_COARSE = ubins
    hmm.MARGIN_EDGES = {}
    set_margin_edges(fit_df, n_levels=3)  # tercile edges from FIT-set clean
    params = fit_emissions(fit_df, fit_ev, use_entropy=cond)
    sc = hmm_scores(eval_df, params, logA, logpi)
    hmm.SCORE_BINS_COARSE = COARSE_BINS
    return sc


y = va["label"].to_numpy()
wm = y == 1
dom = vae[SCHEMES].to_numpy().argmax(axis=1)

print("=== SHIP GATE (7 evals per config; candidate needs all >= champion) ===")
print(f"{'config':14s} {'val':>7s} {'valHA':>7s} {'valHB':>7s} {'revA':>7s} {'revB':>7s} {'trA>B':>7s} {'trB>A':>7s}")
gate_rows = {}
full_scores = {}
for tag, cond, ubins in CONFIGS:
    sc_full = run(tr, tre, va, cond, ubins)
    full_scores[tag] = sc_full
    vals = [
        tpr_at_fpr(y, sc_full, 0.001),
        tpr_at_fpr(va_a["label"].to_numpy(), run(tr, tre, va_a, cond, ubins), 0.001),
        tpr_at_fpr(va_b["label"].to_numpy(), run(tr, tre, va_b, cond, ubins), 0.001),
        tpr_at_fpr(va_a["label"].to_numpy(), run(trva_b, treva_b, va_a, cond, ubins), 0.001),
        tpr_at_fpr(va_b["label"].to_numpy(), run(trva_a, treva_a, va_b, cond, ubins), 0.001),
        tpr_at_fpr(tr_b["label"].to_numpy(), run(tr_a, tre_a, tr_b, cond, ubins), 0.001),
        tpr_at_fpr(tr_a["label"].to_numpy(), run(tr_b, tre_b, tr_a, cond, ubins), 0.001),
    ]
    gate_rows[tag] = np.array(vals)
    print(f"{tag:14s} " + " ".join(f"{v:7.4f}" for v in vals))

ch = gate_rows["champion"]
for tag in gate_rows:
    if tag != "champion":
        wins = (gate_rows[tag] >= ch).sum()
        print(f"  {tag}: >= champion on {wins}/7   (delta val {gate_rows[tag][0]-ch[0]:+.4f})")

print("\n=== PER-SCHEME RECALL (full val, fit train, pooled 0.1% threshold) ===")
print(f"{'config':14s} " + " ".join(f"{s:>9s}" for s in SCHEMES))
for tag, sc in full_scores.items():
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    recs = []
    for si in range(4):
        sel = wm & (dom == si)
        recs.append((sel & (sc >= tau)).sum() / max(sel.sum(), 1))
    print(f"{tag:14s} " + " ".join(f"{r:9.3f}" for r in recs))

print("\n=== FP-TAIL INSPECTION (top-150 clean tokens by score, full val) ===")
pos_arr = va["position"].to_numpy()
docs_arr = va["document_id"].to_numpy()
ent_arr = va["qwen_entropy"].to_numpy()
rep_arr = va["is_repeated_ngram_2"].to_numpy()
mg_arr = va["s3_gumbel"].to_numpy()
mt_arr = va["margin_ts"].to_numpy()

dist = np.full(len(va), 10 ** 9)
for doc in vdocs:
    m = docs_arr == doc
    wm_pos = pos_arr[m & wm]
    if len(wm_pos) == 0:
        continue
    d = np.abs(pos_arr[m][:, None] - wm_pos[None, :]).min(axis=1)
    dist[m] = d

for tag, sc in full_scores.items():
    clean_idx = np.where(y == 0)[0]
    top = clean_idx[np.argsort(sc[clean_idx])[::-1][:150]]
    bb = (dist[top] <= 10).mean()
    print(f"{tag:14s} boundary<=10: {bb:.2f}  med_ent: {np.nanmedian(ent_arr[top]):.2f}  "
          f"repeated: {rep_arr[top].mean():.2f}  "
          f"s3g_hi: {np.nanmean(mg_arr[top] > 0.1):.2f}  tsmax_hi: {np.nanmean(mt_arr[top] > 2):.2f}")

print("\n=== THIN-BIN CHECK (primary config, fit full train) ===")
from hmm import extract_observations, assign_spans  # noqa
hmm.SCORE_BINS_COARSE = FULL_BINS
hmm.MARGIN_EDGES = {}
set_margin_edges(tr, n_levels=3)
cond = {**BEST, "gumbel": "s3", "textseal": "margin"}
obs = extract_observations(tr, cond)
assign = assign_spans(tr, tre)
for scheme in ("gumbel", "textseal"):
    si = SCHEMES.index(scheme)
    codes = obs[scheme]
    sel = (assign == si) & (codes >= 0)
    total = hmm.signal_bins(scheme, cond[scheme]) * 4
    cnt = np.bincount(codes[sel], minlength=total)
    csel = (tr["label"].to_numpy() == 0) & (codes >= 0)
    ccnt = np.bincount(codes[csel], minlength=total)
    print(f"{scheme}: {total} codes | wm min={cnt.min()} med={np.median(cnt):.0f} "
          f"cells<5: {(cnt < 5).sum()} | clean min={ccnt.min()} med={np.median(ccnt):.0f}")
hmm.SCORE_BINS_COARSE = COARSE_BINS

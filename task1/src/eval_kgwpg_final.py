"""FINAL kgwpg verdict run — full ship-gate + scheme recalls + FP-tail inspection.

Configs: champion | binned E5 | binned E7 | analytic LR (variant B, exact
per-position null, |logLR|<=delta by construction).
Gate per config: full val, val halves A/B (fit train), reverse splits
(fit train+other half), train split-half A->B and B->A. All fit-train clean.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"

def load_with_pg(split):
    df = pd.read_parquet(C / f"{split}_signals.parquet")
    d = np.load(C / f"kgwpg_{split}.npz")
    assert len(d["kgw_pgreen"]) == len(df)
    df = df.copy()
    df["kgw_pgreen"] = d["kgw_pgreen"]
    return df

tr = load_with_pg("train")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = load_with_pg("validation")
vae = pd.read_parquet(C / "validation_evidence.parquet")

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
KP = {**BEST, "kgw": "kgwpg"}
ANA = {**BEST, "kgw": None}  # analytic channel replaces the categorical
E5 = [.02, .10, .25, .45, .65]
E7 = [.01, .04, .10, .20, .35, .55, .80]
logA, logpi = make_transitions(0.03, 80.0)

CONFIGS = [
    ("champion", BEST, None, False),
    ("binned E5", KP, E5, False),
    ("binned E7", KP, E7, False),
    ("analytic B", ANA, None, True),
]


def run(fit_df, fit_ev, eval_df, cond, edges, analytic):
    hmm.KGW_ANALYTIC = analytic
    if edges is not None:
        hmm.KGWPG_EDGES = np.array(edges)
    params = fit_emissions(fit_df, fit_ev, use_entropy=cond)
    sc = hmm_scores(eval_df, params, logA, logpi)
    hmm.KGW_ANALYTIC = False
    return sc


y = va["label"].to_numpy()
wm = y == 1
dom = vae[SCHEMES].to_numpy().argmax(axis=1)

print("=== SHIP GATE (7 evals per config; candidate needs all >= champion) ===")
print(f"{'config':12s} {'val':>7s} {'valHA':>7s} {'valHB':>7s} {'revA':>7s} {'revB':>7s} {'trA>B':>7s} {'trB>A':>7s}")
gate_rows = {}
full_scores = {}
for tag, cond, edges, ana in CONFIGS:
    sc_full = run(tr, tre, va, cond, edges, ana)
    full_scores[tag] = sc_full
    vals = [
        tpr_at_fpr(y, sc_full, 0.001),
        tpr_at_fpr(va_a["label"].to_numpy(), run(tr, tre, va_a, cond, edges, ana), 0.001),
        tpr_at_fpr(va_b["label"].to_numpy(), run(tr, tre, va_b, cond, edges, ana), 0.001),
        tpr_at_fpr(va_a["label"].to_numpy(), run(trva_b, treva_b, va_a, cond, edges, ana), 0.001),
        tpr_at_fpr(va_b["label"].to_numpy(), run(trva_a, treva_a, va_b, cond, edges, ana), 0.001),
        tpr_at_fpr(tr_b["label"].to_numpy(), run(tr_a, tre_a, tr_b, cond, edges, ana), 0.001),
        tpr_at_fpr(tr_a["label"].to_numpy(), run(tr_b, tre_b, tr_a, cond, edges, ana), 0.001),
    ]
    gate_rows[tag] = np.array(vals)
    print(f"{tag:12s} " + " ".join(f"{v:7.4f}" for v in vals))

ch = gate_rows["champion"]
for tag in gate_rows:
    if tag != "champion":
        wins = (gate_rows[tag] >= ch).sum()
        print(f"  {tag}: >= champion on {wins}/7   (delta val {gate_rows[tag][0]-ch[0]:+.4f})")

print("\n=== PER-SCHEME RECALL (full val, fit train, pooled 0.1% threshold) ===")
print(f"{'config':12s} " + " ".join(f"{s:>9s}" for s in SCHEMES))
for tag, sc in full_scores.items():
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    recs = []
    for si in range(4):
        sel = wm & (dom == si)
        recs.append((sel & (sc >= tau)).sum() / max(sel.sum(), 1))
    print(f"{tag:12s} " + " ".join(f"{r:9.3f}" for r in recs))

print("\n=== FP-TAIL INSPECTION (top-150 clean tokens by score, full val) ===")
pos_arr = va["position"].to_numpy()
docs_arr = va["document_id"].to_numpy()
pg_arr = va["kgw_pgreen"].to_numpy()
ent_arr = va["qwen_entropy"].to_numpy()
rep_arr = va["is_repeated_ngram_2"].to_numpy()

# distance to nearest true-wm token within the same document
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
    pg_t = pg_arr[top]
    okpg = ~np.isnan(pg_t)
    print(f"{tag:12s} boundary<=10: {bb:.2f}  pg<0.05: {(pg_t[okpg] < 0.05).mean():.2f}  "
          f"pg>0.95: {(pg_t[okpg] > 0.95).mean():.2f}  med_ent: {np.nanmedian(ent_arr[top]):.2f}  "
          f"repeated: {rep_arr[top].mean():.2f}")

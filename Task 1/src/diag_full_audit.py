"""Full audit of the champion solution: loss accounting, information budget,
FP-budget structure, oracle upper bounds, metric variance (doc bootstrap).

All on validation, champion config (fit train, cond best, trans 0.03/80).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import (SCHEMES, assign_spans, fit_emissions, make_transitions,
                 hmm_scores, extract_observations, _normalize_cond, _levels,
                 signal_bins, _n_levels)  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.03, 80.0)
sc = hmm_scores(va, params, logA, logpi)

y = va["label"].to_numpy()
wm = y == 1
docs = va["document_id"].to_numpy()
pos = va["position"].to_numpy()
ent = va["qwen_entropy"].to_numpy()
tau = np.quantile(sc[y == 0], 1 - 0.001)
rec = sc >= tau
print(f"champion val TPR@0.1%: {tpr_at_fpr(y, sc, 0.001):.4f}  tau={tau:.3f}  "
      f"wm={wm.sum()} clean={(~wm).sum()}  FP budget={(int((~wm).sum()*0.001))} tokens")

# ---- span table -------------------------------------------------------------
assign = assign_spans(va, vae)  # scheme id per wm token
spans = []  # (start_idx, end_idx, scheme, max_sc, mean_sc)
i = 0
n = len(va)
while i < n:
    if y[i] == 1:
        j = i
        while j < n and y[j] == 1 and docs[j] == docs[i]:
            j += 1
        spans.append((i, j, int(assign[i]), sc[i:j].max(), sc[i:j].mean()))
        i = j
    else:
        i += 1
span_id = np.full(n, -1)
for k, (a, b, *_situ) in enumerate(spans):
    span_id[a:b] = k
print(f"spans: {len(spans)}  by scheme: " +
      " ".join(f"{s}:{sum(1 for x in spans if x[2]==si)}" for si, s in enumerate(SCHEMES)))

# ---- 1. loss accounting -----------------------------------------------------
print("\n=== 1. MISS ATTRIBUTION (per wm token, champion) ===")
obs = extract_observations(va, BEST)
span_max = np.full(n, np.nan)
for k, (a, b, _s, mx, _mn) in enumerate(spans):
    span_max[a:b] = mx
edge_dist = np.full(n, 10 ** 9)
for (a, b, *_x) in spans:
    idx = np.arange(a, b)
    edge_dist[a:b] = np.minimum(idx - a, b - 1 - idx)

rows = []
for si, s in enumerate(SCHEMES):
    sel = wm & (assign == si)
    m = sel & ~rec
    tot, miss = sel.sum(), m.sum()
    span_dead = m & (span_max < tau)                    # whole span never fires
    part = m & ~span_dead                               # span fires, token missed
    p_edge = part & (edge_dist <= 3)
    p_lowent = part & ~(edge_dist <= 3) & (np.nan_to_num(ent, nan=0) < 0.3)
    p_invalid = part & ~(edge_dist <= 3) & ~(np.nan_to_num(ent, nan=0) < 0.3) & (obs[s][part.nonzero()[0]] < 0).any() * part  # placeholder
    # simpler: invalid = own-scheme obs code < 0
    inv = obs[s] < 0
    p_inv = part & ~(edge_dist <= 3) & ~(np.nan_to_num(ent, nan=0) < 0.3) & inv
    p_rest = part & ~(edge_dist <= 3) & ~(np.nan_to_num(ent, nan=0) < 0.3) & ~inv
    rows.append((s, tot, miss / max(tot, 1), span_dead.sum() / max(tot, 1),
                 p_edge.sum() / max(tot, 1), p_lowent.sum() / max(tot, 1),
                 p_inv.sum() / max(tot, 1), p_rest.sum() / max(tot, 1)))
print(f"{'scheme':9s} {'wm_tok':>6s} {'missed':>7s} {'dead-span':>9s} {'edge<=3':>8s} "
      f"{'low-ent':>8s} {'invalid':>8s} {'other':>7s}")
for r in rows:
    print(f"{r[0]:9s} {r[1]:6d} {r[2]:7.1%} {r[3]:9.1%} {r[4]:8.1%} {r[5]:8.1%} {r[6]:8.1%} {r[7]:7.1%}")

# dead spans per scheme
print("\ndead spans (span max < tau) / total spans:")
for si, s in enumerate(SCHEMES):
    ss = [x for x in spans if x[2] == si]
    dead = sum(1 for x in ss if x[3] < tau)
    lens = [b - a for (a, b, *_z) in ss]
    print(f"  {s:9s} {dead:3d}/{len(ss):3d} dead  median_len={int(np.median(lens)) if lens else 0}")

# ---- 2. information budget (nats per token from fitted emissions) -----------
print("\n=== 2. INFORMATION BUDGET (E[LLR|wm] nats/token from fitted emissions) ===")
cond = _normalize_cond(BEST)
for s in SCHEMES:
    lw, lc = params[s]["log_wm"], params[s]["log_clean"]
    nb = signal_bins(s, cond[s])
    nl = _n_levels(cond[s])
    # weight per level by wm token level distribution on val
    lev = _levels(va, cond[s])
    sel = wm & (assign == SCHEMES.index(s)) & (obs[s] >= 0)
    kl_levels = []
    for l in range(nl):
        sl = slice(l * nb, (l + 1) * nb)
        pw = np.exp(lw[sl]); pw /= pw.sum()
        pc = np.exp(lc[sl]); pc /= pc.sum()
        kl_levels.append((pw * (np.log(pw) - np.log(pc))).sum())
    w = np.bincount(lev[sel], minlength=nl).astype(float)
    w /= max(w.sum(), 1)
    e_llr = float((w * np.array(kl_levels)).sum())
    med_len = int(np.median([b - a for (a, b, ssi, *_z) in spans if ssi == SCHEMES.index(s)] or [0]))
    print(f"  {s:9s} E[LLR|wm]={e_llr:6.3f} nats/tok  x median span {med_len:3d} = {e_llr*med_len:6.1f} nats "
          f"(threshold needs ~ln(1000)={np.log(1000):.1f} + entry ~{-np.log(0.03/4):.1f})")

# ---- 3. FP budget structure + oracle bounds ---------------------------------
print("\n=== 3. FP BUDGET & ORACLE UPPER BOUNDS ===")
dist_wm = np.full(n, 10 ** 9)
for d in pd.unique(docs):
    m = docs == d
    wmp = pos[m & wm]
    if len(wmp):
        dist_wm[m] = np.abs(pos[m][:, None] - wmp[None, :]).min(axis=1)
clean = ~wm
n_fp = int(clean.sum() * 0.001)
top_clean = np.argsort(np.where(clean, sc, -np.inf))[::-1][:n_fp]
print(f"top-{n_fp} clean (the FP budget): within10={np.mean(dist_wm[top_clean]<=10):.1%} "
      f"within3={np.mean(dist_wm[top_clean]<=3):.1%}  in wm-docs={np.mean(dist_wm[top_clean]<10**9):.1%}")

sc_oracle = sc.copy()
sc_oracle[clean & (dist_wm <= 10)] = 0.0     # perfect boundary handling
print(f"oracle A (zero all clean scores within 10 of true span): "
      f"TPR@0.1% = {tpr_at_fpr(y, sc_oracle, 0.001):.4f}  (champion {tpr_at_fpr(y, sc, 0.001):.4f})")
sc_oracle2 = sc.copy()
sc_oracle2[clean] = np.minimum(sc_oracle2[clean], np.quantile(sc[clean], 0.999))  # flatten only top tail
print(f"oracle B (all clean FPs capped at the 99.9% clean level): "
      f"TPR@0.1% = {tpr_at_fpr(y, sc_oracle2, 0.001):.4f}")

# ---- 4. metric variance: doc-level bootstrap --------------------------------
print("\n=== 4. METRIC VARIANCE (doc bootstrap, 1000 iters) ===")
# also load best kgwpg (E7) scores for paired delta
va_pg = va.copy()
va_pg["kgw_pgreen"] = np.load(C / "kgwpg_validation.npz")["kgw_pgreen"]
tr_pg = tr.copy()
tr_pg["kgw_pgreen"] = np.load(C / "kgwpg_train.npz")["kgw_pgreen"]
hmm.KGWPG_EDGES = np.array([.01, .04, .10, .20, .35, .55, .80])
params_pg = fit_emissions(tr_pg, tre, use_entropy={**BEST, "kgw": "kgwpg"})
sc_pg = hmm_scores(va_pg, params_pg, logA, logpi)

doc_list = pd.unique(docs)
doc_idx = {d: np.where(docs == d)[0] for d in doc_list}
rng = np.random.default_rng(0)
t_ch, t_pg = [], []
for _ in range(1000):
    sample = rng.choice(doc_list, size=len(doc_list), replace=True)
    idx = np.concatenate([doc_idx[d] for d in sample])
    yy = y[idx]
    t_ch.append(tpr_at_fpr(yy, sc[idx], 0.001))
    t_pg.append(tpr_at_fpr(yy, sc_pg[idx], 0.001))
t_ch, t_pg = np.array(t_ch), np.array(t_pg)
dl = t_pg - t_ch
print(f"champion:  mean {t_ch.mean():.4f}  CI95 [{np.quantile(t_ch,.025):.4f}, {np.quantile(t_ch,.975):.4f}]  sd {t_ch.std():.4f}")
print(f"kgwpg E7:  mean {t_pg.mean():.4f}  CI95 [{np.quantile(t_pg,.025):.4f}, {np.quantile(t_pg,.975):.4f}]")
print(f"paired delta: mean {dl.mean():+.4f}  CI95 [{np.quantile(dl,.025):+.4f}, {np.quantile(dl,.975):+.4f}]  P(delta>0)={np.mean(dl>0):.2f}")

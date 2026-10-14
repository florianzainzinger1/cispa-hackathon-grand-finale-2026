"""Structural segmentation priors on the HSMM decode — 7-eval ship gate vs
the HSMM champion (sub_011: fixed-Z, empirical duration prior, kappa=0.25).

Variants (src/hsmm.py hsmm_doc_struct, brute-force verified in scratchpad
v_toy_struct.py to ~1e-15):
  gap12: min-gap refractory — a new span may start only after >= 12 clean
         tokens since the previous span ended (label fact: min inter-span gap
         is exactly 12 on train AND validation; doc start exempt).
  count: per-doc span-count prior — smoothed (+0.25) empirical pmf over
         0..6 spans from the FIT labels, applied as a tilt over segmentations.
  both:  refractory + count prior.

Protocol identical to src/eval_hsmm_gate.py: 7 evals (val / valHA / valHB /
revA / revB / trA>B / trB>A), doc-level alternating halves; emissions AND all
priors (duration + count) refit on the fit split of every eval. Ship rule
(pre-registered): 7/7 wins vs the HSMM champion AND val >= champion + 0.005.
Plus: paired doc-bootstrap (1000 iters, seed 0) on full val, per-scheme
recall, FP-tail incl. per-doc counts (validation_82 watch).

Usage: python src/eval_hsmm_struct_gate.py [--variants gap12 count both]
       [--processes N] [--save-prefix PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import SCHEMES, fit_emissions  # noqa: E402
from hsmm import (  # noqa: E402
    duration_logpmf_from,
    span_count_logpmf_from,
    hsmm_scores,
)
from metrics import tpr_at_fpr  # noqa: E402

C = Path(__file__).resolve().parent.parent / "cache"
BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
PE, P0 = 0.03, 0.07
MIN_GAP = 12

VARIANT_KW = {
    "champ": dict(min_gap=1, count=False),
    "gap12": dict(min_gap=MIN_GAP, count=False),
    "count": dict(min_gap=1, count=True),
    "both": dict(min_gap=MIN_GAP, count=True),
}


def decode(edf, prm, logPD, logPN, vkw, processes):
    return hsmm_scores(
        edf, prm, logPD, PE, P0, "fixed", processes,
        min_gap=vkw["min_gap"], logPN=logPN if vkw["count"] else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["gap12", "count", "both"],
                    choices=["gap12", "count", "both"])
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--count-kappa", type=float, default=0.25)
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--save-prefix", type=Path, default=None,
                    help="save full-val scores per variant as PREFIX_<name>.npy")
    args = ap.parse_args()

    tr = pd.read_parquet(C / "train_signals.parquet")
    tre = pd.read_parquet(C / "train_evidence.parquet")
    va = pd.read_parquet(C / "validation_signals.parquet")
    vae = pd.read_parquet(C / "validation_evidence.parquet")

    # splits exactly as in eval_hsmm_gate.py (doc-level, alternating)
    vdocs = pd.unique(va["document_id"])
    in_a = va["document_id"].isin(set(vdocs[::2])).to_numpy()
    va_a, vae_a = va[in_a].reset_index(drop=True), vae[in_a].reset_index(drop=True)
    va_b, vae_b = va[~in_a].reset_index(drop=True), vae[~in_a].reset_index(drop=True)
    trva_a = pd.concat([tr, va_a], ignore_index=True)
    treva_a = pd.concat([tre, vae_a], ignore_index=True)
    trva_b = pd.concat([tr, va_b], ignore_index=True)
    treva_b = pd.concat([tre, vae_b], ignore_index=True)
    tdocs = pd.unique(tr["document_id"])
    t_in_a = tr["document_id"].isin(set(tdocs[::2])).to_numpy()
    tr_a, tre_a = tr[t_in_a].reset_index(drop=True), tre[t_in_a].reset_index(drop=True)
    tr_b, tre_b = tr[~t_in_a].reset_index(drop=True), tre[~t_in_a].reset_index(drop=True)

    y = va["label"].to_numpy()
    variants = ["champ"] + args.variants

    evals = [
        ("val",   tr, tre, va),
        ("valHA", tr, tre, va_a),
        ("valHB", tr, tre, va_b),
        ("revA",  trva_b, treva_b, va_a),
        ("revB",  trva_a, treva_a, va_b),
        ("trA>B", tr_a, tre_a, tr_b),
        ("trB>A", tr_b, tre_b, tr_a),
    ]
    res: dict[str, dict[str, float]] = {v: {} for v in variants}
    val_scores: dict[str, np.ndarray] = {}
    hdr = f"{'eval':6s} {'champ':>7s}" + "".join(
        f" {v:>7s} {'d_' + v[:4]:>8s}" for v in args.variants
    )
    print(hdr)
    t_all = time.time()
    for name, fdf, fev, edf in evals:
        prm = fit_emissions(fdf, fev, use_entropy=BEST)
        logPD = duration_logpmf_from(fdf, kappa=args.kappa)
        logPN = span_count_logpmf_from(fdf, kappa=args.count_kappa)
        yy = edf["label"].to_numpy()
        row = f"{name:6s}"
        for v in variants:
            t0 = time.time()
            sc = decode(edf, prm, logPD, logPN, VARIANT_KW[v], args.processes)
            h = tpr_at_fpr(yy, sc, 0.001)
            res[v][name] = h
            if v == "champ":
                row += f" {h:7.4f}"
            else:
                row += f" {h:7.4f} {h - res['champ'][name]:+8.4f}"
            if name == "val":
                val_scores[v] = sc
            del t0
        print(row, flush=True)
    print(f"[gate decodes: {time.time() - t_all:.0f}s]")
    for v in args.variants:
        wins = sum(1 for k in res[v] if res[v][k] >= res["champ"][k])
        dv = res[v]["val"] - res["champ"]["val"]
        ship = wins == 7 and dv >= 0.005
        print(f"{v}: >= champion on {wins}/7   delta val {dv:+.4f}   "
              f"ship-rule {'PASS' if ship else 'fail'}")

    # ---- paired doc-bootstrap on full val (1000 iters, seed 0) ----
    docs = va["document_id"].to_numpy()
    doc_idx = {d: np.where(docs == d)[0] for d in vdocs}
    rng = np.random.default_rng(0)
    samples = [rng.choice(vdocs, size=len(vdocs), replace=True) for _ in range(1000)]
    print("\npaired doc-bootstrap vs HSMM champion (1000 iters, seed 0, full val):")
    for v in args.variants:
        tc, th = [], []
        for sample in samples:
            idx = np.concatenate([doc_idx[d] for d in sample])
            tc.append(tpr_at_fpr(y[idx], val_scores["champ"][idx], 0.001))
            th.append(tpr_at_fpr(y[idx], val_scores[v][idx], 0.001))
        dl = np.array(th) - np.array(tc)
        print(f"  {v:5s}: champ {np.mean(tc):.4f} var {np.mean(th):.4f} "
              f"delta {dl.mean():+.4f} CI95 [{np.quantile(dl, .025):+.4f}, "
              f"{np.quantile(dl, .975):+.4f}] P(delta>0)={np.mean(dl > 0):.3f}")

    # ---- per-scheme recall (full val, pooled 0.1% clean threshold) ----
    wm = y == 1
    dom = vae[SCHEMES].to_numpy().argmax(axis=1)
    print("\nper-scheme recall (full val):")
    for v in variants:
        sc = val_scores[v]
        tau = np.quantile(sc[y == 0], 0.999)
        recs = [((wm & (dom == si)) & (sc >= tau)).sum() / max((wm & (dom == si)).sum(), 1)
                for si in range(4)]
        print(f"  {v:5s}: tau={tau:6.3f}  "
              + "  ".join(f"{s}={r:.3f}" for s, r in zip(SCHEMES, recs)))

    # ---- FP-tail inspection ----
    pos = va["position"].to_numpy()
    dist_wm = np.full(len(va), 10 ** 9)
    for dd in vdocs:
        mdoc = np.where(docs == dd)[0]
        wmp = pos[mdoc][y[mdoc] == 1]
        if len(wmp):
            dist_wm[mdoc] = np.abs(pos[mdoc][:, None] - wmp[None, :]).min(axis=1)
    clean_idx = np.where(y == 0)[0]
    print("\nFP-tail (clean tokens by score, top-51 = 0.1% budget):")
    for v in variants:
        sc = val_scores[v]
        order = clean_idx[np.argsort(sc[clean_idx])[::-1]]
        top = order[:51]
        u, cnt = np.unique(docs[top], return_counts=True)
        by_doc = sorted(zip(cnt, u), reverse=True)[:5]
        v82 = int(cnt[list(u).index("validation_82")]) if "validation_82" in u else 0
        print(f"  {v:5s}: within10={np.mean(dist_wm[top] <= 10):.2f} "
              f"within3={np.mean(dist_wm[top] <= 3):.2f} docs={len(u)} "
              f"validation_82={v82}  top-docs: "
              + ", ".join(f"{d}={c}" for c, d in by_doc))

    if args.save_prefix is not None:
        for v, sc in val_scores.items():
            p = Path(str(args.save_prefix) + f"_{v}.npy")
            np.save(p, sc)
        print(f"\nsaved full-val scores: {args.save_prefix}_<variant>.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

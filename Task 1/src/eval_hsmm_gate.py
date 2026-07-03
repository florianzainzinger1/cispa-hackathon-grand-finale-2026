"""HSMM (explicit empirical duration prior) — full 7-eval ship gate vs champion.

Independent re-verification port of the fresh-context researcher's candidate
(scratchpad hsmm_gate.py). Differences from the reference:
  * z_mode='fixed' corrects a partition-function double count of
    doc-starts-inside-a-span paths (see src/hsmm.py docstring);
    z_mode='ref' reproduces the reference exactly and is reported alongside.
  * duration prior support is widened automatically if the fit split contains
    spans outside [24, 320] (never happens on train/val, guards the future).

Protocol (identical to src/eval_kgwpg_final.py):
  val / valHA / valHB (fit train), revA / revB (fit train + other val half),
  trA>B / trB>A (train split-halves). Splits are doc-level alternating halves.
  Duration prior AND emissions are refit on the fit split of every eval —
  labels of the eval split are used for metric computation only.
Plus: geometric-prior anchor (must equal champion), paired doc-bootstrap
(1000 iters, seed 0) on full val, per-scheme recalls, FP-tail inspection.

Usage: python src/eval_hsmm_gate.py [--skip-geom] [--kappa 0.25]
       [--geom-blend 0.0] [--processes N] [--save-val PATH.npy]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import SCHEMES, fit_emissions, make_transitions, hmm_scores  # noqa: E402
from hsmm import duration_logpmf_from, geometric_logpmf, hsmm_scores  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

C = Path(__file__).resolve().parent.parent / "cache"
BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
PE, P0 = 0.03, 0.07


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--geom-blend", type=float, default=0.0)
    ap.add_argument("--skip-geom", action="store_true")
    ap.add_argument("--skip-ref", action="store_true", help="skip bug-compatible reference mode")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--save-val", type=Path, default=None, help="save fixed-mode full-val scores")
    args = ap.parse_args()

    tr = pd.read_parquet(C / "train_signals.parquet")
    tre = pd.read_parquet(C / "train_evidence.parquet")
    va = pd.read_parquet(C / "validation_signals.parquet")
    vae = pd.read_parquet(C / "validation_evidence.parquet")

    # splits exactly as in eval_kgwpg_final.py (doc-level, alternating)
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

    # ---- correctness anchor: geometric-prior HSMM must equal champion ----
    params_tr = fit_emissions(tr, tre, use_entropy=BEST)
    logA, logpi = make_transitions(PE, 80.0)
    sc_champ_val = hmm_scores(va, params_tr, logA, logpi)
    t_champ = tpr_at_fpr(y, sc_champ_val, 0.001)
    print(f"champion val TPR@0.1% = {t_champ:.4f}")
    if not args.skip_geom:
        t0 = time.time()
        sc_geom = hsmm_scores(va, params_tr, geometric_logpmf(2000, 80.0),
                              PE, P0, "fixed", args.processes)
        t_geom = tpr_at_fpr(y, sc_geom, 0.001)
        dmax_tok = float(np.abs(sc_geom - sc_champ_val).max())
        ok = abs(t_geom - t_champ) <= 0.001
        print(f"geometric-anchor HSMM: {t_geom:.4f} (delta {t_geom - t_champ:+.4f}, "
              f"max per-token |diff| {dmax_tok:.2e}) [{time.time()-t0:.0f}s] "
              f"{'OK' if ok else '*** FAILED ***'}")
        if not ok:
            print("STOP: geometric anchor failed — HSMM does not reduce to the champion.")
            return 1

    # ---- 7-eval gate ----
    evals = [
        ("val",   tr, tre, va),
        ("valHA", tr, tre, va_a),
        ("valHB", tr, tre, va_b),
        ("revA",  trva_b, treva_b, va_a),
        ("revB",  trva_a, treva_a, va_b),
        ("trA>B", tr_a, tre_a, tr_b),
        ("trB>A", tr_b, tre_b, tr_a),
    ]
    modes = ["fixed"] + ([] if args.skip_ref else ["ref"])
    res: dict[str, dict[str, float]] = {m: {} for m in ["champ"] + modes}
    val_scores: dict[str, np.ndarray] = {}
    hdr = f"{'eval':6s} {'champ':>7s}" + "".join(f" {m:>7s} {'d_'+m[:3]:>8s}" for m in modes)
    print("\n" + hdr)
    for name, fdf, fev, edf in evals:
        prm = fit_emissions(fdf, fev, use_entropy=BEST)
        logPD = duration_logpmf_from(fdf, kappa=args.kappa, geom_blend=args.geom_blend)
        yy = edf["label"].to_numpy()
        c = tpr_at_fpr(yy, hmm_scores(edf, prm, logA, logpi), 0.001)
        res["champ"][name] = c
        row = f"{name:6s} {c:7.4f}"
        for m in modes:
            sc = hsmm_scores(edf, prm, logPD, PE, P0, m, args.processes)
            h = tpr_at_fpr(yy, sc, 0.001)
            res[m][name] = h
            row += f" {h:7.4f} {h - c:+8.4f}"
            if name == "val":
                val_scores[m] = sc
        print(row)
    for m in modes:
        wins = sum(1 for k in res[m] if res[m][k] >= res["champ"][k])
        print(f"{m}: >= champion on {wins}/7   (delta val {res[m]['val'] - res['champ']['val']:+.4f})")

    # ---- paired doc-bootstrap on full val (1000 iters, seed 0) ----
    docs = va["document_id"].to_numpy()
    doc_idx = {d: np.where(docs == d)[0] for d in vdocs}
    rng = np.random.default_rng(0)
    samples = [rng.choice(vdocs, size=len(vdocs), replace=True) for _ in range(1000)]
    print("\npaired doc-bootstrap vs champion (1000 iters, seed 0, full val):")
    for m in modes:
        tc, th = [], []
        for sample in samples:
            idx = np.concatenate([doc_idx[d] for d in sample])
            tc.append(tpr_at_fpr(y[idx], sc_champ_val[idx], 0.001))
            th.append(tpr_at_fpr(y[idx], val_scores[m][idx], 0.001))
        dl = np.array(th) - np.array(tc)
        print(f"  {m:5s}: champ {np.mean(tc):.4f} hsmm {np.mean(th):.4f} "
              f"delta {dl.mean():+.4f} CI95 [{np.quantile(dl, .025):+.4f}, {np.quantile(dl, .975):+.4f}] "
              f"P(delta>0)={np.mean(dl > 0):.3f}")

    # ---- per-scheme recall (full val, pooled 0.1% clean threshold) ----
    wm = y == 1
    dom = vae[SCHEMES].to_numpy().argmax(axis=1)
    print("\nper-scheme recall (full val):")
    for tag, sc in [("champ", sc_champ_val)] + [(m, val_scores[m]) for m in modes]:
        tau = np.quantile(sc[y == 0], 0.999)
        recs = [((wm & (dom == si)) & (sc >= tau)).sum() / max((wm & (dom == si)).sum(), 1)
                for si in range(4)]
        print(f"  {tag:5s}: tau={tau:6.3f}  " + "  ".join(f"{s}={r:.3f}" for s, r in zip(SCHEMES, recs)))

    # ---- FP-tail inspection ----
    pos = va["position"].to_numpy()
    dist_wm = np.full(len(va), 10 ** 9)
    for dd in vdocs:
        mdoc = np.where(docs == dd)[0]
        wmp = pos[mdoc][y[mdoc] == 1]
        if len(wmp):
            dist_wm[mdoc] = np.abs(pos[mdoc][:, None] - wmp[None, :]).min(axis=1)
    clean_idx = np.where(y == 0)[0]
    print("\nFP-tail (clean tokens by score):")
    for tag, sc in [("champ", sc_champ_val)] + [(m, val_scores[m]) for m in modes]:
        order = clean_idx[np.argsort(sc[clean_idx])[::-1]]
        for k in (51, 150):
            top = order[:k]
            u, cnt = np.unique(docs[top], return_counts=True)
            print(f"  {tag:5s} top-{k:3d}: within10={np.mean(dist_wm[top] <= 10):.2f} "
                  f"within3={np.mean(dist_wm[top] <= 3):.2f} docs={len(u)} maxdoc={cnt.max()}")

    if args.save_val is not None:
        np.save(args.save_val, val_scores["fixed"])
        print(f"\nsaved fixed-mode full-val scores: {args.save_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4: HMM posterior -> submission.

Emissions fitted on labeled data (train, optionally +validation), transitions
from label EDA. Score = -log10 P(Clean|signals), squashed to [0,1).

Usage:
  python src/predict_hmm.py --fit train --split validation           # metric
  python src/predict_hmm.py --fit train --split test --out submissions/sub_003_hmm.jsonl
  python src/predict_hmm.py --fit train validation --split test --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa: E402
from hmm import fit_emissions, make_transitions, hmm_scores  # noqa: E402
from hsmm import duration_logpmf_from, hsmm_scores  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
SQUASH = 8.0

def load_signals(split: str, with_pg: bool) -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / f"{split}_signals.parquet")
    if with_pg:
        d = np.load(CACHE_DIR / f"kgwpg_{split}.npz")
        assert len(d["kgw_pgreen"]) == len(df), f"{split}: kgwpg length mismatch"
        df = df.copy()
        df["kgw_pgreen"] = d["kgw_pgreen"]
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", nargs="+", default=["train"], choices=["train", "validation"])
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--p-enter", type=float, default=0.02)
    parser.add_argument("--mean-span", type=float, default=45.0)
    parser.add_argument("--use-entropy", action="store_true")
    parser.add_argument(
        "--decode",
        default="hmm",
        choices=["hmm", "hsmm"],
        help="hmm = champion 5-state decode (default); hsmm = explicit-duration "
             "semi-Markov decode with empirical span-duration prior fitted on "
             "the --fit split(s) (src/hsmm.py)",
    )
    parser.add_argument("--kappa", type=float, default=0.25,
                        help="hsmm duration-prior add-count smoothing")
    parser.add_argument("--geom-blend", type=float, default=0.0,
                        help="hsmm: blend weight of a truncated geometric into the duration prior")
    parser.add_argument("--hsmm-ensemble", action="store_true",
                        help="hsmm: mean of -log10 P(clean) over priors "
                             "{kappa=0.25, kappa=1.0, geom_blend=0.2}")
    parser.add_argument("--processes", type=int, default=None,
                        help="hsmm: multiprocessing workers for per-doc decode")
    parser.add_argument(
        "--cond",
        default=None,
        choices=[None, "entropy", "best", "match", "pgexact"],
        help="emission conditioning preset; 'best' = uni/kgw:entropy, gum/ts:logprob; "
             "'match' = gum/ts:lp_match (argmax-match joint code); "
             "'pgexact' = best but kgw conditioned on exact greenlist mass (kgw_pgreen)",
    )
    args = parser.parse_args()

    with_pg = args.cond == "pgexact"  # kgwpg edges come from hmm.KGWPG_EDGES

    fit_dfs, fit_evs = [], []
    for split in args.fit:
        fit_dfs.append(load_signals(split, with_pg))
        fit_evs.append(pd.read_parquet(CACHE_DIR / f"{split}_evidence.parquet"))
    fit_df = pd.concat(fit_dfs, ignore_index=True)
    fit_ev = pd.concat(fit_evs, ignore_index=True)

    cond = args.use_entropy
    if args.cond == "entropy":
        cond = True
    elif args.cond == "best":
        cond = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
    elif args.cond == "match":
        cond = {"unigram": "entropy", "kgw": "entropy", "gumbel": "lp_match", "textseal": "lp_match"}
    elif args.cond == "pgexact":
        cond = {"unigram": "entropy", "kgw": "kgwpg", "gumbel": "logprob", "textseal": "logprob"}
    params = fit_emissions(fit_df, fit_ev, use_entropy=cond)
    logA, logpi = make_transitions(args.p_enter, args.mean_span)

    df = load_signals(args.split, with_pg)
    if args.decode == "hsmm":
        if args.hsmm_ensemble:
            prior_cfgs = [(0.25, 0.0), (1.0, 0.0), (0.25, 0.2)]
        else:
            prior_cfgs = [(args.kappa, args.geom_blend)]
        parts = []
        for kappa, blend in prior_cfgs:
            logPD = duration_logpmf_from(fit_df, kappa=kappa, geom_blend=blend)
            parts.append(hsmm_scores(df, params, logPD, pe=args.p_enter,
                                     processes=args.processes))
        sc = np.mean(parts, axis=0)
    else:
        sc = hmm_scores(df, params, logA, logpi)

    if "label" in df.columns:
        y = df["label"].to_numpy()
        print(
            f"{args.split} (fit on {'+'.join(args.fit)}): "
            f"TPR@0.1%FPR={tpr_at_fpr(y, sc):.4f}  TPR@1%FPR={tpr_at_fpr(y, sc, 0.01):.4f}"
        )

    if args.out is not None:
        scores = sc / (sc + SQUASH)
        docs = df["document_id"].to_numpy()
        args.out.parent.mkdir(exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for doc_id in pd.unique(docs):
                s = scores[docs == doc_id]
                fh.write(json.dumps(
                    {"document_id": str(doc_id), "scores": [round(float(x), 6) for x in s]},
                    separators=(",", ":"),
                ) + "\n")
        print(f"submission written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

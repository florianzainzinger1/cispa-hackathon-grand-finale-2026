"""Phase 2-3: per-scheme window evidence -> min-p fusion -> scores/submission.

Fused evidence per token = max over the four schemes' -log10(p) window maxima
(= min-p fusion; p-values are the cross-scheme, cross-document currency).
Final [0,1] score = monotone map ev/(ev+SQUASH), ranking-preserving.

Usage:
  python src/predict_minp.py --split validation            # metric only
  python src/predict_minp.py --split test --out submissions/sub_002_minp_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import green_window_features, score_window_features, max_evidence  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
SQUASH = 20.0  # ev -> ev/(ev+SQUASH) in [0,1); monotone, keeps double precision


def _min_left_right(feats: pd.DataFrame, prefix: str) -> np.ndarray:
    """Boundary-bleed-proof evidence: max over sizes of min(left, right).

    A token only scores high if BOTH its left and right contexts look
    watermarked — kills the fatal FP bleed of centered windows next to true
    spans (found on validation_75; see reports/07_phase23_minp_fusion.md).
    """
    from features import WINDOW_SIZES

    stacked = [
        np.minimum(
            feats[f"{prefix}_nlp_left{w}"].to_numpy(),
            feats[f"{prefix}_nlp_right{w}"].to_numpy(),
        )
        for w in WINDOW_SIZES
    ]
    return np.stack(stacked).max(axis=0)


def compute_evidence(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Fused min-p evidence + per-scheme evidence columns (min(L,R) variant)."""
    df = df.copy()
    df["kgw_g"] = (df["kgw_green"] == 1).astype("int8")
    df["kgw_valid"] = ((df["kgw_green"] >= 0) & (df["kgw_first"] == 1)).astype("int8")
    df["unigram_valid"] = (
        (df["is_repeated_ngram_2"] == 0) & (df["is_special_token"] == 0)
    ).astype("int8")

    raw = {
        "unigram": green_window_features(df, "unigram_green", 0.5, "uni", "unigram_valid"),
        "kgw": green_window_features(df, "kgw_g", 0.25, "kgw", "kgw_valid"),
        "gumbel": score_window_features(df, "gumbel_score", "gum", 1.0, "gumbel_first"),
        "textseal": score_window_features(df, "textseal_score", "ts", 0.5, "textseal_first"),
    }
    prefixes = {"unigram": "uni", "kgw": "kgw", "gumbel": "gum", "textseal": "ts"}
    per_scheme = {s: _min_left_right(F, prefixes[s]) for s, F in raw.items()}
    ev_df = pd.DataFrame(per_scheme, index=df.index)
    fused = ev_df.to_numpy().max(axis=1)
    return fused, ev_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-evidence", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(CACHE_DIR / f"{args.split}_signals.parquet")
    fused, ev_df = compute_evidence(df)

    if args.save_evidence:
        ev_df.assign(fused=fused).to_parquet(CACHE_DIR / f"{args.split}_evidence.parquet")
        print(f"evidence saved to cache/{args.split}_evidence.parquet")

    if "label" in df.columns:
        y = df["label"].to_numpy()
        print(f"{args.split}: fused    TPR@0.1%FPR={tpr_at_fpr(y, fused):.4f}  TPR@1%FPR={tpr_at_fpr(y, fused, 0.01):.4f}")
        for scheme in ev_df.columns:
            e = ev_df[scheme].to_numpy()
            print(f"{args.split}: {scheme:9s}TPR@0.1%FPR={tpr_at_fpr(y, e):.4f}  TPR@1%FPR={tpr_at_fpr(y, e, 0.01):.4f}")

    if args.out is not None:
        scores = fused / (fused + SQUASH)
        docs = df["document_id"].to_numpy()
        args.out.parent.mkdir(exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for doc_id in pd.unique(docs):
                sc = scores[docs == doc_id]
                fh.write(json.dumps(
                    {"document_id": str(doc_id), "scores": [round(float(x), 6) for x in sc]},
                    separators=(",", ":"),
                ) + "\n")
        print(f"submission written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build margin / loser-suppression (S3) channel from top-48 candidate sets.

Mechanism: under Gumbel-max/TextSeal the generated token is
c = argmax_w u_w^(1/p_w) over candidates w. Reconstruction per position:
u_w for all top-48 candidate ids under each PRF key (bit-exact port in
detectors/gumbel_textseal.py), p'_w = topk_probs renormalized over the
candidate set (T=1.0 softmax mass; missing tail mass treated as absent).
Effective value eff_w = log(u_w) / p'_w (monotone transform of u_w^(1/p'_w)).

Per-token observables (winner c = actual next token):
- margin  = eff_c - max_{w != c} eff_w
  Watermarked: c is the true argmax -> margin >= 0 (modulo top-48 truncation,
  float32 generator numerics, and the generator's temperature != 1.0).
  Clean: u independent of c -> margin usually strongly negative.
- s3lit   = sum_{w != c} p'_w * (-log b_w),  b_w = u_c^(p'_w/p'_c)
  Literal researcher formula ("p-weighted loser-LLR"). Depends only on u_c
  and the candidate distribution (algebraically (-log u_c)/p'_c * sum p'_w^2).
- s3sup   = sum_{w != c} p'_w * [ I(u_w <= b_w) * (-log b_w)
                                  - I(u_w >  b_w) * (-log(1 - b_w)) ]
  Suppression-checked variant: b_w is the bound losers must satisfy given c
  won (u_w^(1/p_w) <= u_c^(1/p_c) <=> u_w <= b_w). Under watermark all losers
  are suppressed -> every term positive; under clean violations contribute
  negative anti-evidence. Guarantees wm HIGHER than clean by construction.
- margin09 = margin recomputed with p at generation temperature 0.9
  (p09 propto p'^(1/0.9)); ablation only — canonical variant is T=1.0 (spec).

Tie handling (PRF u-resolution 8191 -> candidate u collisions ~14%/row):
margin is the direct difference eff_c - max losers; an exact effective-value
tie between winner and best loser yields margin == 0.0 (deterministic; tied
losers are NOT excluded). Suppression test uses log(u_w) <= (p_w/p_c)log(u_c)
with <= (boundary tie counts as suppressed). u clipped to >= 0.5/8191 (half a
PRF quantum) so u=0 stays finite; log terms clipped to [0, 60].

NaN positions: pos < ngram+1 (no context), dedup non-first
(gumbel_first==0 / textseal_first==0), winner not in top-48 (~2%),
winner prob <= 0, or no valid loser.

Outputs cache/margin_{split}.npz with keys:
  margin_gumbel, s3lit_gumbel, s3sup_gumbel, margin09_gumbel,
  margin_tsa, margin_tsb, margin09_tsa, margin09_tsb,
  s3lit_ts_mean, s3sup_ts_mean          (MEAN over keys A/B, not max)
aligned to cache/{split}_signals.parquet rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors.gumbel_textseal import (  # noqa: E402
    GUMBEL_KEY,
    GUMBEL_NGRAM,
    TEXTSEAL_KEY_A,
    TEXTSEAL_KEY_B,
    TEXTSEAL_NGRAM,
    _contexts,
    prf_uniform_batch,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA = ROOT / "data"

U_FLOOR = 0.5 / 8191.0  # half a PRF quantum: keeps log(u) finite at u == 0
P_FLOOR = 1e-12         # winner/candidate prob floor (float32 softmax underflow)
LOG_CAP = 60.0          # cap on per-loser log terms
GEN_TEMP = 0.9          # generation temperature for the margin09 ablation


def doc_channel(
    ids: np.ndarray,
    cand_ids: np.ndarray,
    cand_probs: np.ndarray,
    first: np.ndarray,
    key: int,
    ngram: int,
) -> dict[str, np.ndarray]:
    """Per-doc margin/s3 arrays (len(ids),) for one PRF key; NaN = no value."""
    n = len(ids)
    out = {k: np.full(n, np.nan) for k in ("margin", "s3lit", "s3sup", "margin09")}
    if n <= ngram + 1:
        return out
    pos, ctx, tok = _contexts(ids, ngram)
    K = cand_ids.shape[1]

    u = prf_uniform_batch(
        np.repeat(ctx, K, axis=0), cand_ids[pos].ravel().astype(np.int64), key
    ).reshape(len(pos), K)
    u = np.clip(u, U_FLOOR, 1.0)
    logu = np.log(u)

    cids = cand_ids[pos]
    probs = cand_probs[pos].astype(np.float64)
    valid = (cids >= 0) & (probs > 0)
    p = np.where(valid, probs, 0.0)
    psum = p.sum(axis=1, keepdims=True)
    p = p / np.where(psum > 0, psum, 1.0)

    win = (cids == tok[:, None]) & valid
    rows = np.arange(len(pos))
    wi = win.argmax(axis=1)  # first match if (theoretically) duplicated ids
    ok = (
        (first[pos] == 1)
        & win.any(axis=1)
        & (psum[:, 0] > 0)
        & (valid.sum(axis=1) >= 2)  # at least one loser
    )

    p_c = np.clip(p[rows, wi], P_FLOOR, 1.0)
    logu_c = logu[rows, wi]

    def margin_of(pmat: np.ndarray, pc: np.ndarray) -> np.ndarray:
        eff = np.where(valid, logu / np.clip(pmat, P_FLOOR, None), -np.inf)
        eff_c = logu_c / pc
        eff[rows, wi] = -np.inf
        return eff_c - eff.max(axis=1)

    out["margin"][pos[ok]] = margin_of(p, p_c)[ok]

    # generation-temperature ablation: p09 propto p^(1/0.9), renormalized
    p09 = np.where(valid, p ** (1.0 / GEN_TEMP), 0.0)
    p09 /= np.where(p09.sum(axis=1, keepdims=True) > 0, p09.sum(axis=1, keepdims=True), 1.0)
    p09_c = np.clip(p09[rows, wi], P_FLOOR, 1.0)
    out["margin09"][pos[ok]] = margin_of(p09, p09_c)[ok]

    # loser bounds: b_w = u_c^(p_w/p_c) -> log b_w = (p_w/p_c) * log u_c
    x = (p / p_c[:, None]) * logu_c[:, None]  # log b_w, <= 0
    loser = valid.copy()
    loser[rows, wi] = False
    neg_log_b = np.clip(-x, 0.0, LOG_CAP)
    out["s3lit"][pos[ok]] = np.where(loser, p * neg_log_b, 0.0).sum(axis=1)[ok]

    suppressed = logu <= x  # u_w <= b_w, tie counts as suppressed
    with np.errstate(divide="ignore", invalid="ignore"):
        neg_log_1mb = np.clip(-np.log(-np.expm1(x)), 0.0, LOG_CAP)
    neg_log_1mb = np.nan_to_num(neg_log_1mb, nan=LOG_CAP, posinf=LOG_CAP)
    contrib = np.where(suppressed, neg_log_b, -neg_log_1mb)
    out["s3sup"][pos[ok]] = np.where(loser, p * contrib, 0.0).sum(axis=1)[ok]
    return out


def build(split: str) -> None:
    df = pd.read_parquet(CACHE / f"{split}_signals.parquet")
    top = np.load(CACHE / f"topk_{split}.npz")
    lengths, topk_ids, topk_probs = top["lengths"], top["topk_ids"], top["topk_probs"]
    assert int(lengths.sum()) == len(df), f"{split}: topk/parquet row mismatch"

    docs = []
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            docs.append(np.asarray(json.loads(line)["token_ids"], dtype=np.int64))
    assert np.array_equal(np.array([len(d) for d in docs]), lengths)

    g_first = df["gumbel_first"].to_numpy()
    t_first = df["textseal_first"].to_numpy()
    tok_col = df["token_id"].to_numpy()

    n = len(df)
    fields = [
        "margin_gumbel", "s3lit_gumbel", "s3sup_gumbel", "margin09_gumbel",
        "margin_tsa", "margin_tsb", "margin09_tsa", "margin09_tsb",
        "s3lit_ts_mean", "s3sup_ts_mean",
    ]
    arr = {k: np.full(n, np.nan, dtype=np.float32) for k in fields}

    off = 0
    for ids in docs:
        sl = slice(off, off + len(ids))
        assert np.array_equal(tok_col[sl], ids), "token_id misalignment"
        ci, cp = topk_ids[sl], topk_probs[sl]

        g = doc_channel(ids, ci, cp, g_first[sl], GUMBEL_KEY, GUMBEL_NGRAM)
        arr["margin_gumbel"][sl] = g["margin"]
        arr["margin09_gumbel"][sl] = g["margin09"]
        arr["s3lit_gumbel"][sl] = g["s3lit"]
        arr["s3sup_gumbel"][sl] = g["s3sup"]

        ta = doc_channel(ids, ci, cp, t_first[sl], TEXTSEAL_KEY_A, TEXTSEAL_NGRAM)
        tb = doc_channel(ids, ci, cp, t_first[sl], TEXTSEAL_KEY_B, TEXTSEAL_NGRAM)
        arr["margin_tsa"][sl] = ta["margin"]
        arr["margin_tsb"][sl] = tb["margin"]
        arr["margin09_tsa"][sl] = ta["margin09"]
        arr["margin09_tsb"][sl] = tb["margin09"]
        arr["s3lit_ts_mean"][sl] = (ta["s3lit"] + tb["s3lit"]) / 2.0
        arr["s3sup_ts_mean"][sl] = (ta["s3sup"] + tb["s3sup"]) / 2.0
        off += len(ids)

    out = CACHE / f"margin_{split}.npz"
    np.savez_compressed(out, **arr)
    n_g = np.isfinite(arr["margin_gumbel"]).sum()
    n_t = np.isfinite(arr["margin_tsa"]).sum()
    print(f"{split}: wrote {out.name}  rows={n}  "
          f"gumbel valid={n_g} ({n_g / (g_first == 1).sum():.3f} of first)  "
          f"ts valid={n_t} ({n_t / (t_first == 1).sum():.3f} of first)")


if __name__ == "__main__":
    for split in ("train", "validation"):
        build(split)

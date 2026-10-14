"""Gumbel-Max & TextSeal per-token scores — numpy port of textseal PRF.

Exactly mirrors external/textseal/textseal/watermarking/core.py:
    w_weighted = sum(ctx[i] * PRIMES[i])                    (int64, wrapping)
    h = (w_weighted + P2*token + P3*key) * P4
    h = h * MIXING_PRIME;  h = h ^ (h >> 13);  u = (h mod 8191) / 8191
Pure integer arithmetic -> deterministic, device-independent (unlike KGW).
numpy int64 reproduces torch int64 wraparound + arithmetic shift semantics.

Scoring (detector.py): token at position pos (>= ngram+1) is scored with
context ids[pos-ngram:pos]. Gumbel: score = -log(1-u), ngram=2.
TextSeal: fused = alpha*(-log(1-u_a)) + (1-alpha)*(-log(1-u_b)), ngram=3.
scoring_method v1 dedups by context window tuple (first occurrence scores).

YAML params: gumbelmax secret_key=2004683203; textseal key_a=947821031,
key_b=1562881159, mixing_alpha=0.5.
"""

from __future__ import annotations

import numpy as np

_PRIMES = np.array(
    [10000019, 10000247, 10000439, 10000643, 10000747, 10000867, 10000993, 10001213, 10001357, 10001501],
    dtype=np.int64,
)
_P2 = np.int64(100000007)
_P3 = np.int64(500001713)
_P4 = np.int64(15485863)
_M = np.int64(2**13 - 1)
_MIXING_PRIME = np.int64(40499)
_MIXING_SHIFT = 13

GUMBEL_KEY = 2004683203
GUMBEL_NGRAM = 2
TEXTSEAL_KEY_A = 947821031
TEXTSEAL_KEY_B = 1562881159
TEXTSEAL_NGRAM = 3
TEXTSEAL_ALPHA = 0.5


def prf_uniform_batch(ctx: np.ndarray, tokens: np.ndarray, key: int) -> np.ndarray:
    """u in [0,1) for each (context window, token) pair.

    ctx: (n, ngram) int64, tokens: (n,) int64.
    """
    with np.errstate(over="ignore"):
        w_weighted = (ctx * _PRIMES[: ctx.shape[1]]).sum(axis=1, dtype=np.int64)
        h = (w_weighted + _P2 * tokens.astype(np.int64) + _P3 * np.int64(key)) * _P4
        h = h * _MIXING_PRIME
        h = h ^ (h >> _MIXING_SHIFT)
        h = h % _M  # numpy floored mod == torch remainder: non-negative
    # repo casts to float32 before dividing
    return (h.astype(np.float32) / np.float32(_M)).astype(np.float64)


def _contexts(ids: np.ndarray, ngram: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions, ctx matrix, target tokens) for pos in [ngram+1, len)."""
    start = ngram + 1  # detector.py: start_pos = ngram + 1
    pos = np.arange(start, len(ids))
    ctx = np.stack([ids[pos - ngram + k] for k in range(ngram)], axis=1)
    return pos, ctx, ids[pos]


def gumbel_scores(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-token Gumbel score (-log(1-u)) and v1 first-occurrence flag.

    Returns (score, first) arrays of len(token_ids); unscored positions NaN/0.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    score = np.full(len(ids), np.nan, dtype=np.float32)
    first = np.zeros(len(ids), dtype=np.int8)
    if len(ids) <= GUMBEL_NGRAM + 1:
        return score, first
    pos, ctx, tok = _contexts(ids, GUMBEL_NGRAM)
    u = prf_uniform_batch(ctx, tok, GUMBEL_KEY)
    score[pos] = -np.log1p(-u).astype(np.float32)
    seen: set = set()
    for p, c in zip(pos, map(tuple, ctx)):
        if c not in seen:
            seen.add(c)
            first[p] = 1
    return score, first


def textseal_scores(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-token TextSeal fused score and v1 first-occurrence flag."""
    ids = np.asarray(token_ids, dtype=np.int64)
    score = np.full(len(ids), np.nan, dtype=np.float32)
    first = np.zeros(len(ids), dtype=np.int8)
    if len(ids) <= TEXTSEAL_NGRAM + 1:
        return score, first
    pos, ctx, tok = _contexts(ids, TEXTSEAL_NGRAM)
    u_a = prf_uniform_batch(ctx, tok, TEXTSEAL_KEY_A)
    u_b = prf_uniform_batch(ctx, tok, TEXTSEAL_KEY_B)
    fused = TEXTSEAL_ALPHA * (-np.log1p(-u_a)) + (1 - TEXTSEAL_ALPHA) * (-np.log1p(-u_b))
    score[pos] = fused.astype(np.float32)
    seen: set = set()
    for p, c in zip(pos, map(tuple, ctx)):
        if c not in seen:
            seen.add(c)
            first[p] = 1
    return score, first

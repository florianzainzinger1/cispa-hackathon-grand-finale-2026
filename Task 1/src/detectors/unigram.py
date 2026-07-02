"""Unigram watermark (Zhao et al., ICLR 2024) — green mask reconstruction.

Exactly mirrors external/Unigram-Watermark/gptwm.py (GPTWatermarkBase):
    rng  = np.random.default_rng(sha256(int64(watermark_key)).digest()[:4] little-endian)
    mask = [True]*int(fraction*V) + [False]*rest, rng.shuffle
Deterministic on CPU (numpy) — no CUDA dependency for this scheme.

Params from data/watermark_config.yaml:
    fraction=0.5, strength=1.0, watermark_key=1873092841
vocab_size is NOT in the YAML. Data contains token ids up to 151909, so the
model-config vocab size 152064 is the primary candidate; the ranking-based
sanity gate on train labels decides.
"""

from __future__ import annotations

import hashlib

import numpy as np

WATERMARK_KEY = 1873092841
FRACTION = 0.5
VOCAB_SIZE_CANDIDATES = (152064,)  # model config; only candidate covering max id 151909


def _hash_fn(x: int) -> int:
    x = np.int64(x)
    return int.from_bytes(hashlib.sha256(x).digest()[:4], "little")


def green_mask(vocab_size: int, watermark_key: int = WATERMARK_KEY, fraction: float = FRACTION) -> np.ndarray:
    rng = np.random.default_rng(_hash_fn(watermark_key))
    n_green = int(fraction * vocab_size)
    mask = np.array([True] * n_green + [False] * (vocab_size - n_green))
    rng.shuffle(mask)
    return mask


def green_indicator(token_ids: list[int], mask: np.ndarray) -> np.ndarray:
    """0/1 per token: is the token in the fixed global green list."""
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.max() >= len(mask):
        raise ValueError(f"token id {ids.max()} >= vocab_size {len(mask)}")
    return mask[ids].astype(np.int8)

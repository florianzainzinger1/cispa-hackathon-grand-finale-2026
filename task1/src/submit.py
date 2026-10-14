"""Submit a scores JSONL to the leaderboard.

Reads API_KEY from the repo-root .env (never hardcode the key).
Usage: python src/submit.py --file submissions/sub_001_random_baseline.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

BASE_URL = "http://<SUBMIT_SERVER>:80"
TASK_ID = "30-watermark-localization"
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def read_api_key() -> str:
    if not ENV_PATH.is_file():
        sys.exit(f"missing {ENV_PATH} — put API_KEY=\"...\" there")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"no API_KEY found in {ENV_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()

    if not args.file.is_file():
        sys.exit(f"file not found: {args.file}")

    api_key = read_api_key()
    with args.file.open("rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": api_key},
            files={"file": (args.file.name, f, "jsonl")},
            timeout=(10, 120),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    print(f"HTTP {resp.status_code}")
    print("Server response:", body)
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

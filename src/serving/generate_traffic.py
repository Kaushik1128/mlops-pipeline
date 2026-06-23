"""Send a stream of prediction requests to the fraud API.

Useful for demos and for watching the Grafana dashboard move. Draws a realistic
mix of legit transactions with an occasional fraud, from the test set.

Usage:
    python -m src.serving.generate_traffic                 # ~30s of traffic
    python -m src.serving.generate_traffic --seconds 60 --rps 10
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.parquet"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def _payload(row: pd.Series) -> dict:
    p = {c: float(row[c]) for c in V_COLUMNS}
    p["amount"] = float(np.expm1(row["log_amount"]))
    return {"transaction": p}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:3000/predict")
    parser.add_argument("--seconds", type=int, default=30, help="How long to send traffic.")
    parser.add_argument("--rps", type=float, default=8.0, help="Requests per second.")
    parser.add_argument("--fraud-frac", type=float, default=0.15,
                        help="Fraction of requests drawn from fraud cases.")
    args = parser.parse_args()

    df = pd.read_parquet(TEST_FILE)
    frauds = df[df["Class"] == 1].reset_index(drop=True)
    legits = df[df["Class"] == 0].reset_index(drop=True)

    delay = 1.0 / args.rps
    deadline = time.time() + args.seconds
    sent = flagged = 0
    with httpx.Client(timeout=10) as cli:
        while time.time() < deadline:
            pool = frauds if random.random() < args.fraud_frac else legits
            row = pool.iloc[random.randrange(len(pool))]
            try:
                r = cli.post(args.url, json=_payload(row)).json()
                sent += 1
                flagged += int(r.get("is_fraud", False))
            except Exception as e:
                print("request failed:", e)
            time.sleep(delay)
    print(f"Sent {sent} predictions over {args.seconds}s ({flagged} flagged fraud).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

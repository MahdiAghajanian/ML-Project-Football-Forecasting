from __future__ import annotations

import argparse
import statistics
import time

import httpx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bonus prediction API latency")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--minute", type=float, default=60.0)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/predict/{args.match_id}"
    latencies_ms: list[float] = []

    with httpx.Client(timeout=20.0) as client:
        for _ in range(args.warmup):
            response = client.get(endpoint, params={"minute": args.minute})
            response.raise_for_status()

        for _ in range(args.requests):
            start = time.perf_counter()
            response = client.get(endpoint, params={"minute": args.minute})
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(latencies_ms)
    result = {
        "requests": int(len(values)),
        "mean_ms": float(values.mean()),
        "median_ms": float(statistics.median(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "under_200ms": bool(np.percentile(values, 99) < 200.0),
    }
    print(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic epsilon/threshold sensitivity study for Credal Harness."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from experiments.run_credal_harness import run_policy


CONFIGS = [
    {"label": "eps=0.00, delta=0.05", "epsilon": 0.00, "risk_threshold": 0.05},
    {"label": "eps=0.10, delta=0.05", "epsilon": 0.10, "risk_threshold": 0.05},
    {"label": "eps=0.25, delta=0.05", "epsilon": 0.25, "risk_threshold": 0.05},
    {"label": "eps=0.50, delta=0.05", "epsilon": 0.50, "risk_threshold": 0.05},
    {"label": "eps=0.10, delta=0.02", "epsilon": 0.10, "risk_threshold": 0.02},
    {"label": "eps=0.10, delta=0.10", "epsilon": 0.10, "risk_threshold": 0.10},
]


def main() -> None:
    seed_count = 50
    rows = []
    for config in CONFIGS:
        per_seed = [
            run_policy(
                "credal",
                seed,
                epsilon=config["epsilon"],
                risk_threshold=config["risk_threshold"],
                learn_credal_weights=False,
            )
            for seed in range(seed_count)
        ]
        summary = {
            key: round(statistics.mean(row[key] for row in per_seed), 4)
            for key in per_seed[0]
        }
        rows.append({**config, "summary": summary, "per_seed": per_seed})
    output = {
        "config": {"seeds": seed_count, "episodes_per_seed": 30, "horizon": 40},
        "rows": rows,
    }
    path = Path("experiments/results/sensitivity_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["label"]: row["summary"] for row in rows}, indent=2))


if __name__ == "__main__":
    main()

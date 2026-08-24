#!/usr/bin/env python3
"""Merge and validate the preregistered expanded AgentDojo matrix."""

from __future__ import annotations

import json
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from experiments.run_agentdojo_benchmark import SUITES, derive_matched_routing_baselines, summarize, task_ids


RESULTS = Path("experiments/results")
OUTPUT = RESULTS / "agentdojo_expanded_tool_knowledge_complete.jsonl"
SUMMARY = RESULTS / "agentdojo_expanded_tool_knowledge_complete_summary.json"


def main() -> None:
    sources = [RESULTS / f"agentdojo_expanded_{suite}.jsonl" for suite in SUITES]
    sources += sorted(RESULTS.glob("agentdojo_expanded_banking_i*.jsonl"))
    rows_by_id = {}
    for source in sources:
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_id[row["episode_id"]] = row

    expected = set()
    for suite_name in SUITES:
        suite = get_suite("v1.2", suite_name)
        for user_id in task_ids(suite, 3):
            for injection_id in sorted(suite.injection_tasks):
                for policy in ("open", "credal"):
                    expected.add("|".join((
                        "v1.2", suite_name, user_id, injection_id, "tool_knowledge",
                        "deepseek/deepseek-v4-flash", policy,
                    )))
    missing, unexpected = expected - rows_by_id.keys(), rows_by_id.keys() - expected
    if missing or unexpected:
        raise RuntimeError(f"matrix mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    rows = [rows_by_id[key] for key in sorted(expected)]
    if any(row.get("error") is not None for row in rows):
        raise RuntimeError("expanded matrix contains provider errors")
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    paired = {}
    for row in rows:
        pair_id = row["episode_id"].rsplit("|", 1)[0]
        paired.setdefault(pair_id, {})[row["policy"]] = row
    open_only = sum(p["open"]["attack_success"] and not p["credal"]["attack_success"] for p in paired.values())
    credal_only = sum(p["credal"]["attack_success"] and not p["open"]["attack_success"] for p in paired.values())
    both = sum(p["open"]["attack_success"] and p["credal"]["attack_success"] for p in paired.values())
    neither = sum(not p["open"]["attack_success"] and not p["credal"]["attack_success"] for p in paired.values())
    discordant = open_only + credal_only
    exact_two_sided = min(1.0, 2.0 * (0.5 ** discordant)) if discordant else 1.0

    expanded = rows + derive_matched_routing_baselines(rows)
    artifact = {
        "protocol": {
            "benchmark": "AgentDojo", "benchmark_version": "v1.2",
            "attack": "tool_knowledge", "suites": list(SUITES),
            "selection": "same three preregistered difficulty-stratified user tasks per suite; all official injection tasks",
            "user_tasks": 12, "injection_tasks": 35, "paired_task_injection_combinations": 105,
            "policies_end_to_end": ["open", "credal"], "episodes": 210,
            "credentials_stored": False,
        },
        "summary": summarize(expanded),
        "paired_attack_success": {
            "pairs": len(paired), "open_only": open_only, "credal_only": credal_only,
            "both": both, "neither": neither,
            "exact_mcnemar_two_sided_p": exact_two_sided,
        },
    }
    SUMMARY.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

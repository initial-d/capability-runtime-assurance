#!/usr/bin/env python3
"""Replay realized Open attack decisions through Credal routing offline."""

from __future__ import annotations

import json
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from experiments.run_agentdojo_benchmark import risk_score


SOURCE = Path("experiments/results/agentdojo_expanded_tool_knowledge_complete.jsonl")
OUTPUT = Path("experiments/results/agentdojo_open_success_counterfactual.json")


def credal_route(decision: dict) -> str:
    if not decision["authorized"]:
        return "deny"
    return "allow" if risk_score("credal", decision["effect"]) <= 0.05 else "confirm"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    successes = [row for row in rows if row["policy"] == "open" and row["attack_success"]]
    replayed = []
    for row in successes:
        suite = get_suite("v1.2", row["suite"])
        injection = suite.get_injection_task_by_id(row["injection_task"])
        env = suite.load_and_inject_default_environment({})
        ground_truth_calls = injection.ground_truth(env.model_copy(deep=True))
        decisions = []
        for decision in row["decisions"]:
            route = credal_route(decision)
            decisions.append({
                "tool": decision["tool"], "effect": decision["effect"],
                "authorized": decision["authorized"], "credal_route": route,
            })
        denied_writes = sum(d["effect"] != "read" and d["credal_route"] == "deny" for d in decisions)
        output_only = len(ground_truth_calls) == 0
        replayed.append({
            "source_episode_id": row["episode_id"],
            "official_attack_success": True,
            "official_attack_channel": "model_output" if output_only else "environment_state",
            "official_ground_truth_tool_calls": len(ground_truth_calls),
            "proposal_decisions": decisions,
            "counterfactual_denied_writes": denied_writes,
            "official_goal_blocked_by_routing_replay": False if output_only else None,
            "interpretation": (
                "The official predicate depends only on fixed model output; execution routing cannot "
                "establish that the goal is blocked." if output_only else
                "A stateful predicate requires replay against a fresh native environment."
            ),
        })
    artifact = {
        "protocol": "fixed audited Open proposal sequence replayed through Credal routing; no hosted call",
        "source_successes": len(successes),
        "results": replayed,
        "conclusion": (
            "Both realized Open successes target an output-only injection predicate. The routing replay "
            "therefore provides no causal evidence that Credal blocks either official AgentDojo goal."
        ),
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

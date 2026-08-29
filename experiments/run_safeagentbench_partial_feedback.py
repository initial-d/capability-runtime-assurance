"""Finite-population selective-feedback audit on SafeAgentBench.

The script does not redistribute benchmark data.  It downloads two detailed-
task JSONL files from a pinned upstream commit, verifies their digests, and
uses the official safe/unsafe task labels only as follows:

* 150 safe and 150 unsafe tasks train a transparent multinomial Naive Bayes
  nominal-risk scorer;
* the remaining 300 tasks form a fixed audit population;
* labels in that population are selectively hidden, with hazardous outcomes
  less likely to be observed;
* trusted probes reveal a subset of the hidden labels.

For the fixed audit population, (observed harms + all unresolved labels) / n
is an exact compatible upper bound, not an asymptotic population statement.
The held-out labels are used only after routing to audit actual harm.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable


PROTOCOL = {
    "name": "safeagentbench-partial-feedback-v1",
    "upstream_commit": "38ca3ab27eb8a5f5034a50bdcc5cbab23ce8f089",
    "files": {
        "safe_detailed_1009.jsonl": "891368d2dac63ba2587e56eec8f02f45a68c4fa8fa2cc02629951dd5f00b58ca",
        "unsafe_detailed_1009.jsonl": "3ac0f9a3ec3ff80ffdc0870e102e15aca7f7ee2092cebde12be7096d1ade4500",
    },
    "formal_seeds": list(range(10_000, 10_100)),
    "training_per_class": 150,
    "audit_per_class": 150,
    "score_strata": 4,
    "automatic_threshold": 0.20,
    "observation_probability": {"safe": 0.95, "harm": 0.35},
    "probe_budgets": [0.0, 0.25, 0.50, 0.75, 0.90, 0.98, 1.0],
}

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "cache" / "safeagentbench"
RESULT = ROOT / "experiments" / "results" / "safeagentbench_partial_feedback.json"


def _protocol_hash() -> str:
    payload = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _download(name: str, digest: str) -> Path:
    path = CACHE / name
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        commit = PROTOCOL["upstream_commit"]
        url = f"https://raw.githubusercontent.com/shengyin1224/SafeAgentBench/{commit}/dataset/{name}"
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"digest mismatch for {name}: {actual}")
    return path


def _load() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, label in (("safe_detailed_1009.jsonl", 0), ("unsafe_detailed_1009.jsonl", 1)):
        path = _download(name, PROTOCOL["files"][name])
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows.append({"instruction": row["instruction"], "harm": label})
    if len(rows) != 600:
        raise RuntimeError(f"expected 600 detailed tasks, found {len(rows)}")
    return rows


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return words + [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]


def _train(rows: list[dict[str, object]], indices: Iterable[int]) -> tuple:
    counts = [Counter(), Counter()]
    totals = [0, 0]
    class_counts = [0, 0]
    for index in indices:
        label = int(rows[index]["harm"])
        tokens = _tokens(str(rows[index]["instruction"]))
        counts[label].update(tokens)
        totals[label] += len(tokens)
        class_counts[label] += 1
    vocabulary = len(set(counts[0]) | set(counts[1]))
    return counts, totals, class_counts, vocabulary


def _score(model: tuple, text: str) -> float:
    counts, totals, class_counts, vocabulary = model
    logits = []
    for label in (0, 1):
        value = math.log((class_counts[label] + 1) / (sum(class_counts) + 2))
        for token in _tokens(text):
            value += math.log((counts[label][token] + 0.5) / (totals[label] + 0.5 * vocabulary))
        logits.append(value)
    difference = min(30.0, max(-30.0, logits[1] - logits[0]))
    return 1.0 / (1.0 + math.exp(-difference))


def _stratify(scores: dict[int, float], strata: int) -> list[list[int]]:
    ordered = sorted(scores, key=lambda index: (scores[index], index))
    groups = [[] for _ in range(strata)]
    for rank, index in enumerate(ordered):
        groups[min(strata - 1, rank * strata // len(ordered))].append(index)
    return groups


def _compatible_upper(harm: int, unresolved: int, total: int) -> float:
    return (harm + unresolved) / total


def _allocate_targeted(
    rows: list[dict[str, object]],
    groups: list[list[int]],
    observed: list[list[int]],
    missing: list[list[int]],
    budget: int,
    threshold: float,
) -> list[list[int]]:
    revealed = [[] for _ in groups]
    frozen_order = []
    for group, indices in enumerate(groups):
        harm = sum(int(rows[index]["harm"]) for index in observed[group])
        unresolved = len(missing[group])
        total = len(indices)
        optimistic_is_certifiable = harm / total <= threshold
        optimistic_safe_probes = max(0, math.ceil(harm + unresolved - threshold * total - 1e-12))
        frozen_order.append((optimistic_safe_probes / total, group, optimistic_is_certifiable))

    for _, group, certifiable in sorted(frozen_order):
        if not certifiable:
            continue
        while len(revealed[group]) < len(missing[group]) and budget > 0:
            index = missing[group][len(revealed[group])]
            revealed[group].append(index)
            budget -= 1
            known = observed[group] + revealed[group]
            harm = sum(int(rows[item]["harm"]) for item in known)
            unresolved = len(missing[group]) - len(revealed[group])
            if _compatible_upper(harm, unresolved, len(groups[group])) <= threshold:
                break
    return revealed


def _allocate_uniform(missing: list[list[int]], budget: int, rng: random.Random) -> list[list[int]]:
    revealed = [[] for _ in missing]
    pool = [(group, index) for group, indices in enumerate(missing) for index in indices]
    rng.shuffle(pool)
    for group, index in pool[:budget]:
        revealed[group].append(index)
    return revealed


def _route_groups(
    rows: list[dict[str, object]],
    groups: list[list[int]],
    observed: list[list[int]],
    missing: list[list[int]],
    revealed: list[list[int]],
    threshold: float,
    mode: str,
) -> list[int]:
    selected: list[int] = []
    for group, indices in enumerate(groups):
        known = observed[group] + revealed[group]
        harm = sum(int(rows[index]["harm"]) for index in known)
        unresolved = len(missing[group]) - len(revealed[group])
        if mode == "credal":
            risk = _compatible_upper(harm, unresolved, len(indices))
        elif mode == "complete_case":
            risk = harm / len(known) if known else 1.0
        elif mode == "oracle":
            risk = sum(int(rows[index]["harm"]) for index in indices) / len(indices)
        else:
            raise ValueError(mode)
        if risk <= threshold:
            selected.extend(indices)
    return selected


def _metrics(rows: list[dict[str, object]], selected: list[int], audit_size: int) -> dict[str, float]:
    harms = sum(int(rows[index]["harm"]) for index in selected)
    coverage = len(selected) / audit_size
    conditional_harm = harms / len(selected) if selected else 0.0
    return {
        "automatic_coverage": coverage,
        "harm_mass": harms / audit_size,
        "conditional_harm": conditional_harm,
        "threshold_violation": float(conditional_harm > float(PROTOCOL["automatic_threshold"]) + 1e-12),
        "safe_automatic_utility": (len(selected) - harms) / audit_size,
    }


def _one_run(rows: list[dict[str, object]], seed: int, budget_fraction: float) -> list[dict[str, object]]:
    rng = random.Random(seed)
    by_class = {
        label: [index for index, row in enumerate(rows) if int(row["harm"]) == label]
        for label in (0, 1)
    }
    for indices in by_class.values():
        rng.shuffle(indices)
    n_train = int(PROTOCOL["training_per_class"])
    training = by_class[0][:n_train] + by_class[1][:n_train]
    audit = by_class[0][n_train:] + by_class[1][n_train:]
    model = _train(rows, training)
    scores = {index: _score(model, str(rows[index]["instruction"])) for index in audit}
    groups = _stratify(scores, int(PROTOCOL["score_strata"]))

    observed: list[list[int]] = []
    missing: list[list[int]] = []
    probabilities = PROTOCOL["observation_probability"]
    for indices in groups:
        group_observed, group_missing = [], []
        for index in indices:
            probability = probabilities["harm"] if int(rows[index]["harm"]) else probabilities["safe"]
            (group_observed if rng.random() < probability else group_missing).append(index)
        rng.shuffle(group_missing)
        observed.append(group_observed)
        missing.append(group_missing)

    initial_missing = sum(len(indices) for indices in missing)
    budget = int(math.floor(budget_fraction * initial_missing))
    threshold = float(PROTOCOL["automatic_threshold"])
    targeted = _allocate_targeted(rows, groups, observed, missing, budget, threshold)
    uniform = _allocate_uniform(missing, budget, random.Random(seed + 1_000_000))
    empty = [[] for _ in groups]

    policy_selections = {
        "nominal_point": [index for index in audit if scores[index] <= threshold],
        "complete_case": _route_groups(rows, groups, observed, missing, empty, threshold, "complete_case"),
        "passive_credal": _route_groups(rows, groups, observed, missing, empty, threshold, "credal"),
        "uniform_credal": _route_groups(rows, groups, observed, missing, uniform, threshold, "credal"),
        "active_complete_case": _route_groups(rows, groups, observed, missing, targeted, threshold, "complete_case"),
        "targeted_credal": _route_groups(rows, groups, observed, missing, targeted, threshold, "credal"),
        "oracle_group": _route_groups(rows, groups, observed, missing, empty, threshold, "oracle"),
        "deny_all": [],
    }
    records = []
    for policy, selected in policy_selections.items():
        metrics = _metrics(rows, selected, len(audit))
        metrics.update(
            {
                "seed": seed,
                "budget_fraction_of_missing": budget_fraction,
                "policy": policy,
                "initial_missing": initial_missing,
                "probe_count": (
                    sum(len(items) for items in targeted)
                    if policy in {"targeted_credal", "active_complete_case"}
                    else sum(len(items) for items in uniform)
                    if policy == "uniform_credal"
                    else 0
                ),
            }
        )
        metrics["probe_fraction_of_audit"] = metrics["probe_count"] / len(audit)
        records.append(metrics)
    return records


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for record in records:
        key = (float(record["budget_fraction_of_missing"]), str(record["policy"]))
        grouped.setdefault(key, []).append(record)
    summary = []
    fields = [
        "automatic_coverage",
        "harm_mass",
        "conditional_harm",
        "threshold_violation",
        "safe_automatic_utility",
        "probe_fraction_of_audit",
    ]
    for (budget, policy), items in sorted(grouped.items()):
        row: dict[str, object] = {"budget_fraction_of_missing": budget, "policy": policy}
        for field in fields:
            values = [float(item[field]) for item in items]
            row[field] = statistics.mean(values)
            row[f"{field}_sd"] = statistics.stdev(values)
        summary.append(row)
    return summary


def main() -> None:
    rows = _load()
    records = []
    for seed in PROTOCOL["formal_seeds"]:
        for budget in PROTOCOL["probe_budgets"]:
            records.extend(_one_run(rows, int(seed), float(budget)))
    payload = {
        "protocol": PROTOCOL,
        "protocol_sha256": _protocol_hash(),
        "records": records,
        "summary": _summarize(records),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RESULT}")
    print(f"protocol_sha256={payload['protocol_sha256']}")


if __name__ == "__main__":
    main()

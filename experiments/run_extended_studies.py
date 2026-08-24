#!/usr/bin/env python3
"""Extended stress tests for selective execution and containment loss.

The experiment is deliberately separate from the stateful demonstration.  It
tests the decision rule on a parameterized population where automatic task
utility, realized harm, hypothesis disagreement, and an omitted execution
world are independently observable.  Every proposal represents a useful write;
an automatic, non-harmful live commit completes the task without handoff.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


SEEDS = 50
PROPOSALS = 10_000
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.25]
WEIGHTS = (0.94, 0.03, 0.03)
GROUP_EPSILONS = {
    "safe": 0.005,
    "moderate_disagreement": 0.02,
    "high_disagreement": 0.08,
}

# (population mass, risks under the three declared worlds, true harm rate)
CALL_TYPES = {
    "safe": (0.50, (0.005, 0.010, 0.010), 0.005),
    "moderate_disagreement": (0.25, (0.010, 0.010, 0.200), 0.015),
    "high_disagreement": (0.25, (0.010, 0.010, 0.800), 0.250),
}


def score(policy: str, risks: tuple[float, ...], epsilon: float = 0.02) -> float:
    nominal = sum(w * r for w, r in zip(WEIGHTS, risks))
    if policy == "point":
        return nominal
    if policy == "credal":
        return (1.0 - epsilon) * nominal + epsilon
    if policy == "group_penalty":
        # Non-credal control with the same learned group information.  It adds
        # the calibrated radius as an ordinary score penalty, without an
        # ambiguity-set interpretation or containment guarantee.
        return min(1.0, nominal + epsilon)
    if policy == "maxrisk":
        return max(risks)
    raise ValueError(policy)


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    half = 2.0096 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "low": mean - half, "high": mean + half}


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def selective_seed(seed: int, policy: str, threshold: float) -> dict[str, float]:
    rng = random.Random(10_000_019 + seed)
    names = list(CALL_TYPES)
    masses = [CALL_TYPES[name][0] for name in names]
    auto = harm = success = 0
    for _ in range(PROPOSALS):
        name = rng.choices(names, weights=masses, k=1)[0]
        _, risks, true_risk = CALL_TYPES[name]
        allowed = score(policy, risks, GROUP_EPSILONS[name]) <= threshold
        if not allowed:
            continue
        auto += 1
        harmed = rng.random() < true_risk
        harm += int(harmed)
        success += int(not harmed)
    return {
        "automatic_coverage": auto / PROPOSALS,
        "automatic_harm_per_proposal": harm / PROPOSALS,
        "selective_harm": harm / max(1, auto),
        "safe_automatic_task_completion": success / PROPOSALS,
    }


def selective_experiment() -> dict[str, Any]:
    rows = []
    for policy in ("point", "credal", "group_penalty", "maxrisk"):
        for threshold in THRESHOLDS:
            seeds = [selective_seed(seed, policy, threshold) for seed in range(SEEDS)]
            rows.append({
                "policy": policy,
                "threshold": threshold,
                **{key: mean_ci([row[key] for row in seeds]) for key in seeds[0]},
            })
    operating = {
        policy: next(row for row in rows if row["policy"] == policy and row["threshold"] == 0.05)
        for policy in ("point", "credal", "group_penalty", "maxrisk")
    }
    operating.update({
        "deny_all": {
            "automatic_coverage": mean_ci([0.0] * SEEDS),
            "automatic_harm_per_proposal": mean_ci([0.0] * SEEDS),
            "selective_harm": mean_ci([0.0] * SEEDS),
            "safe_automatic_task_completion": mean_ci([0.0] * SEEDS),
        },
        "confirm_all": {
            "automatic_coverage": mean_ci([0.0] * SEEDS),
            "automatic_harm_per_proposal": mean_ci([0.0] * SEEDS),
            "selective_harm": mean_ci([0.0] * SEEDS),
            "safe_automatic_task_completion": mean_ci([0.0] * SEEDS),
            "confirmation_rate": mean_ci([1.0] * SEEDS),
        },
        "sandbox_all": {
            "automatic_coverage": mean_ci([0.0] * SEEDS),
            "automatic_harm_per_proposal": mean_ci([0.0] * SEEDS),
            "selective_harm": mean_ci([0.0] * SEEDS),
            "safe_automatic_task_completion": mean_ci([0.0] * SEEDS),
            "preview_rate": mean_ci([1.0] * SEEDS),
        },
    })
    return {"rows": rows, "operating_point": operating}


def population_suite() -> dict[str, Any]:
    """Evaluate ranking robustness across randomly generated populations.

    Group radii correlate with unmodeled excess risk but are noisy. Policies
    are compared at matched target automatic coverage, so a mere affine
    threshold shift cannot count as a Credal improvement.
    """
    target_coverages = (0.25, 0.50, 0.75)
    records = []
    for seed in range(300):
        rng = random.Random(50_000_153 + seed)
        nominal_weights = [rng.gammavariate(2, 1) for _ in range(3)]
        z = sum(nominal_weights); nominal_weights = [x / z for x in nominal_weights]
        groups = []
        for _ in range(30):
            mass = rng.gammavariate(1.5, 1)
            risks = tuple(rng.betavariate(0.8, 8.0) for _ in range(3))
            radius = rng.choice((0.005, 0.02, 0.05, 0.10))
            nominal = sum(w * r for w, r in zip(nominal_weights, risks))
            # The group radius is informative but imperfect: excess risk is
            # centered on the declared radius with independent noise.
            true_risk = min(1.0, max(0.0, nominal + radius * rng.uniform(0.3, 1.5) + rng.gauss(0, 0.01)))
            groups.append((mass, risks, radius, true_risk))
        z = sum(row[0] for row in groups)
        shuffled_radii = [row[2] for row in groups]
        rng.shuffle(shuffled_radii)
        groups = [(m / z, risks, radius, shuffled, truth)
                  for (m, risks, radius, truth), shuffled in zip(groups, shuffled_radii)]

        for target in target_coverages:
            for policy in ("point", "credal", "group_penalty", "credal_shuffled", "maxrisk"):
                score_policy = "credal" if policy == "credal_shuffled" else policy
                ranked = sorted(groups, key=lambda row: score(score_policy, row[1], row[3] if policy == "credal_shuffled" else row[2]))
                selected = []
                coverage = 0.0
                for row in ranked:
                    selected.append(row); coverage += row[0]
                    if coverage >= target:
                        break
                selective_harm = sum(m * truth for m, _, _, _, truth in selected) / coverage
                records.append({"seed": seed, "target_coverage": target, "policy": policy,
                                "achieved_coverage": coverage, "selective_harm": selective_harm})
    summary = []
    for target in target_coverages:
        for policy in ("point", "credal", "group_penalty", "credal_shuffled", "maxrisk"):
            values = [row["selective_harm"] for row in records if row["target_coverage"] == target and row["policy"] == policy]
            summary.append({"target_coverage": target, "policy": policy, "selective_harm": mean_ci(values)})
    wins = {}
    comparisons = {}
    for target in target_coverages:
        by_seed = {(row["seed"], row["policy"]): row for row in records if row["target_coverage"] == target}
        wins[str(target)] = {
            "credal_better_than_point": sum(by_seed[(seed, "credal")]["selective_harm"] < by_seed[(seed, "point")]["selective_harm"] for seed in range(300)),
            "credal_better_than_shuffled": sum(by_seed[(seed, "credal")]["selective_harm"] < by_seed[(seed, "credal_shuffled")]["selective_harm"] for seed in range(300)),
            "credal_better_than_maxrisk": sum(by_seed[(seed, "credal")]["selective_harm"] < by_seed[(seed, "maxrisk")]["selective_harm"] for seed in range(300)),
        }
        comparisons[str(target)] = {}
        for baseline in ("point", "group_penalty", "credal_shuffled", "maxrisk"):
            deltas = [by_seed[(seed, "credal")]["selective_harm"] - by_seed[(seed, baseline)]["selective_harm"] for seed in range(300)]
            n_wins = sum(delta < 0 for delta in deltas)
            n_losses = sum(delta > 0 for delta in deltas)
            comparisons[str(target)][baseline] = {
                "paired_difference": mean_ci(deltas),
                "wins": n_wins,
                "losses": n_losses,
                "ties": 300 - n_wins - n_losses,
                "exact_sign_p": exact_sign_p(n_wins, n_losses),
            }
    return {"populations": 300, "records": records, "summary": summary, "wins": wins,
            "paired_comparisons": comparisons}


def coverage_loss_seed(
    seed: int,
    omitted_rate: float,
    probe_rate: float,
    detector_sensitivity: float = 0.90,
    detector_specificity: float = 0.99,
) -> dict[str, float]:
    """Leave-one-world-out stream with a sound but incomplete mismatch detector.

    The omitted world looks safe to every declared hypothesis.  A shadow probe
    reveals its contract marker without live harm; without a probe, a harmful
    live transition can reveal it only after the commit.  Once detected, the
    controller fails closed for the rest of the episode.
    """
    rng = random.Random(30_000_079 + seed + int(omitted_rate * 10_000) + int(probe_rate * 100_000))
    episodes, horizon = 250, 200
    harms = omitted = detected = pre_detection_harms = auto = 0
    delays: list[int] = []
    false_invalidations = 0
    for _ in range(episodes):
        covered = True
        first_omitted = None
        detection_at = None
        for t in range(horizon):
            is_omitted = rng.random() < omitted_rate
            omitted += int(is_omitted)
            if is_omitted and first_omitted is None:
                first_omitted = t
            if not covered:
                continue
            probe = rng.random() < probe_rate
            if probe:
                detected_marker = (
                    rng.random() < detector_sensitivity if is_omitted
                    else rng.random() > detector_specificity
                )
                if detected_marker:
                    covered = False
                    detected += 1
                    detection_at = t
                    false_invalidations += int(not is_omitted)
                continue
            auto += 1
            harmed = bool(is_omitted and rng.random() < 0.60)
            harms += int(harmed)
            pre_detection_harms += int(harmed)
            # A harmful transition exposes the omitted contract, but only
            # after the live mutation. Non-harmful omitted executions remain
            # observationally indistinguishable in this stress test.
            if harmed:
                covered = False
                detected += 1
                detection_at = t
        if first_omitted is not None and detection_at is not None:
            delays.append(detection_at - first_omitted)
    return {
        "harm_per_proposal": harms / (episodes * horizon),
        "automatic_coverage": auto / (episodes * horizon),
        "detection_rate_per_omitted": detected / max(1, omitted),
        "mean_detection_delay": statistics.mean(delays) if delays else 0.0,
        "pre_detection_harm_per_episode": pre_detection_harms / episodes,
        "false_invalidation_rate": false_invalidations / episodes,
    }


def coverage_loss_experiment() -> dict[str, Any]:
    rows = []
    for omitted_rate in (0.0, 0.01, 0.05, 0.10, 0.20, 0.50):
        for probe_rate in (0.0, 0.05, 0.10, 0.20):
            seeds = [coverage_loss_seed(seed, omitted_rate, probe_rate) for seed in range(SEEDS)]
            rows.append({
                "omitted_world_rate": omitted_rate,
                "probe_rate": probe_rate,
                **{key: mean_ci([row[key] for row in seeds]) for key in seeds[0]},
            })
    return {"rows": rows}


def calibration_certificate_study() -> dict[str, Any]:
    """Repeated-sampling audit of the simultaneous group certificate.

    The iid arm matches the proposition exactly.  The drift arm deliberately
    changes the deployment risks after calibration and is therefore a
    negative control, not a counterexample to the proposition.
    """
    trials, groups, alpha = 5_000, 4, 0.05
    sample_sizes = (50, 100, 200, 500)
    rows = []
    for n in sample_sizes:
        iid_covered: list[float] = []
        drift_covered: list[float] = []
        mean_widths: list[float] = []
        for trial in range(trials):
            rng = random.Random(71_000_003 + 10_007 * n + trial)
            truths = [rng.uniform(0.005, 0.25) for _ in range(groups)]
            # Nominal risks are intentionally misspecified in both directions.
            nominal = [min(.95, max(0.0, h + rng.uniform(-.08, .04))) for h in truths]
            uppers = []
            margin = math.sqrt(math.log(groups / alpha) / (2 * n))
            for h, rhat in zip(truths, nominal):
                harms = sum(rng.random() < h for _ in range(n))
                empirical = harms / n
                epsilon = max(0.0, (max(0.0, empirical + margin - rhat) / (1.0 - rhat)))
                epsilon = min(1.0, epsilon)
                uppers.append(rhat + epsilon * (1.0 - rhat))
            iid_covered.append(float(all(h <= u + 1e-12 for h, u in zip(truths, uppers))))
            shifted = [min(1.0, h + .05) for h in truths]
            drift_covered.append(float(all(h <= u + 1e-12 for h, u in zip(shifted, uppers))))
            mean_widths.append(statistics.mean(u - r for u, r in zip(uppers, nominal)))
        rows.append({
            "n_per_group": n,
            "iid_simultaneous_coverage": mean_ci(iid_covered),
            "post_calibration_drift_coverage": mean_ci(drift_covered),
            "mean_certificate_width": mean_ci(mean_widths),
        })
    return {"trials": trials, "groups": groups, "alpha": alpha, "rows": rows}


def sequence_budget_study() -> dict[str, Any]:
    """Monte Carlo verification of adaptive sequence-risk accounting."""
    episodes, horizon, delta = 50_000, 100, 0.10
    policies = ("per_call_only", "uniform_ledger", "adaptive_ledger")
    records = {p: {"any_harm": [], "commits": [], "charged_risk": []} for p in policies}
    for episode in range(episodes):
        for policy in policies:
            rng = random.Random(91_000_019 + episode * 17 + policies.index(policy))
            spent = 0.0
            any_harm = False
            commits = 0
            recent_safe = 0
            for t in range(horizon):
                if policy == "adaptive_ledger":
                    # Predictable risk depends on the observed history, not on
                    # the unseen next outcome; independence is not assumed by
                    # the theorem.
                    risk = (0.002, 0.005, 0.010, 0.020)[min(3, recent_safe // 4)]
                else:
                    risk = 0.010
                if policy != "per_call_only" and spent + risk > delta + 1e-12:
                    continue
                commits += 1
                spent += risk
                harmed = rng.random() < risk
                any_harm = any_harm or harmed
                recent_safe = 0 if harmed else recent_safe + 1
            records[policy]["any_harm"].append(float(any_harm))
            records[policy]["commits"].append(float(commits))
            records[policy]["charged_risk"].append(spent)
    rows = []
    for policy in policies:
        rows.append({"policy": policy, **{key: mean_ci(values) for key, values in records[policy].items()}})
    return {"episodes": episodes, "horizon": horizon, "budget": delta, "rows": rows}


def main() -> None:
    output = {
        "config": {
            "seeds": SEEDS,
            "proposals_per_seed": PROPOSALS,
            "weights": WEIGHTS,
            "group_epsilons": GROUP_EPSILONS,
            "thresholds": THRESHOLDS,
            "call_types": CALL_TYPES,
        },
        "selective": selective_experiment(),
        "population_suite": population_suite(),
        "leave_one_world_out": coverage_loss_experiment(),
        "calibration_certificate": calibration_certificate_study(),
        "sequence_budget": sequence_budget_study(),
    }
    path = Path("experiments/results/extended_studies.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "operating_point": output["selective"]["operating_point"],
        "coverage_rows": len(output["leave_one_world_out"]["rows"]),
    }, indent=2))


if __name__ == "__main__":
    main()

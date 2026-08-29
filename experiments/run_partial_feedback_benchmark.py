#!/usr/bin/env python3
"""Frozen benchmark for risk routing with selectively missing feedback.

The benchmark targets a structural property of side-effectful agent logs:
denied, failed, or externally suppressed calls need not reveal the outcome that
would have followed a live commit.  Outcome-dependent observation makes point
completion and complete-case upper bounds unsound.  Credal policies retain
every binary completion of the missing mass.  The active variant receives the
same trusted shadow outcomes as the active point baselines and differs only in
whether the remaining unidentified mass is represented.

The controlled generator exposes ground-truth stratum risks solely for audit.
Every generator constant, seed, baseline, probe rate, and metric is declared in
``PROTOCOL`` and hashed into the result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from credal_harness.partial_identification import (
    PartialFeedbackCounts,
    PartialFeedbackCredalCalibrator,
)


PROTOCOL: dict[str, Any] = {
    "version": "partial-feedback-v1",
    "seeds": 50,
    "strata": 120,
    "calibration_per_stratum": 20_000,
    "familywise_alpha": 0.05,
    "adaptive_checkpoints_per_stratum": 16,
    "automatic_risk_threshold": 0.05,
    "active_probe_fraction": 0.50,
    "probe_sensitivity": [0.0, 0.25, 0.50, 0.75, 0.90, 0.98],
    "missingness_regimes": ["mcar", "selective", "severe_selective"],
    "risk_class_probabilities": [0.70, 0.20, 0.10],
    "risk_class_intervals": [[0.003, 0.018], [0.035, 0.070], [0.120, 0.300]],
    "additional_blind_spot_probability": 0.08,
    "policies": [
        "nominal_point",
        "complete_case",
        "nominal_imputation",
        "observed_ucb",
        "max_nominal_observed_ucb",
        "passive_credal",
        "uniform_active_credal",
        "active_complete_case",
        "active_observed_ucb",
        "active_credal",
        "oracle",
        "deny_all",
    ],
}


def _observation_probabilities(
    regime: str, blind: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return observation probabilities conditional on harm and safety."""
    if regime == "mcar":
        return np.full(len(blind), 0.97), np.full(len(blind), 0.97)
    if regime == "selective":
        return np.where(blind, 0.08, 0.72), np.where(blind, 0.975, 0.985)
    if regime == "severe_selective":
        return np.where(blind, 0.02, 0.45), np.where(blind, 0.960, 0.980)
    raise ValueError(f"unknown missingness regime: {regime}")


def _upper_observed_harm(observed: int, harms: int, alpha: float) -> float:
    if observed <= 0:
        return 1.0
    margin = math.sqrt(math.log(2.0 / alpha) / (2.0 * observed))
    return min(1.0, harms / observed + margin)


def _safe_reveals_needed(
    calibrator: PartialFeedbackCredalCalibrator,
    harm: int,
    safe: int,
    missing: int,
    threshold: float,
) -> int | None:
    """Optimistic probe count required to certify a stratum.

    The planner uses this only to allocate evidence acquisition.  Actual
    probes may reveal harm, in which case the requirement is recomputed.  A
    ``None`` result means that even revealing every missing outcome as safe
    would not certify the stratum with the available sample size.
    """
    current = PartialFeedbackCounts(harm, safe, missing)
    if calibrator.certificate(current).confidence_upper <= threshold:
        return 0
    if missing == 0:
        return None
    optimistic = PartialFeedbackCounts(harm, safe + missing, 0)
    if calibrator.certificate(optimistic).confidence_upper > threshold:
        return None
    low, high = 1, missing
    while low < high:
        middle = (low + high) // 2
        candidate = PartialFeedbackCounts(harm, safe + middle, missing - middle)
        if calibrator.certificate(candidate).confidence_upper <= threshold:
            high = middle
        else:
            low = middle + 1
    return low


def _targeted_probe(
    rng: np.random.Generator,
    observed_harm: np.ndarray,
    observed_safe: np.ndarray,
    missing_harm: np.ndarray,
    missing_safe: np.ndarray,
    nominal_risk: np.ndarray,
    deployment_mass: np.ndarray,
    *,
    budget: int,
    calibrator: PartialFeedbackCredalCalibrator,
    threshold: float,
    max_updates_per_stratum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Greedily contract high-value credal sets without outcome leakage.

    Priority is deployment mass divided by a pre-outcome estimate of the probes
    needed for certification.  The allocation order is frozen before any new
    label is revealed.  A hypergeometric draw then exposes the trusted random
    subset, so hidden outcomes never steer which stratum is selected next.
    """
    harm = observed_harm.copy()
    safe = observed_safe.copy()
    remaining_harm = missing_harm.copy()
    remaining_safe = missing_safe.copy()
    missing = remaining_harm + remaining_safe
    used = 0
    updates = np.zeros(len(harm), dtype=int)
    candidates: list[tuple[float, int]] = []
    for index in range(len(harm)):
        needed = _safe_reveals_needed(
            calibrator,
            int(harm[index]),
            int(safe[index]),
            int(missing[index]),
            threshold,
        )
        if needed is None or needed <= 0:
            continue
        candidates.append((needed / max(1e-12, deployment_mass[index]), index))
    for _, index in sorted(candidates):
        while budget > 0 and updates[index] < max_updates_per_stratum:
            needed = _safe_reveals_needed(
                calibrator,
                int(harm[index]),
                int(safe[index]),
                int(missing[index]),
                threshold,
            )
            if needed is None or needed == 0 or needed > budget:
                break
            observed = harm[index] + safe[index]
            observed_rate = harm[index] / observed if observed else 1.0
            estimated_harm = min(0.95, max(float(nominal_risk[index]), observed_rate))
            reveal = min(
                int(missing[index]),
                budget,
                max(needed, int(math.ceil(needed / max(0.05, 1.0 - estimated_harm)))),
            )
            revealed_harm = int(rng.hypergeometric(
                int(remaining_harm[index]), int(remaining_safe[index]), reveal
            ))
            revealed_safe = reveal - revealed_harm
            harm[index] += revealed_harm
            safe[index] += revealed_safe
            remaining_harm[index] -= revealed_harm
            remaining_safe[index] -= revealed_safe
            missing[index] -= reveal
            budget -= reveal
            used += reveal
            updates[index] += 1
    return harm, safe, missing, used, int(updates.max()) if len(updates) else 0


def generate_seed(seed: int, regime: str, probe_fraction: float, *, quick: bool) -> dict[str, Any]:
    rng = np.random.default_rng(81_731_009 + seed)
    strata = 36 if quick else int(PROTOCOL["strata"])
    per_stratum = 4_000 if quick else int(PROTOCOL["calibration_per_stratum"])
    class_probability = np.asarray(PROTOCOL["risk_class_probabilities"], dtype=float)
    risk_class = rng.choice(3, size=strata, p=class_probability)
    true_risk = np.empty(strata)
    for index, (low, high) in enumerate(PROTOCOL["risk_class_intervals"]):
        selected = risk_class == index
        true_risk[selected] = rng.uniform(low, high, int(selected.sum()))

    blind = (risk_class == 2) | (
        rng.random(strata) < float(PROTOCOL["additional_blind_spot_probability"])
    )
    nominal = np.clip(true_risk + rng.normal(0.0, 0.008, strata), 0.0, 1.0)
    nominal[blind] = np.clip(
        0.12 * true_risk[blind] + rng.normal(0.0, 0.003, int(blind.sum())),
        0.0,
        1.0,
    )
    observe_harm, observe_safe = _observation_probabilities(regime, blind)

    category_counts = []
    for index in range(strata):
        probabilities = np.asarray([
            true_risk[index] * observe_harm[index],
            (1.0 - true_risk[index]) * observe_safe[index],
            true_risk[index] * (1.0 - observe_harm[index]),
            (1.0 - true_risk[index]) * (1.0 - observe_safe[index]),
        ])
        category_counts.append(rng.multinomial(per_stratum, probabilities))
    counts = np.asarray(category_counts, dtype=int)
    observed_harm, observed_safe, missing_harm, missing_safe = counts.T

    # Deployment mass is sampled independently of labels and is visible to the
    # evidence planner as an estimated workload frequency.
    deployment_mass = rng.dirichlet(np.full(strata, 3.0))
    revealed_harm = rng.binomial(missing_harm, probe_fraction)
    revealed_safe = rng.binomial(missing_safe, probe_fraction)

    observed = observed_harm + observed_safe
    uniform_harm = observed_harm + revealed_harm
    uniform_safe = observed_safe + revealed_safe
    uniform_observed = uniform_harm + uniform_safe
    missing = missing_harm + missing_safe
    uniform_missing = missing - revealed_harm - revealed_safe

    alpha_per_stratum = float(PROTOCOL["familywise_alpha"]) / strata
    active_checkpoints = int(PROTOCOL["adaptive_checkpoints_per_stratum"])
    calibrator = PartialFeedbackCredalCalibrator(alpha=alpha_per_stratum)
    active_calibrator = PartialFeedbackCredalCalibrator(
        alpha=alpha_per_stratum / active_checkpoints
    )
    threshold = float(PROTOCOL["automatic_risk_threshold"])
    active_harm, active_safe, active_missing, targeted_used, max_updates_used = _targeted_probe(
        rng,
        observed_harm,
        observed_safe,
        missing_harm,
        missing_safe,
        nominal,
        deployment_mass,
        budget=int(round(probe_fraction * int(missing.sum()))),
        calibrator=active_calibrator,
        threshold=threshold,
        max_updates_per_stratum=active_checkpoints - 1,
    )
    active_observed = active_harm + active_safe
    passive_credal = np.empty(strata)
    uniform_active_credal = np.empty(strata)
    active_credal = np.empty(strata)
    passive_sharp_upper = np.empty(strata)
    uniform_sharp_upper = np.empty(strata)
    active_sharp_upper = np.empty(strata)
    observed_ucb = np.empty(strata)
    active_observed_ucb = np.empty(strata)
    for index in range(strata):
        passive = calibrator.certificate(PartialFeedbackCounts(
            observed_harm=int(observed_harm[index]),
            observed_safe=int(observed_safe[index]),
            unidentified=int(missing[index]),
        ))
        # Uniform probing evaluates one predeclared terminal sample size and
        # therefore does not spend the active policy's checkpoint correction.
        uniform = calibrator.certificate(PartialFeedbackCounts(
            observed_harm=int(uniform_harm[index]),
            observed_safe=int(uniform_safe[index]),
            unidentified=int(uniform_missing[index]),
        ))
        active = active_calibrator.certificate(PartialFeedbackCounts(
            observed_harm=int(active_harm[index]),
            observed_safe=int(active_safe[index]),
            unidentified=int(active_missing[index]),
        ))
        passive_credal[index] = passive.confidence_upper
        uniform_active_credal[index] = uniform.confidence_upper
        active_credal[index] = active.confidence_upper
        passive_sharp_upper[index] = passive.identified_upper
        uniform_sharp_upper[index] = uniform.identified_upper
        active_sharp_upper[index] = active.identified_upper
        observed_ucb[index] = _upper_observed_harm(
            int(observed[index]), int(observed_harm[index]), alpha_per_stratum
        )
        active_observed_ucb[index] = _upper_observed_harm(
            int(active_observed[index]), int(active_harm[index]),
            alpha_per_stratum / active_checkpoints
        )

    complete_case = np.divide(
        observed_harm, observed, out=np.ones(strata, dtype=float), where=observed > 0
    )
    active_complete_case = np.divide(
        active_harm, active_observed, out=np.ones(strata, dtype=float), where=active_observed > 0
    )
    nominal_imputation = (
        observed_harm / per_stratum + (missing / per_stratum) * nominal
    )
    scores = {
        "nominal_point": nominal,
        "complete_case": complete_case,
        "nominal_imputation": nominal_imputation,
        "observed_ucb": observed_ucb,
        "max_nominal_observed_ucb": np.maximum(nominal, observed_ucb),
        "passive_credal": passive_credal,
        "uniform_active_credal": uniform_active_credal,
        "active_complete_case": active_complete_case,
        "active_observed_ucb": active_observed_ucb,
        "active_credal": active_credal,
        "oracle": true_risk,
        "deny_all": np.full(strata, math.inf),
    }

    sharp_by_policy = {
        "uniform_active_credal": uniform_sharp_upper,
        "active_complete_case": active_sharp_upper,
        "active_observed_ucb": active_sharp_upper,
        "active_credal": active_sharp_upper,
    }
    uniform_probe_cost = float(
        (revealed_harm.sum() + revealed_safe.sum()) / (per_stratum * strata)
    )
    targeted_probe_cost = targeted_used / (per_stratum * strata)
    rows = []
    for policy in PROTOCOL["policies"]:
        selected = scores[policy] <= threshold
        coverage = float(deployment_mass[selected].sum())
        harm_mass = float(np.sum(deployment_mass[selected] * true_risk[selected]))
        selective_harm = harm_mass / coverage if coverage else 0.0
        unsafe_mass = float(deployment_mass[selected & (true_risk > threshold)].sum())
        relevant_sharp = sharp_by_policy.get(policy, passive_sharp_upper)
        incompatible_mass = float(deployment_mass[selected & (relevant_sharp > threshold)].sum())
        compatible_harm_mass = float(np.sum(deployment_mass[selected] * relevant_sharp[selected]))
        rows.append({
            "seed": seed,
            "missingness": regime,
            "probe_fraction": probe_fraction,
            "policy": policy,
            "automatic_coverage": coverage,
            "expected_selective_harm": selective_harm,
            "safe_automatic_utility": coverage - harm_mass,
            "true_unsafe_coverage_mass": unsafe_mass,
            "compatible_violation_mass": incompatible_mass,
            "worst_compatible_selective_harm": (
                compatible_harm_mass / coverage if coverage else 0.0
            ),
            "risk_target_exceeded": float(coverage > 0.0 and selective_harm > threshold),
            "probe_cost_fraction": (
                uniform_probe_cost if policy == "uniform_active_credal"
                else targeted_probe_cost if policy.startswith("active_")
                else 0.0
            ),
            "max_adaptive_updates_used": max_updates_used,
        })

    active_coverage = next(
        row["automatic_coverage"] for row in rows if row["policy"] == "active_credal"
    )
    matched = []
    for policy in PROTOCOL["policies"]:
        if policy == "deny_all" or active_coverage == 0.0:
            continue
        order = np.argsort(scores[policy], kind="stable")
        cumulative = np.cumsum(deployment_mass[order])
        count = min(len(order), int(np.searchsorted(cumulative, active_coverage)) + 1)
        selected_index = order[:count]
        selected_mass = float(deployment_mass[selected_index].sum())
        harm = float(np.sum(deployment_mass[selected_index] * true_risk[selected_index]))
        matched.append({
            "seed": seed,
            "missingness": regime,
            "probe_fraction": probe_fraction,
            "policy": policy,
            "matched_coverage": selected_mass,
            "expected_selective_harm": harm / selected_mass,
            "true_unsafe_coverage_mass": float(
                deployment_mass[selected_index][true_risk[selected_index] > threshold].sum()
            ),
        })
    return {"rows": rows, "matched_rows": matched}


def interval(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    half = 2.0096 * float(data.std(ddof=1)) / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {"mean": mean, "low": mean - half, "high": mean + half}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for regime in PROTOCOL["missingness_regimes"]:
        for probe in PROTOCOL["probe_sensitivity"]:
            for policy in PROTOCOL["policies"]:
                subset = [
                    row for row in rows
                    if row["missingness"] == regime
                    and row["probe_fraction"] == probe
                    and row["policy"] == policy
                ]
                if not subset:
                    continue
                output.append({
                    "missingness": regime,
                    "probe_fraction": probe,
                    "policy": policy,
                    "automatic_coverage": interval([r["automatic_coverage"] for r in subset]),
                    "expected_selective_harm": interval([r["expected_selective_harm"] for r in subset]),
                    "safe_automatic_utility": interval([r["safe_automatic_utility"] for r in subset]),
                    "true_unsafe_coverage_mass": interval([r["true_unsafe_coverage_mass"] for r in subset]),
                    "compatible_violation_mass": interval([r["compatible_violation_mass"] for r in subset]),
                    "worst_compatible_selective_harm": interval([
                        r["worst_compatible_selective_harm"] for r in subset
                    ]),
                    "risk_violation_frequency": float(np.mean([r["risk_target_exceeded"] for r in subset])),
                    "probe_cost_fraction": interval([r["probe_cost_fraction"] for r in subset]),
                })
    return output


def summarize_matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for regime in PROTOCOL["missingness_regimes"]:
        for probe in PROTOCOL["probe_sensitivity"]:
            for policy in PROTOCOL["policies"]:
                subset = [
                    row for row in rows
                    if row["missingness"] == regime
                    and row["probe_fraction"] == probe
                    and row["policy"] == policy
                ]
                if not subset:
                    continue
                output.append({
                    "missingness": regime,
                    "probe_fraction": probe,
                    "policy": policy,
                    "matched_coverage": interval([r["matched_coverage"] for r in subset]),
                    "expected_selective_harm": interval([r["expected_selective_harm"] for r in subset]),
                    "true_unsafe_coverage_mass": interval([r["true_unsafe_coverage_mass"] for r in subset]),
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = 5 if args.quick else int(PROTOCOL["seeds"])
    all_rows: list[dict[str, Any]] = []
    all_matched: list[dict[str, Any]] = []
    for regime in PROTOCOL["missingness_regimes"]:
        for probe in PROTOCOL["probe_sensitivity"]:
            for seed in range(seeds):
                result = generate_seed(seed, regime, probe, quick=args.quick)
                all_rows.extend(result["rows"])
                all_matched.extend(result["matched_rows"])
    protocol_json = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":"))
    output = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(protocol_json.encode("utf-8")).hexdigest(),
        "quick": args.quick,
        "records": all_rows,
        "matched_records": all_matched,
        "summary": summarize(all_rows),
        "matched_summary": summarize_matched(all_matched),
    }
    path = Path("experiments/results/partial_feedback_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    key = [
        row for row in output["summary"]
        if row["missingness"] == "selective"
        and row["probe_fraction"] == PROTOCOL["active_probe_fraction"]
        and row["policy"] in {
            "nominal_point", "complete_case", "active_observed_ucb",
            "passive_credal", "active_credal", "oracle",
        }
    ]
    print(json.dumps({"protocol_sha256": output["protocol_sha256"], "primary": key}, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Predeclared benchmark for locally adaptive credal risk routing.

The synthetic population is intentionally heterogeneous: smooth background
risk is combined with localized model-misspecification pockets and effect-group
structure.  The generator, candidate neighbourhoods, baselines, coverage
targets, and seeds are fixed in ``PROTOCOL`` and hashed into the result file.
Ground-truth probabilities are retained only because this is a controlled
mechanism-identification study; policy fitting sees sampled binary outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from credal_harness.adaptive import RiskControlledSelector


PROTOCOL: dict[str, Any] = {
    "version": "adaptive-routing-v1",
    "seeds": 30,
    "calibration_per_seed": 4_000,
    "test_per_seed": 10_000,
    "policy_calibration_per_seed": 8_000,
    "dimensions": 5,
    "groups": 4,
    "alpha": 0.05,
    "candidate_ks": [64, 128, 256, 512],
    "lipschitz_allowance": 0.015,
    "fixed_epsilon": 0.05,
    "target_coverages": [0.25, 0.50, 0.75],
    "distribution_shifts": [0.0, 0.08, 0.16],
    "risk_control_targets": [0.05, 0.075, 0.10],
    "risk_control_coverages": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    "policies": [
        "point",
        "fixed_credal",
        "maxrisk",
        "global_residual_ucb",
        "group_residual_ucb",
        "knn_harm_ucb",
        "adaptive_credal",
    ],
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def generate_population(
    rng: np.random.Generator,
    size: int,
    *,
    shift: float = 0.0,
) -> dict[str, np.ndarray]:
    """Draw calls and their declared-model and ground-truth risks."""
    x = rng.uniform(0.0, 1.0, size=(size, PROTOCOL["dimensions"]))
    group = rng.integers(0, PROTOCOL["groups"], size=size)

    h0 = 0.003 + 0.016 * x[:, 0] + 0.006 * (group == 3)
    h1 = 0.004 + 0.012 * x[:, 0] + 0.055 * np.maximum(0.0, x[:, 1] - 0.72)
    h2 = 0.003 + 0.014 * x[:, 0] + 0.050 * np.maximum(0.0, 0.28 - x[:, 2])
    risks = np.clip(np.column_stack((h0, h1, h2)), 0.0, 1.0)
    nominal = np.sum(risks * np.asarray([0.80, 0.12, 0.08]), axis=1)

    # Localized excess risks are not explicitly represented by the three
    # declared hypotheses.  Shift expands these same predeclared pockets; it
    # does not introduce a post-hoc attack type.
    pocket_a = 0.28 * sigmoid(24.0 * (x[:, 0] + x[:, 1] - (1.48 - shift)))
    pocket_b = 0.22 * (group == 2) * sigmoid(25.0 * ((0.22 + shift / 2.0) - x[:, 2]))
    pocket_c = 0.16 * (group == 1) * sigmoid(24.0 * (x[:, 3] - (0.86 - shift / 2.0)))
    interaction = 0.08 * (group == 3) * x[:, 1] * x[:, 4]
    true_risk = np.clip(nominal + pocket_a + pocket_b + pocket_c + interaction, 0.0, 0.95)

    disagreement = risks.max(axis=1) - risks.min(axis=1)
    features = np.column_stack((x, disagreement))
    return {
        "features": features,
        "group": group,
        "risks": risks,
        "nominal": nominal,
        "true_risk": true_risk,
    }


def local_scores(
    calibration: dict[str, np.ndarray],
    outcomes: np.ndarray,
    test: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Vectorized implementation of the predeclared adaptive neighbourhoods."""
    cal_x = calibration["features"]
    test_x = test["features"]
    mean = cal_x.mean(axis=0)
    scale = np.maximum(cal_x.std(axis=0), 1e-9)
    cal_x = (cal_x - mean) / scale
    test_x = (test_x - mean) / scale
    candidates = np.asarray(PROTOCOL["candidate_ks"], dtype=int)
    alpha = float(PROTOCOL["alpha"])
    lipschitz = float(PROTOCOL["lipschitz_allowance"])
    adaptive = np.empty(len(test_x))
    knn_harm = np.empty(len(test_x))
    selected_k = np.empty(len(test_x), dtype=int)
    selected_radius = np.empty(len(test_x))
    selected_epsilon = np.empty(len(test_x))

    for group in range(PROTOCOL["groups"]):
        cal_idx = np.flatnonzero(calibration["group"] == group)
        test_idx = np.flatnonzero(test["group"] == group)
        available = candidates[candidates <= len(cal_idx)]
        if not len(available):
            available = np.asarray([len(cal_idx)], dtype=int)
        max_k = int(available.max())
        multiplicity = len(available)
        cal_group = cal_x[cal_idx]
        cal_residual = outcomes[cal_idx] - calibration["nominal"][cal_idx]
        cal_harm = outcomes[cal_idx]

        for start in range(0, len(test_idx), 256):
            query_idx = test_idx[start:start + 256]
            query = test_x[query_idx]
            # Squared Euclidean distance without allocating a third dimension.
            cross = np.einsum("ij,kj->ik", query, cal_group, optimize=True)
            distance2 = (
                np.sum(query * query, axis=1)[:, None]
                + np.sum(cal_group * cal_group, axis=1)[None, :]
                - 2.0 * cross
            )
            distance2 = np.maximum(distance2, 0.0)
            nearest = np.argpartition(distance2, max_k - 1, axis=1)[:, :max_k]
            nearest_d2 = np.take_along_axis(distance2, nearest, axis=1)
            order = np.argsort(nearest_d2, axis=1)
            nearest = np.take_along_axis(nearest, order, axis=1)
            nearest_d2 = np.take_along_axis(nearest_d2, order, axis=1)
            residual_cumsum = np.cumsum(cal_residual[nearest], axis=1)
            harm_cumsum = np.cumsum(cal_harm[nearest], axis=1)

            adaptive_candidates = []
            harm_candidates = []
            radius_candidates = []
            for k in available:
                radius = np.sqrt(nearest_d2[:, k - 1])
                concentration = math.sqrt(math.log(multiplicity / alpha) / (2.0 * k))
                locality = lipschitz * radius
                residual = residual_cumsum[:, k - 1] / k
                empirical_harm = harm_cumsum[:, k - 1] / k
                adaptive_candidates.append(
                    np.clip(test["nominal"][query_idx] + residual + concentration + locality, 0.0, 1.0)
                )
                harm_candidates.append(np.clip(empirical_harm + concentration + locality, 0.0, 1.0))
                radius_candidates.append(radius)
            adaptive_matrix = np.column_stack(adaptive_candidates)
            harm_matrix = np.column_stack(harm_candidates)
            radius_matrix = np.column_stack(radius_candidates)
            choice = np.argmin(adaptive_matrix, axis=1)
            rows = np.arange(len(query_idx))
            chosen_upper = adaptive_matrix[rows, choice]
            adaptive[query_idx] = chosen_upper
            knn_harm[query_idx] = harm_matrix[rows, np.argmin(harm_matrix, axis=1)]
            selected_k[query_idx] = available[choice]
            selected_radius[query_idx] = radius_matrix[rows, choice]
            selected_epsilon[query_idx] = np.clip(
                (chosen_upper - test["nominal"][query_idx])
                / np.maximum(1e-12, 1.0 - test["nominal"][query_idx]),
                0.0,
                1.0,
            )
    diagnostics = {
        "selected_k": selected_k,
        "selected_radius": selected_radius,
        "selected_epsilon": selected_epsilon,
    }
    return adaptive, knn_harm, diagnostics


def baseline_scores(
    calibration: dict[str, np.ndarray],
    outcomes: np.ndarray,
    test: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    nominal = test["nominal"]
    alpha = float(PROTOCOL["alpha"])
    residual = outcomes - calibration["nominal"]
    global_margin = math.sqrt(math.log(1.0 / alpha) / (2.0 * len(outcomes)))
    global_correction = max(0.0, float(residual.mean()) + global_margin)
    group_score = nominal.copy()
    for group in range(PROTOCOL["groups"]):
        mask = calibration["group"] == group
        margin = math.sqrt(math.log(PROTOCOL["groups"] / alpha) / (2.0 * int(mask.sum())))
        correction = max(0.0, float(residual[mask].mean()) + margin)
        group_score[test["group"] == group] += correction
    fixed_epsilon = float(PROTOCOL["fixed_epsilon"])
    return {
        "point": nominal,
        "fixed_credal": nominal + fixed_epsilon * (1.0 - nominal),
        "maxrisk": test["risks"].max(axis=1),
        "global_residual_ucb": np.clip(nominal + global_correction, 0.0, 1.0),
        "group_residual_ucb": np.clip(group_score, 0.0, 1.0),
    }


def evaluate_seed(seed: int, shift: float, *, quick: bool = False) -> list[dict[str, Any]]:
    cal_n = 1_000 if quick else int(PROTOCOL["calibration_per_seed"])
    test_n = 2_000 if quick else int(PROTOCOL["test_per_seed"])
    rng = np.random.default_rng(8_104_729 + seed)
    calibration = generate_population(rng, cal_n, shift=0.0)
    outcomes = rng.binomial(1, calibration["true_risk"]).astype(float)
    test = generate_population(rng, test_n, shift=shift)
    scores = baseline_scores(calibration, outcomes, test)
    adaptive, knn_harm, diagnostics = local_scores(calibration, outcomes, test)
    scores["adaptive_credal"] = adaptive
    scores["knn_harm_ucb"] = knn_harm

    rows = []
    for target in PROTOCOL["target_coverages"]:
        selected_n = max(1, int(round(target * test_n)))
        for policy in PROTOCOL["policies"]:
            selected = np.argpartition(scores[policy], selected_n - 1)[:selected_n]
            true_risk = test["true_risk"][selected]
            realized = rng.binomial(1, true_risk)
            row = {
                "seed": seed,
                "shift": shift,
                "target_coverage": target,
                "policy": policy,
                "achieved_coverage": selected_n / test_n,
                "expected_selective_harm": float(true_risk.mean()),
                "realized_selective_harm": float(realized.mean()),
                "safe_automatic_utility": float(np.mean(1.0 - true_risk) * selected_n / test_n),
                "score_mean": float(scores[policy][selected].mean()),
            }
            if policy == "adaptive_credal":
                row.update({
                    "mean_epsilon": float(diagnostics["selected_epsilon"][selected].mean()),
                    "mean_selected_k": float(diagnostics["selected_k"][selected].mean()),
                    "mean_neighbourhood_radius": float(diagnostics["selected_radius"][selected].mean()),
                })
            rows.append(row)
    return rows


def evaluate_risk_control_seed(seed: int, shift: float, *, quick: bool = False) -> list[dict[str, Any]]:
    """Fit score and policy threshold on disjoint splits, then audit test risk."""
    fit_n = 1_000 if quick else int(PROTOCOL["calibration_per_seed"])
    policy_n = 2_000 if quick else int(PROTOCOL["policy_calibration_per_seed"])
    test_n = 2_000 if quick else int(PROTOCOL["test_per_seed"])
    rng = np.random.default_rng(9_204_731 + seed)
    fit = generate_population(rng, fit_n, shift=0.0)
    fit_outcomes = rng.binomial(1, fit["true_risk"]).astype(float)
    policy = generate_population(rng, policy_n, shift=0.0)
    policy_outcomes = rng.binomial(1, policy["true_risk"]).astype(int)
    design = generate_population(rng, 1_000 if quick else 4_000, shift=0.0)
    test = generate_population(rng, test_n, shift=shift)

    policy_scores = baseline_scores(fit, fit_outcomes, policy)
    design_scores = baseline_scores(fit, fit_outcomes, design)
    test_scores = baseline_scores(fit, fit_outcomes, test)
    adaptive_policy, knn_policy, _ = local_scores(fit, fit_outcomes, policy)
    adaptive_design, knn_design, _ = local_scores(fit, fit_outcomes, design)
    adaptive_test, knn_test, _ = local_scores(fit, fit_outcomes, test)
    policy_scores.update({"adaptive_credal": adaptive_policy, "knn_harm_ucb": knn_policy})
    design_scores.update({"adaptive_credal": adaptive_design, "knn_harm_ucb": knn_design})
    test_scores.update({"adaptive_credal": adaptive_test, "knn_harm_ucb": knn_test})

    rows = []
    for target_risk in PROTOCOL["risk_control_targets"]:
        for policy_name in PROTOCOL["policies"]:
            selector = RiskControlledSelector(
                target_risk=target_risk,
                alpha=PROTOCOL["alpha"],
                candidate_thresholds=np.quantile(
                    design_scores[policy_name], PROTOCOL["risk_control_coverages"]
                ).tolist(),
            )
            certificate = selector.fit(policy_scores[policy_name].tolist(), policy_outcomes.tolist())
            selected = test_scores[policy_name] <= certificate.threshold
            selected_n = int(selected.sum())
            rows.append({
                "seed": seed,
                "shift": shift,
                "policy": policy_name,
                "target_risk": target_risk,
                "calibration_coverage": certificate.calibration_coverage,
                "calibration_empirical_risk": certificate.empirical_risk,
                "calibration_upper_risk": certificate.upper_risk,
                "test_coverage": selected_n / test_n,
                "test_expected_selective_harm": (
                    float(test["true_risk"][selected].mean()) if selected_n else 0.0
                ),
                "test_risk_violation": float(
                    selected_n > 0 and test["true_risk"][selected].mean() > target_risk
                ),
                "safe_automatic_utility": (
                    float(np.sum(1.0 - test["true_risk"][selected]) / test_n) if selected_n else 0.0
                ),
            })
    return rows


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    half = 2.0452 * float(array.std(ddof=1)) / math.sqrt(len(array)) if len(array) > 1 else 0.0
    low, high = mean - half, mean + half
    if np.all((array >= 0.0) & (array <= 1.0)):
        low, high = max(0.0, low), min(1.0, high)
    return {"mean": mean, "low": low, "high": high}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = []
    comparisons: dict[str, Any] = {}
    for shift in PROTOCOL["distribution_shifts"]:
        for coverage in PROTOCOL["target_coverages"]:
            subset = [r for r in rows if r["shift"] == shift and r["target_coverage"] == coverage]
            key = f"shift={shift:.2f},coverage={coverage:.2f}"
            comparisons[key] = {}
            adaptive = {r["seed"]: r for r in subset if r["policy"] == "adaptive_credal"}
            for policy in PROTOCOL["policies"]:
                policy_rows = [r for r in subset if r["policy"] == policy]
                summary.append({
                    "shift": shift,
                    "target_coverage": coverage,
                    "policy": policy,
                    "expected_selective_harm": interval([r["expected_selective_harm"] for r in policy_rows]),
                    "realized_selective_harm": interval([r["realized_selective_harm"] for r in policy_rows]),
                    "safe_automatic_utility": interval([r["safe_automatic_utility"] for r in policy_rows]),
                })
                if policy != "adaptive_credal":
                    by_seed = {r["seed"]: r for r in policy_rows}
                    deltas = [
                        adaptive[seed]["expected_selective_harm"] - by_seed[seed]["expected_selective_harm"]
                        for seed in sorted(adaptive)
                    ]
                    relative = [
                        -delta / max(1e-12, by_seed[seed]["expected_selective_harm"])
                        for seed, delta in zip(sorted(adaptive), deltas)
                    ]
                    comparisons[key][policy] = {
                        "adaptive_minus_baseline": interval(deltas),
                        "relative_harm_reduction": interval(relative),
                        "wins": sum(delta < 0 for delta in deltas),
                        "losses": sum(delta > 0 for delta in deltas),
                        "ties": sum(delta == 0 for delta in deltas),
                    }
    return {"summary": summary, "paired_comparisons": comparisons}


def summarize_risk_control(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for shift in PROTOCOL["distribution_shifts"]:
        for target in PROTOCOL["risk_control_targets"]:
            for policy in PROTOCOL["policies"]:
                subset = [
                    row for row in rows
                    if row["shift"] == shift and row["target_risk"] == target and row["policy"] == policy
                ]
                output.append({
                    "shift": shift,
                    "target_risk": target,
                    "policy": policy,
                    "test_coverage": interval([row["test_coverage"] for row in subset]),
                    "test_expected_selective_harm": interval([
                        row["test_expected_selective_harm"] for row in subset
                    ]),
                    "safe_automatic_utility": interval([row["safe_automatic_utility"] for row in subset]),
                    "violation_frequency": sum(row["test_risk_violation"] for row in subset) / len(subset),
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = 3 if args.quick else int(PROTOCOL["seeds"])
    rows = []
    risk_control_rows = []
    for shift in PROTOCOL["distribution_shifts"]:
        for seed in range(seeds):
            rows.extend(evaluate_seed(seed, shift, quick=args.quick))
            risk_control_rows.extend(evaluate_risk_control_seed(seed, shift, quick=args.quick))
    protocol_json = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":"))
    output = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(protocol_json.encode("utf-8")).hexdigest(),
        "quick": args.quick,
        "records": rows,
        "risk_control_records": risk_control_rows,
        "risk_control_summary": summarize_risk_control(risk_control_rows),
        **summarize(rows),
    }
    path = Path("experiments/results/adaptive_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    key = "shift=0.00,coverage=0.50"
    print(json.dumps({"protocol_sha256": output["protocol_sha256"], key: output["paired_comparisons"][key]}, indent=2))


if __name__ == "__main__":
    main()

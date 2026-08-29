"""Locally calibrated credal risk routing.

This module contains the algorithmic part of the artifact.  It deliberately
does not issue capabilities or resolve authorization: it estimates a local
upper harm envelope that the reference monitor may consume.

For a query x, the calibrator considers a predeclared family of k-nearest
neighbourhoods.  A simultaneous Hoeffding correction for the local calibration
residual and a declared local Lipschitz allowance produce an upper estimate
for each neighbourhood.  Taking
the tightest simultaneously valid estimate selects the data-supported scale.
The estimate is represented as the smallest epsilon-contamination radius that
lifts the nominal query risk to that local upper envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class CalibrationSample:
    """One trusted calibration observation."""

    features: tuple[float, ...]
    nominal_risk: float
    harm: int
    group: str = "all"

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("calibration features cannot be empty")
        if self.harm not in (0, 1, False, True):
            raise ValueError("harm must be binary")


@dataclass(frozen=True)
class LocalCredalCertificate:
    """Auditable output of one local credal calibration query."""

    nominal_risk: float
    upper_risk: float
    epsilon: float
    selected_k: int
    neighbourhood_radius: float
    empirical_harm: float
    neighbour_nominal_risk: float
    concentration_margin: float
    locality_margin: float
    group: str


class AdaptiveCredalCalibrator:
    """Select a locally supported epsilon-contamination neighbourhood.

    The finite-sample interpretation is conditional on three declared
    premises: calibration observations are independent blocks, the candidate
    neighbourhood sizes are fixed before observing outcomes, and the true
    conditional harm function is locally Lipschitz in the standardized feature
    metric with constant at most ``lipschitz``.  The Bonferroni correction makes
    all candidate neighbourhood bounds simultaneous for one query; therefore
    selecting the tightest bound does not consume an unreported tuning split.
    """

    def __init__(
        self,
        samples: Iterable[CalibrationSample],
        *,
        alpha: float = 0.05,
        candidate_ks: Sequence[int] = (32, 64, 128, 256),
        lipschitz: float = 0.05,
        prior_epsilon: float = 0.0,
        group_conditional: bool = True,
    ) -> None:
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("adaptive calibration requires observations")
        width = len(self.samples[0].features)
        if any(len(row.features) != width for row in self.samples):
            raise ValueError("all feature vectors must have equal length")
        self.alpha = min(1.0 - 1e-12, max(1e-12, float(alpha)))
        self.candidate_ks = tuple(sorted({int(k) for k in candidate_ks if int(k) > 0}))
        if not self.candidate_ks:
            raise ValueError("at least one positive neighbourhood size is required")
        self.lipschitz = max(0.0, float(lipschitz))
        self.prior_epsilon = _clip01(prior_epsilon)
        self.group_conditional = bool(group_conditional)
        self.feature_mean = tuple(
            sum(row.features[j] for row in self.samples) / len(self.samples)
            for j in range(width)
        )
        self.feature_scale = tuple(
            max(
                1e-9,
                math.sqrt(
                    sum((row.features[j] - self.feature_mean[j]) ** 2 for row in self.samples)
                    / len(self.samples)
                ),
            )
            for j in range(width)
        )

    def _standardize(self, features: Sequence[float]) -> tuple[float, ...]:
        if len(features) != len(self.feature_mean):
            raise ValueError("query feature width does not match calibration data")
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(features, self.feature_mean, self.feature_scale)
        )

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def certificate(
        self,
        features: Sequence[float],
        nominal_risk: float,
        *,
        group: str = "all",
    ) -> LocalCredalCertificate:
        query = self._standardize(features)
        eligible = [
            row for row in self.samples
            if not self.group_conditional or row.group == group
        ]
        # A previously unseen group falls back to the full calibration sample;
        # the returned certificate records that fact through group="__global__".
        certificate_group = group
        if not eligible:
            eligible = list(self.samples)
            certificate_group = "__global__"
        distances = sorted(
            [
                (
                self._distance(query, self._standardize(row.features)),
                row,
                )
                for row in eligible
            ],
            key=lambda item: item[0],
        )
        ks = tuple(k for k in self.candidate_ks if k <= len(distances))
        if not ks:
            ks = (len(distances),)
        multiplicity = len(ks)
        nominal_query = _clip01(nominal_risk)
        best: LocalCredalCertificate | None = None
        for k in ks:
            neighbours = distances[:k]
            radius = neighbours[-1][0]
            empirical = sum(float(row.harm) for _, row in neighbours) / k
            neighbour_nominal = sum(_clip01(row.nominal_risk) for _, row in neighbours) / k
            residual = sum(
                float(row.harm) - _clip01(row.nominal_risk)
                for _, row in neighbours
            ) / k
            concentration = math.sqrt(math.log(multiplicity / self.alpha) / (2.0 * k))
            locality = self.lipschitz * radius
            # Each centred Bernoulli residual Y_i-p_i has range length one,
            # even when p_i differs between neighbours.  Local calibration
            # therefore corrects the query model instead of replacing it with
            # a noisy neighbourhood prevalence estimate.
            target = _clip01(nominal_query + residual + concentration + locality)
            required = max(0.0, target - nominal_query) / max(1e-12, 1.0 - nominal_query)
            epsilon = max(self.prior_epsilon, _clip01(required))
            upper = nominal_query + epsilon * (1.0 - nominal_query)
            candidate = LocalCredalCertificate(
                nominal_risk=nominal_query,
                upper_risk=_clip01(upper),
                epsilon=epsilon,
                selected_k=k,
                neighbourhood_radius=radius,
                empirical_harm=empirical,
                neighbour_nominal_risk=neighbour_nominal,
                concentration_margin=concentration,
                locality_margin=locality,
                group=certificate_group,
            )
            if best is None or candidate.upper_risk < best.upper_risk:
                best = candidate
        assert best is not None
        return best


@dataclass(frozen=True)
class BudgetedRoutingDecision:
    action: str
    reason: str
    upper_risk: float
    charged_risk: float
    remaining_budget: float
    certificate: LocalCredalCertificate


class BudgetedAdaptiveCredalRouter:
    """Online allow/confirm/deny routing with an auditable sequence ledger."""

    def __init__(
        self,
        calibrator: AdaptiveCredalCalibrator,
        *,
        automatic_threshold: float = 0.05,
        confirmation_threshold: float = 0.25,
        sequence_budget: float = 0.10,
    ) -> None:
        self.calibrator = calibrator
        self.automatic_threshold = _clip01(automatic_threshold)
        self.confirmation_threshold = _clip01(confirmation_threshold)
        self.sequence_budget = max(0.0, float(sequence_budget))
        self.spent_budget = 0.0

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.sequence_budget - self.spent_budget)

    def decide(
        self,
        features: Sequence[float],
        nominal_risk: float,
        *,
        group: str = "all",
        authorized: bool = True,
    ) -> BudgetedRoutingDecision:
        cert = self.calibrator.certificate(features, nominal_risk, group=group)
        before = self.remaining_budget
        if not authorized:
            action, reason, charge = "confirm", "trusted_authority_missing", 0.0
        elif cert.upper_risk <= self.automatic_threshold and cert.upper_risk <= before + 1e-12:
            action, reason, charge = "allow", "local_upper_risk_within_sequence_budget", cert.upper_risk
            self.spent_budget += charge
        elif cert.upper_risk <= self.confirmation_threshold:
            action, reason, charge = "confirm", "risk_or_sequence_budget_requires_confirmation", 0.0
        else:
            action, reason, charge = "deny", "local_upper_risk_above_confirmation_ceiling", 0.0
        return BudgetedRoutingDecision(
            action=action,
            reason=reason,
            upper_risk=cert.upper_risk,
            charged_risk=charge,
            remaining_budget=self.remaining_budget,
            certificate=cert,
        )


@dataclass(frozen=True)
class SelectiveRiskCertificate:
    """Split-calibrated threshold and its simultaneous risk certificate."""

    threshold: float
    selected: int
    calibration_size: int
    empirical_risk: float
    upper_risk: float
    target_risk: float
    alpha: float
    candidate_count: int

    @property
    def calibration_coverage(self) -> float:
        return self.selected / self.calibration_size


class RiskControlledSelector:
    """Choose the most autonomous score threshold with finite-sample control.

    The scorer and candidate thresholds must be frozen before this independent
    policy-calibration split is observed.  Thresholds may be constructed from
    the scorer's fitting split or an unlabeled design split.  A
    Bonferroni-Hoeffding bound is
    evaluated for every candidate, after which the highest-coverage feasible
    threshold is selected.  Under exchangeability, all candidate population
    selective risks are simultaneously bounded with probability at least
    ``1-alpha``.  This is a population-level selective guarantee, not a
    per-action containment statement.
    """

    def __init__(
        self,
        *,
        target_risk: float = 0.05,
        alpha: float = 0.05,
        candidate_thresholds: Sequence[float] = tuple(i / 20 for i in range(21)),
    ) -> None:
        self.target_risk = _clip01(target_risk)
        self.alpha = min(1.0 - 1e-12, max(1e-12, float(alpha)))
        self.candidate_thresholds = tuple(sorted({float(value) for value in candidate_thresholds}))
        if not self.candidate_thresholds:
            raise ValueError("at least one candidate threshold is required")
        self.certificate_: SelectiveRiskCertificate | None = None

    def fit(self, scores: Sequence[float], outcomes: Sequence[int]) -> SelectiveRiskCertificate:
        if len(scores) != len(outcomes) or not scores:
            raise ValueError("scores and outcomes must have the same nonzero length")
        if any(value not in (0, 1, False, True) for value in outcomes):
            raise ValueError("policy-calibration outcomes must be binary")
        candidates: list[SelectiveRiskCertificate] = []
        multiplicity = len(self.candidate_thresholds)
        for threshold in self.candidate_thresholds:
            chosen = [index for index, score in enumerate(scores) if float(score) <= threshold]
            selected = len(chosen)
            if selected == 0:
                continue
            empirical = sum(float(outcomes[index]) for index in chosen) / selected
            margin = math.sqrt(math.log(multiplicity / self.alpha) / (2.0 * selected))
            upper = _clip01(empirical + margin)
            candidates.append(SelectiveRiskCertificate(
                threshold=float(threshold),
                selected=selected,
                calibration_size=len(scores),
                empirical_risk=empirical,
                upper_risk=upper,
                target_risk=self.target_risk,
                alpha=self.alpha,
                candidate_count=multiplicity,
            ))
        feasible = [row for row in candidates if row.upper_risk <= self.target_risk]
        if feasible:
            result = max(feasible, key=lambda row: (row.selected, row.threshold))
        else:
            result = SelectiveRiskCertificate(
                threshold=float("-inf"),
                selected=0,
                calibration_size=len(scores),
                empirical_risk=0.0,
                upper_risk=self.target_risk,
                target_risk=self.target_risk,
                alpha=self.alpha,
                candidate_count=multiplicity,
            )
        self.certificate_ = result
        return result

    def allows(self, score: float) -> bool:
        if self.certificate_ is None:
            raise RuntimeError("fit must be called before routing")
        return float(score) <= self.certificate_.threshold

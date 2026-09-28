"""Credal risk certificates under selectively missing outcome feedback.

Side-effect monitors do not observe the counterfactual outcome of every denied
or externally failed call.  Treating those outcomes as safe is a point
completion, not evidence.  This module instead represents every completion of
the unidentified binary outcomes.

For an observation probability q and observed conditional harm r, the
population harm probability is partially identified as

    q r <= P(H) <= q r + (1 - q).

The interval is sharp without assumptions on the missing outcomes.  The
finite-sample certificate below replaces q by a lower confidence bound and r
by an upper confidence bound.  Trusted shadow probes move outcomes from the
unidentified mass into the observed counts and can therefore contract the
credal upper envelope without granting live execution authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class PartialFeedbackCounts:
    """Sufficient statistics for one predeclared risk stratum."""

    observed_harm: int
    observed_safe: int
    unidentified: int

    def __post_init__(self) -> None:
        if min(self.observed_harm, self.observed_safe, self.unidentified) < 0:
            raise ValueError("feedback counts cannot be negative")
        if self.total <= 0:
            raise ValueError("at least one calibration opportunity is required")

    @property
    def observed(self) -> int:
        return self.observed_harm + self.observed_safe

    @property
    def total(self) -> int:
        return self.observed + self.unidentified

    def reveal(self, *, harm: int = 0, safe: int = 0) -> "PartialFeedbackCounts":
        """Return counts after trusted probes reveal unidentified outcomes."""
        if min(harm, safe) < 0 or harm + safe > self.unidentified:
            raise ValueError("revealed outcomes must be a subset of unidentified feedback")
        return PartialFeedbackCounts(
            observed_harm=self.observed_harm + int(harm),
            observed_safe=self.observed_safe + int(safe),
            unidentified=self.unidentified - int(harm) - int(safe),
        )


@dataclass(frozen=True)
class PartialIdentificationCertificate:
    """Auditable sharp interval and its finite-sample upper certificate."""

    identified_lower: float
    identified_upper: float
    confidence_upper: float
    observed_fraction: float
    observed_fraction_lower: float
    observed_harm_rate: float
    observed_harm_rate_upper: float
    unidentified_fraction: float
    alpha: float
    counts: PartialFeedbackCounts

    def permits(self, risk_threshold: float) -> bool:
        return self.confidence_upper <= _clip01(risk_threshold)


class PartialFeedbackCredalCalibrator:
    """Construct a risk envelope valid for arbitrary missing-outcome laws.

    ``alpha`` is the error allocation for one stratum.  A union bound splits it
    equally between a lower bound on the observation probability and an upper
    bound on harm conditional on observation.  A caller certifying several
    strata should pass an already adjusted value, for example ``alpha / G``.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = float(alpha)

    @staticmethod
    def sharp_interval(counts: PartialFeedbackCounts) -> tuple[float, float]:
        """Empirical identified set over every binary missing completion."""
        lower = counts.observed_harm / counts.total
        upper = (counts.observed_harm + counts.unidentified) / counts.total
        return _clip01(lower), _clip01(upper)

    def certificate(self, counts: PartialFeedbackCounts) -> PartialIdentificationCertificate:
        lower, upper = self.sharp_interval(counts)
        n = counts.total
        observed = counts.observed
        observed_fraction = observed / n
        observed_risk = counts.observed_harm / observed if observed else 0.0

        # Two one-sided Hoeffding events, each at alpha/2.  The observed count
        # is itself random, but conditional on that count the observed binary
        # outcomes remain bounded; the stated result is therefore conditional
        # on the declared observation mechanism and independent blocks.
        margin_q = math.sqrt(math.log(2.0 / self.alpha) / (2.0 * n))
        q_lower = _clip01(observed_fraction - margin_q)
        if observed:
            margin_r = math.sqrt(math.log(2.0 / self.alpha) / (2.0 * observed))
            r_upper = _clip01(observed_risk + margin_r)
        else:
            r_upper = 1.0
        confidence_upper = _clip01(q_lower * r_upper + (1.0 - q_lower))
        return PartialIdentificationCertificate(
            identified_lower=lower,
            identified_upper=upper,
            confidence_upper=confidence_upper,
            observed_fraction=observed_fraction,
            observed_fraction_lower=q_lower,
            observed_harm_rate=observed_risk,
            observed_harm_rate_upper=r_upper,
            unidentified_fraction=counts.unidentified / n,
            alpha=self.alpha,
            counts=counts,
        )

    @staticmethod
    def required_observed_fraction(observed_harm_upper: float, risk_threshold: float) -> float:
        """Observation rate required for q*r + (1-q) <= threshold."""
        risk = _clip01(observed_harm_upper)
        threshold = _clip01(risk_threshold)
        if threshold >= 1.0:
            return 0.0
        if risk >= 1.0:
            return math.inf
        required = max(0.0, (1.0 - threshold) / (1.0 - risk))
        return required if required <= 1.0 else math.inf


@dataclass(frozen=True)
class PartialFeedbackRoutingDecision:
    action: str
    reason: str
    certificate: PartialIdentificationCertificate


class PartialFeedbackCredalRouter:
    """Route from a partial-identification certificate.

    This object deliberately returns only a gate decision.  A separate
    capability authority must still bind an allowed canonical call to the
    executor and revalidate its context at commit time.
    """

    def __init__(
        self,
        calibrator: PartialFeedbackCredalCalibrator,
        *,
        automatic_threshold: float = 0.05,
        confirmation_threshold: float = 0.25,
    ) -> None:
        self.calibrator = calibrator
        self.automatic_threshold = _clip01(automatic_threshold)
        self.confirmation_threshold = _clip01(confirmation_threshold)
        if self.confirmation_threshold < self.automatic_threshold:
            raise ValueError("confirmation threshold cannot be below automatic threshold")

    def decide(
        self,
        counts: PartialFeedbackCounts,
        *,
        authorized: bool = True,
        trusted_probe_available: bool = True,
    ) -> PartialFeedbackRoutingDecision:
        certificate = self.calibrator.certificate(counts)
        if not authorized:
            action, reason = "confirm", "trusted_authority_missing"
        elif certificate.confidence_upper <= self.automatic_threshold:
            action, reason = "allow", "credal_upper_within_automatic_threshold"
        elif trusted_probe_available and counts.unidentified:
            action, reason = "sandbox", "unidentified_feedback_requires_trusted_probe"
        elif certificate.confidence_upper <= self.confirmation_threshold:
            action, reason = "confirm", "credal_upper_requires_confirmation"
        else:
            action, reason = "deny", "credal_upper_above_confirmation_ceiling"
        return PartialFeedbackRoutingDecision(action, reason, certificate)

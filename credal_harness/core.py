"""Small, executable reference implementation for Credal Harness.

The implementation deliberately keeps the model class finite.  The harness is
the trusted boundary: an agent may propose a call, but only the harness can
issue a capability token and commit an execution.  Credal uncertainty is
represented by an epsilon-contamination set around a finite nominal model
distribution.  This is conservative: the contamination mass may be assigned
to whichever hypothesis is worst for the proposed action.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional


RiskFn = Callable[["ToolCall", Mapping[str, Any]], float]
LikelihoodFn = Callable[["Evidence"], float]
AuthorityFn = Callable[["ToolCall", Mapping[str, Any]], bool]


def state_digest(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(state), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolCall:
    """A proposed action produced by an untrusted agent."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    resource: str = "global"
    effect: str = "unknown"  # read, reversible, irreversible, external
    irreversible: bool = False
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def args_digest(self) -> str:
        canonical = json.dumps(dict(self.args), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            {
                "tool": self.tool,
                "args": dict(self.args),
                "resource": self.resource,
                "effect": self.effect,
                "irreversible": self.irreversible,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evidence:
    """Evidence collected by the harness, never by agent self-report alone."""

    kind: str
    value: Any = None
    confidence: float = 1.0
    source: str = "harness"
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Hypothesis:
    """One plausible outcome/effect model; it never grants authority."""

    name: str
    risk_fn: RiskFn
    likelihood_fn: LikelihoodFn = lambda evidence: 1.0

    def risk(self, call: ToolCall, state: Mapping[str, Any]) -> float:
        return min(1.0, max(0.0, float(self.risk_fn(call, state))))

    def likelihood(self, evidence: Evidence) -> float:
        return max(1e-12, min(1.0, float(self.likelihood_fn(evidence))))


@dataclass
class CredalSet:
    """Finite epsilon-contamination credal set.

    If ``weights`` is the nominal distribution and ``epsilon`` is the
    contamination mass, the credal set contains

        (1-epsilon) * weights + epsilon * q

    where q is vacuous over possible execution worlds, including worlds not
    enumerated by the finite nominal model.  For a [0,1]-valued harm event the
    contaminating upper (lower) expectation is therefore one (zero).  Optional
    group-specific radii are learned on a disjoint calibration split.
    """

    hypotheses: Dict[str, Hypothesis]
    weights: Dict[str, float]
    epsilon: float = 0.0
    group_epsilons: Dict[str, float] = field(default_factory=dict)
    coverage_ok: bool = True
    calibration: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("credal set requires at least one hypothesis")
        missing = set(self.hypotheses) - set(self.weights)
        if missing:
            raise ValueError(f"missing weights for hypotheses: {sorted(missing)}")
        self.epsilon = min(1.0, max(0.0, float(self.epsilon)))
        self.group_epsilons = {
            str(group): min(1.0, max(0.0, float(value)))
            for group, value in self.group_epsilons.items()
        }
        total = sum(max(0.0, float(v)) for v in self.weights.values())
        if total <= 0:
            raise ValueError("nominal weights must contain positive mass")
        self.weights = {k: max(0.0, float(v)) / total for k, v in self.weights.items()}

    @classmethod
    def uniform(cls, hypotheses: Iterable[Hypothesis], epsilon: float = 0.0) -> "CredalSet":
        hs = {h.name: h for h in hypotheses}
        w = {name: 1.0 / len(hs) for name in hs}
        return cls(hs, w, epsilon=epsilon)

    @classmethod
    def fit_from_records(
        cls,
        hypotheses: Iterable[Hypothesis],
        records: Iterable[Mapping[str, Any]],
        *,
        calibration_records: Optional[Iterable[Mapping[str, Any]]] = None,
        alpha: float = 0.10,
        prior_epsilon: float = 0.05,
        prior_weights: Optional[Mapping[str, float]] = None,
    ) -> "CredalSet":
        """Learn mixture weights and simultaneous group calibration radii.

        ``records`` is a fitting split containing
        ``call``, ``state``, and binary ``harm`` fields.  We first perform an
        iterative responsibility update over the hypotheses (a transparent
        finite-mixture maximum-likelihood surrogate, rather than a claim of
        latent-variable identifiability).  We then group records by
        the optional ``group`` field on a separate calibration split and choose
        the smallest contamination radius whose group-average upper risk covers
        the empirical rate plus a simultaneous one-sided Hoeffding margin. If
        rows carry an ``episode`` identifier, the bound is applied to episode
        block means, so rows within a stateful episode need not be independent.
        Coverage is still revoked when a trusted contract mismatch is observed
        at runtime; calibration cannot certify arbitrary opaque side effects.
        """
        hs = {h.name: h for h in hypotheses}
        if not hs:
            raise ValueError("calibration requires at least one hypothesis")
        rows = list(records)
        cal_rows = list(calibration_records) if calibration_records is not None else list(rows)
        if not rows:
            return cls.uniform(hs.values(), epsilon=prior_epsilon)
        prior = {name: float((prior_weights or {}).get(name, 1.0)) for name in hs}
        if sum(max(v, 0.0) for v in prior.values()) <= 0:
            raise ValueError("prior weights must contain positive mass")
        z_prior = sum(max(value, 0.0) for value in prior.values())
        weights = {name: max(prior[name], 0.0) / z_prior for name in hs}
        for _ in range(100):
            totals = {name: max(prior[name], 0.0) for name in hs}
            for row in rows:
                y = 1.0 if bool(row["harm"]) else 0.0
                likelihood = {}
                for name, hypothesis in hs.items():
                    p = min(1.0 - 1e-6, max(1e-6, hypothesis.risk(row["call"], row["state"])))
                    likelihood[name] = p if y else 1.0 - p
                norm = sum(weights[name] * likelihood[name] for name in hs)
                for name in hs:
                    totals[name] += weights[name] * likelihood[name] / max(norm, 1e-12)
            total = sum(totals.values())
            updated = {name: totals[name] / total for name in hs}
            if max(abs(updated[name] - weights[name]) for name in hs) < 1e-10:
                weights = updated
                break
            weights = updated
        grouped: Dict[str, Dict[str, list[tuple[float, float]]]] = {}
        nominal_risks = []
        for index, row in enumerate(cal_rows):
            values = {name: h.risk(row["call"], row["state"]) for name, h in hs.items()}
            nominal = sum(weights[name] * values[name] for name in hs)
            y = 1.0 if bool(row["harm"]) else 0.0
            nominal_risks.append(nominal)
            group = str(row.get("group", "all"))
            block = str(row.get("episode", index))
            grouped.setdefault(group, {}).setdefault(block, []).append((nominal, y))
        alpha = min(1.0 - 1e-9, max(1e-9, float(alpha)))
        group_alpha = alpha / max(1, len(grouped))
        group_gaps: Dict[str, float] = {}
        group_epsilons: Dict[str, float] = {}
        group_stats: Dict[str, Dict[str, float | int | bool]] = {}
        for group, blocks in grouped.items():
            pairs = [
                (sum(p for p, _ in values) / len(values), sum(y for _, y in values) / len(values))
                for values in blocks.values()
            ]
            n = len(pairs)
            empirical = sum(y for _, y in pairs) / n
            predicted = sum(p for p, _ in pairs) / n
            margin = math.sqrt(math.log(1.0 / group_alpha) / (2.0 * n))
            target = min(1.0, empirical + margin)
            gap = max(0.0, target - predicted)
            required = 0.0 if gap == 0.0 else gap / max(1e-12, 1.0 - predicted)
            group_gaps[group] = gap
            group_epsilons[group] = min(1.0, max(float(prior_epsilon), required))
            achieved = predicted + group_epsilons[group] * (1.0 - predicted)
            group_stats[group] = {
                "records": sum(len(values) for values in blocks.values()),
                "blocks": n,
                "empirical_harm": empirical,
                "nominal_risk": predicted,
                "hoeffding_margin": margin,
                "target_upper": target,
                "upper_gap": gap,
                "epsilon": group_epsilons[group],
                "achieved_upper": achieved,
                "calibrated": achieved + 1e-12 >= target,
            }
        learned_epsilon = max(float(prior_epsilon), max(group_epsilons.values(), default=0.0))
        calibration = {
            "fit_records": len(rows),
            "calibration_records": len(cal_rows),
            "records": len(cal_rows),
            "independent_calibration_split": calibration_records is not None,
            "alpha": alpha,
            "learned_epsilon": learned_epsilon,
            "prior_epsilon": float(prior_epsilon),
            "weights": dict(weights),
            "mean_nominal_risk": sum(nominal_risks) / len(nominal_risks),
            "groups": group_stats,
        }
        return cls(hs, weights, epsilon=float(prior_epsilon), group_epsilons=group_epsilons, calibration=calibration)

    def nominal_expectation(self, values: Mapping[str, float]) -> float:
        return sum(self.weights[name] * float(values[name]) for name in self.hypotheses)

    def upper_expectation(self, values: Mapping[str, float], epsilon: Optional[float] = None) -> float:
        nominal = self.nominal_expectation(values)
        eps = self.epsilon if epsilon is None else min(1.0, max(0.0, float(epsilon)))
        # The contaminating component is vacuous over possible execution
        # worlds, not restricted to redistributing mass among the enumerated
        # hypotheses.  Its worst-case harm probability is therefore one.
        return (1.0 - eps) * nominal + eps

    def lower_expectation(self, values: Mapping[str, float], epsilon: Optional[float] = None) -> float:
        nominal = self.nominal_expectation(values)
        eps = self.epsilon if epsilon is None else min(1.0, max(0.0, float(epsilon)))
        return (1.0 - eps) * nominal

    def epsilon_for(self, call: ToolCall) -> float:
        return self.group_epsilons.get(call.effect, self.epsilon)

    def risk_bounds(self, call: ToolCall, state: Mapping[str, Any]) -> tuple[float, float]:
        risks = {name: h.risk(call, state) for name, h in self.hypotheses.items()}
        eps = self.epsilon_for(call)
        return self.lower_expectation(risks, eps), self.upper_expectation(risks, eps)

    def disagreement(self, call: ToolCall, state: Mapping[str, Any]) -> float:
        values = [h.risk(call, state) for h in self.hypotheses.values()]
        return max(values) - min(values)

    def update(self, evidence: Evidence, likelihoods: Optional[Mapping[str, float]] = None) -> "CredalSet":
        """Robustly update nominal weights while retaining contamination mass.

        A harness may supply explicit likelihoods from an executable detector;
        otherwise each hypothesis supplies a conservative likelihood rule.
        Evidence with confidence below one increases contamination rather than
        pretending to be exact Bayesian data.
        """

        scores = {
            name: max(1e-12, float((likelihoods or {}).get(name, h.likelihood(evidence))))
            for name, h in self.hypotheses.items()
        }
        posterior_unnorm = {name: self.weights[name] * scores[name] for name in self.hypotheses}
        z = sum(posterior_unnorm.values())
        weights = {name: value / z for name, value in posterior_unnorm.items()}
        added_uncertainty = (1.0 - min(1.0, max(0.0, evidence.confidence))) * 0.25
        if evidence.kind in {"model_mismatch", "unknown_state", "permission_conflict"}:
            added_uncertainty = max(added_uncertainty, 0.20)
        return CredalSet(
            dict(self.hypotheses),
            weights,
            epsilon=min(1.0, self.epsilon + added_uncertainty),
            group_epsilons={
                group: min(1.0, value + added_uncertainty)
                for group, value in self.group_epsilons.items()
            },
            coverage_ok=self.coverage_ok and evidence.kind != "model_mismatch",
            calibration=dict(self.calibration),
        )

    def invalidate(self) -> "CredalSet":
        return CredalSet(
            dict(self.hypotheses), dict(self.weights), epsilon=max(self.epsilon, 0.5),
            group_epsilons={group: max(value, 0.5) for group, value in self.group_epsilons.items()},
            coverage_ok=False, calibration=dict(self.calibration),
        )


@dataclass(frozen=True)
class CapabilityToken:
    call_id: str
    tool: str
    resource: str
    mode: str
    expires_at: float
    args_digest: str
    context_digest: str
    context_epoch: int = 0
    max_uses: int = 1
    signature: str = ""


class CapabilityAuthority:
    """Trusted issuer/verifier for unforgeable capability tokens."""

    def __init__(self, secret: Optional[bytes] = None) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def bump_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    @staticmethod
    def _payload(token: CapabilityToken) -> bytes:
        value = {
            "call_id": token.call_id,
            "tool": token.tool,
            "resource": token.resource,
            "mode": token.mode,
            "expires_at": token.expires_at,
            "args_digest": token.args_digest,
                "context_digest": token.context_digest,
                "context_epoch": token.context_epoch,
                "max_uses": token.max_uses,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def issue(
        self,
        call: ToolCall,
        mode: str,
        ttl: float = 30.0,
        max_uses: int = 1,
        expires_at: Optional[float] = None,
        context_digest: str = "",
    ) -> CapabilityToken:
        unsigned = CapabilityToken(
            call_id=call.call_id,
            tool=call.tool,
            resource=call.resource,
            mode=mode,
            expires_at=time.time() + ttl if expires_at is None else float(expires_at),
            args_digest=call.args_digest,
            context_digest=context_digest,
            context_epoch=self._epoch,
            max_uses=max_uses,
        )
        signature = hmac.new(self._secret, self._payload(unsigned), hashlib.sha256).hexdigest()
        return CapabilityToken(**{**unsigned.__dict__, "signature": signature})

    def verify(self, token: CapabilityToken) -> None:
        expected = hmac.new(self._secret, self._payload(token), hashlib.sha256).hexdigest()
        if not token.signature or not hmac.compare_digest(token.signature, expected):
            raise PermissionError("capability token signature is invalid")
        if token.context_epoch != self._epoch:
            raise PermissionError("capability token belongs to a stale context epoch")


@dataclass(frozen=True)
class ActionDecision:
    call_id: str
    action: str
    reason: str
    lower_risk: float
    upper_risk: float
    upper_unauthorized: float
    disagreement: float
    call_fingerprint: str
    token: Optional[CapabilityToken] = None


class Harness:
    """Trusted runtime gate between an agent and tools."""

    def __init__(
        self,
        credal: CredalSet,
        risk_threshold: float = 0.05,
        confirmation_threshold: float = 0.25,
        cumulative_budget: float = 0.10,
        override_budget: float = 1.0,
        token_ttl: float = 30.0,
        max_repeated_calls: int = 2,
        authority: Optional[CapabilityAuthority] = None,
        authorization_resolver: Optional[AuthorityFn] = None,
    ) -> None:
        self.credal = credal
        self.risk_threshold = float(risk_threshold)
        self.confirmation_threshold = float(confirmation_threshold)
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0, 1]")
        if not 0.0 <= self.confirmation_threshold <= 1.0:
            raise ValueError("confirmation_threshold must be in [0, 1]")
        if self.confirmation_threshold < self.risk_threshold:
            raise ValueError("confirmation_threshold cannot be below risk_threshold")
        self.cumulative_budget = float(cumulative_budget)
        self.spent_budget = 0.0
        self.override_budget = float(override_budget)
        self.spent_override_budget = 0.0
        self.token_ttl = float(token_ttl)
        if self.token_ttl <= 0.0:
            raise ValueError("token_ttl must be positive")
        self.max_repeated_calls = int(max_repeated_calls)
        if self.cumulative_budget < 0.0:
            raise ValueError("cumulative_budget cannot be negative")
        if self.override_budget < 0.0:
            raise ValueError("override_budget cannot be negative")
        if self.max_repeated_calls < 1:
            raise ValueError("max_repeated_calls must be positive")
        self.authority = authority or CapabilityAuthority()
        self.authorization_resolver = authorization_resolver
        self._uses: Dict[str, int] = {}
        self._proposal_counts: Dict[str, int] = {}
        self._evidence_ids: set[str] = set()
        self._audit_seq = 0
        self.audit_log: list[dict[str, Any]] = []

    def _audit(self, event: str, **payload: Any) -> None:
        self._audit_seq += 1
        self.audit_log.append({"seq": self._audit_seq, "epoch": self.authority.epoch, "event": event, **payload})

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.cumulative_budget - self.spent_budget)

    @property
    def remaining_override_budget(self) -> float:
        return max(0.0, self.override_budget - self.spent_override_budget)

    def decide(self, call: ToolCall, state: Mapping[str, Any]) -> ActionDecision:
        proposal_key = json.dumps(
            {"tool": call.tool, "resource": call.resource, "args": dict(call.args)},
            sort_keys=True,
            default=str,
        )
        self._proposal_counts[proposal_key] = self._proposal_counts.get(proposal_key, 0) + 1
        lower, upper = self.credal.risk_bounds(call, state)
        # Authorization is never inferred from the credal risk model.  A
        # missing trusted resolver fails closed.
        upper_unauthorized = 0.0 if (
            self.authorization_resolver is not None and self.authorization_resolver(call, state)
        ) else 1.0
        disagreement = self.credal.disagreement(call, state)

        repeat_limit = 1 if call.irreversible else self.max_repeated_calls
        if not self.credal.coverage_ok:
            action, reason = "deny", "model_containment_invalid_fail_closed"
        elif self._proposal_counts[proposal_key] > repeat_limit:
            action, reason = "deny", "repeated_action_no_progress"
        elif upper_unauthorized > 0.0:
            action, reason = "confirm", "hard_authority_missing"
        elif call.irreversible and upper > self.risk_threshold:
            action, reason = "confirm", "irreversible_risk_above_threshold"
        elif upper <= self.risk_threshold and self.spent_budget + upper <= self.cumulative_budget:
            action, reason = "allow", "upper_risk_within_budget"
        elif not call.irreversible and upper <= self.confirmation_threshold:
            action, reason = "sandbox", "reversible_but_uncertain"
        elif upper <= self.confirmation_threshold:
            action, reason = "confirm", "risk_requires_approval"
        else:
            action, reason = "deny", "upper_risk_too_high"

        token = None
        if action in {"allow", "sandbox"}:
            token = self.authority.issue(
                call,
                action,
                ttl=self.token_ttl,
                context_digest=state_digest(state),
            )
            self._uses[call.call_id] = 0

        decision = ActionDecision(
            call.call_id, action, reason, lower, upper, upper_unauthorized,
            disagreement, call.fingerprint, token,
        )
        token_digest = hashlib.sha256(token.signature.encode("utf-8")).hexdigest() if token else None
        self._audit("decision", call=call, decision=decision, token_digest=token_digest)
        return decision

    def validate_token(
        self,
        token: CapabilityToken,
        call: ToolCall,
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.authority.verify(token)
        if token.call_id != call.call_id or token.tool != call.tool or token.resource != call.resource:
            raise PermissionError("capability token does not match proposed call")
        if token.args_digest != call.args_digest:
            raise PermissionError("capability token does not match call arguments")
        if state is not None and token.context_digest != state_digest(state):
            raise PermissionError("capability token was issued for a stale execution context")
        if time.time() > token.expires_at:
            raise PermissionError("capability token expired")
        if self._uses.get(token.call_id, 0) >= token.max_uses:
            raise PermissionError("capability token exhausted")

    def approve(self, decision: ActionDecision, call: ToolCall, state: Mapping[str, Any]) -> ActionDecision:
        """Issue a fresh live capability after explicit operator approval."""
        if (
            decision.action != "confirm"
            or decision.call_id != call.call_id
            or decision.call_fingerprint != call.fingerprint
        ):
            raise PermissionError("only the matching confirmation can be approved")
        if not self.credal.coverage_ok:
            raise PermissionError("cannot approve after model-containment invalidation")
        lower, upper = self.credal.risk_bounds(call, state)
        if self.authorization_resolver is None or not self.authorization_resolver(call, state):
            raise PermissionError("operator approval does not replace authoritative scope validation")
        if self.spent_override_budget + upper > self.override_budget:
            raise PermissionError("operator risk-override budget exhausted")
        upper_unauthorized = 0.0
        disagreement = self.credal.disagreement(call, state)
        token = self.authority.issue(
            call,
            "allow",
            ttl=self.token_ttl,
            context_digest=state_digest(state),
        )
        self._uses[call.call_id] = 0
        approved = ActionDecision(
            call.call_id, "allow", "explicit_operator_approval", lower, upper,
            upper_unauthorized, disagreement, call.fingerprint, token,
        )
        self._audit(
            "approval", call=call, decision=approved,
            token_digest=hashlib.sha256(token.signature.encode("utf-8")).hexdigest(),
        )
        return approved

    def commit(self, decision: ActionDecision, call: ToolCall, observed_harm: bool = False) -> None:
        if decision.action not in {"allow", "sandbox"} or decision.token is None:
            raise PermissionError("only allow/sandbox decisions can be committed")
        if decision.token.mode != decision.action:
            raise PermissionError("capability mode does not match decision")
        self.validate_token(decision.token, call)
        self._uses.setdefault(decision.token.call_id, 0)
        self._uses[decision.token.call_id] += 1
        if decision.action == "allow" and decision.reason == "explicit_operator_approval":
            self.spent_override_budget += decision.upper_risk
        elif decision.action == "allow":
            self.spent_budget += decision.upper_risk
        self._audit("commit", call=call, harm=bool(observed_harm))
        if observed_harm:
            self.credal = self.credal.invalidate()
            self.authority.bump_epoch()

    def observe(self, evidence: Evidence, likelihoods: Optional[Mapping[str, float]] = None) -> bool:
        if evidence.event_id in self._evidence_ids:
            self._audit("evidence_duplicate_ignored", evidence_id=evidence.event_id)
            return False
        self._evidence_ids.add(evidence.event_id)
        coverage_before = self.credal.coverage_ok
        self.credal = self.credal.update(evidence, likelihoods)
        if coverage_before and not self.credal.coverage_ok:
            self.authority.bump_epoch()
        self._audit("evidence", evidence=evidence, evidence_id=evidence.event_id, coverage_ok=self.credal.coverage_ok)
        return True

    def revalidate(self, state: Mapping[str, Any]) -> None:
        """Force a fresh decision boundary after a state/authority change."""
        _ = state
        # Revalidation is different from an unexplained model mismatch: the
        # harness remains operational, but broadens uncertainty until fresh
        # state/authority evidence is collected.
        self.credal = CredalSet(
            dict(self.credal.hypotheses),
            dict(self.credal.weights),
            epsilon=max(self.credal.epsilon, 0.20),
            group_epsilons={group: max(value, 0.20) for group, value in self.credal.group_epsilons.items()},
            coverage_ok=True,
            calibration=dict(self.credal.calibration),
        )
        self.authority.bump_epoch()
        self._audit("revalidate", reason="stale_context_or_state_change", state_digest=state_digest(state))


class RollbackSandbox:
    """Executable stateful tool sandbox with commit/rollback semantics."""

    def __init__(
        self,
        initial_state: Optional[Mapping[str, Any]] = None,
        authority: Optional[CapabilityAuthority] = None,
    ) -> None:
        self.state: Dict[str, Any] = copy.deepcopy(dict(initial_state or {}))
        self.tools: Dict[str, Callable[[MutableMapping[str, Any], Mapping[str, Any]], Any]] = {}
        self.authority = authority or CapabilityAuthority()
        self._token_uses: Dict[tuple[str, str, float], int] = {}

    def register(self, name: str, fn: Callable[[MutableMapping[str, Any], Mapping[str, Any]], Any]) -> None:
        self.tools[name] = fn

    def execute(self, call: ToolCall, token: CapabilityToken, commit: bool = False) -> tuple[Any, Dict[str, Any]]:
        self.authority.verify(token)
        if token.call_id != call.call_id or token.tool != call.tool or token.resource != call.resource:
            raise PermissionError("capability token does not match proposed call")
        if token.args_digest != call.args_digest:
            raise PermissionError("capability token does not match call arguments")
        if token.context_digest != state_digest(self.state):
            raise PermissionError("capability token was issued for a stale execution context")
        if time.time() > token.expires_at:
            raise PermissionError("capability token expired")
        if token.mode not in {"allow", "sandbox"}:
            raise PermissionError("unsupported capability mode")
        if commit and token.mode != "allow":
            raise PermissionError("sandbox capability cannot commit state")
        token_key = (token.call_id, token.mode, token.expires_at)
        if self._token_uses.get(token_key, 0) >= token.max_uses:
            raise PermissionError("capability token exhausted")
        if call.tool not in self.tools:
            raise KeyError(f"unknown tool: {call.tool}")
        before = copy.deepcopy(self.state)
        working = copy.deepcopy(self.state)
        result = self.tools[call.tool](working, call.args)
        diff = _state_diff(before, working)
        if commit:
            self.state = working
        self._token_uses[token_key] = self._token_uses.get(token_key, 0) + 1
        return result, diff


def _state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    keys = set(before) | set(after)
    return {k: {"before": before.get(k), "after": after.get(k)} for k in sorted(keys) if before.get(k) != after.get(k)}

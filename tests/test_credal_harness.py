import time

import unittest

from credal_harness import CapabilityAuthority, CapabilityToken, CredalSet, Evidence, Harness, Hypothesis, RollbackSandbox, ToolCall, state_digest
from experiments.run_agent_harness import execute_episode, SCENARIOS
from experiments.run_extended_studies import score, coverage_loss_seed


class CredalHarnessTests(unittest.TestCase):
  AUTHORIZED = staticmethod(lambda c, s: True)
  def test_upper_expectation_is_conservative(self):
    hs = [
        Hypothesis("safe", lambda c, s: 0.0),
        Hypothesis("unsafe", lambda c, s: 1.0),
    ]
    k = CredalSet.uniform(hs, epsilon=0.2)
    call = ToolCall("x")
    lower, upper = k.risk_bounds(call, {})
    self.assertAlmostEqual(lower, 0.4)
    self.assertAlmostEqual(upper, 0.6)

  def test_vacuous_contamination_reserves_unmodeled_harm(self):
    hs = [
        Hypothesis("safe_a", lambda c, s: 0.0),
        Hypothesis("safe_b", lambda c, s: 0.0),
    ]
    k = CredalSet.uniform(hs, epsilon=0.2)
    lower, upper = k.risk_bounds(ToolCall("x"), {})
    self.assertAlmostEqual(lower, 0.0)
    self.assertAlmostEqual(upper, 0.2)

  def test_hard_authority_is_not_replaced_by_credal_allowance(self):
    k = CredalSet.uniform([Hypothesis("permissive", lambda c, s: 0.0)])
    authority = CapabilityAuthority()
    h = Harness(k, authority=authority, authorization_resolver=lambda c, s: False)
    call = ToolCall("delete_note", {"key": "k"}, irreversible=True)
    decision = h.decide(call, {})
    self.assertEqual(decision.action, "confirm")
    with self.assertRaises(PermissionError):
      h.approve(decision, call, {})

  def test_evidence_updates_are_idempotent(self):
    k = CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0), Hypothesis("unsafe", lambda c, s: 1.0)])
    h = Harness(k, authorization_resolver=self.AUTHORIZED)
    evidence = Evidence("trusted_shadow_outcome", event_id="event-1")
    self.assertTrue(h.observe(evidence, {"safe": 0.9, "unsafe": 0.1}))
    weights = dict(h.credal.weights)
    self.assertFalse(h.observe(evidence, {"safe": 0.1, "unsafe": 0.9}))
    self.assertEqual(weights, h.credal.weights)

  def test_context_epoch_invalidates_old_capability(self):
    authority = CapabilityAuthority()
    h = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), authority=authority, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("read")
    decision = h.decide(call, {})
    authority.bump_epoch()
    with self.assertRaises(PermissionError):
      h.validate_token(decision.token, call, {})

  def test_calibration_learns_weights_and_radius(self):
    hs = [
        Hypothesis("safe", lambda c, s: 0.01),
        Hypothesis("risky", lambda c, s: 0.80),
    ]
    call = ToolCall("write_note", {"key": "k", "value": "v"}, resource="note:k", effect="reversible")
    records = [{"call": call, "state": {"version": 1}, "harm": False, "group": "reversible"} for _ in range(20)]
    learned = CredalSet.fit_from_records(hs, records, alpha=0.10, prior_epsilon=0.02)
    self.assertGreater(learned.weights["safe"], learned.weights["risky"])
    self.assertGreaterEqual(learned.epsilon, 0.02)
    self.assertEqual(learned.calibration["records"], 20)


  def test_harness_fails_closed_after_model_mismatch(self):
    k = CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)])
    h = Harness(k, risk_threshold=0.05, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("read")
    self.assertEqual(h.decide(call, {}).action, "allow")
    h.observe(Evidence("model_mismatch"))
    self.assertEqual(h.decide(call, {}).action, "deny")


  def test_capability_token_is_bound_to_call(self):
    k = CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)])
    h = Harness(k, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("read", resource="account")
    decision = h.decide(call, {})
    self.assertIsNotNone(decision.token)
    with self.assertRaises(PermissionError):
        h.validate_token(decision.token, ToolCall("read", resource="other"))


  def test_rollback_sandbox_does_not_commit_by_default(self):
    authority = CapabilityAuthority()
    box = RollbackSandbox({"x": 0}, authority=authority)
    box.register("inc", lambda state, args: state.__setitem__("x", state["x"] + 1))
    call = ToolCall("inc", effect="reversible")
    token = authority.issue(call, "sandbox", ttl=60, context_digest=state_digest(box.state))
    box.execute(call, token, commit=False)
    self.assertEqual(box.state["x"], 0)
    with self.assertRaises(PermissionError):
      box.execute(call, token, commit=True)
    allow_call = ToolCall("inc", effect="reversible")
    allow_token = authority.issue(allow_call, "allow", ttl=60, context_digest=state_digest(box.state))
    box.execute(allow_call, allow_token, commit=True)
    self.assertEqual(box.state["x"], 1)
    with self.assertRaises(PermissionError):
      box.execute(allow_call, allow_token, commit=True)
    with self.assertRaises(PermissionError):
        h = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), authorization_resolver=self.AUTHORIZED)
        h.commit(h.decide(ToolCall("read"), {}), ToolCall("read"), observed_harm=False)

  def test_confirmation_requires_harness_approval_token(self):
    k = CredalSet.uniform([Hypothesis("unsafe", lambda c, s: 0.2)])
    h = Harness(k, risk_threshold=0.05, confirmation_threshold=0.25, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("delete", irreversible=True)
    confirmation = h.decide(call, {})
    self.assertEqual(confirmation.action, "confirm")
    self.assertIsNone(confirmation.token)
    approved = h.approve(confirmation, call, {})
    self.assertEqual(approved.token.mode, "allow")
    h.validate_token(approved.token, call)
    altered = ToolCall("delete", {"scope": "all"}, irreversible=True, call_id=call.call_id)
    with self.assertRaises(PermissionError):
      h.approve(confirmation, altered, {})

  def test_confirmation_uses_separate_override_budget(self):
    k = CredalSet.uniform([Hypothesis("risk", lambda c, s: 0.2)])
    h = Harness(k, risk_threshold=0.05, confirmation_threshold=0.4,
                cumulative_budget=0.1, override_budget=0.3,
                authorization_resolver=self.AUTHORIZED)
    call = ToolCall("write", {"value": 1}, effect="irreversible", irreversible=True)
    approved = h.approve(h.decide(call, {}), call, {})
    h.commit(approved, call)
    self.assertAlmostEqual(h.spent_budget, 0.0)
    self.assertAlmostEqual(h.spent_override_budget, 0.2)

  def test_capability_token_is_bound_to_arguments(self):
    h = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), authorization_resolver=self.AUTHORIZED)
    call = ToolCall("write", {"value": "safe"}, resource="note")
    decision = h.decide(call, {})
    altered = ToolCall("write", {"value": "dangerous"}, resource="note", call_id=call.call_id)
    with self.assertRaises(PermissionError):
      h.validate_token(decision.token, altered)

  def test_forged_capability_is_rejected_by_executor(self):
    authority = CapabilityAuthority()
    box = RollbackSandbox({"x": 0}, authority=authority)
    box.register("inc", lambda state, args: state.__setitem__("x", state["x"] + 1))
    call = ToolCall("inc", effect="reversible")
    forged = CapabilityToken(
      call.call_id, call.tool, call.resource, "allow", time.time() + 60,
      args_digest=call.args_digest, context_digest=state_digest(box.state)
    )
    with self.assertRaises(PermissionError):
      box.execute(call, forged, commit=True)

  def test_state_change_invalidates_issued_capability(self):
    authority = CapabilityAuthority()
    box = RollbackSandbox({"version": 1}, authority=authority)
    box.register("read", lambda state, args: state["version"])
    harness = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), authority=authority, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("read", effect="read")
    decision = harness.decide(call, box.state)
    box.state["version"] = 2
    with self.assertRaises(PermissionError):
      box.execute(call, decision.token, commit=False)

  def test_multistep_harness_blocks_injection_and_detects_drift(self):
    class OfflineAgent:
      def propose(self, task, transcript):
        if "IGNORE" in task:
          return {"action": "tool", "tool": "read_note", "args": {"key": "k2"}}
        return {"action": "tool", "tool": "read_note", "args": {"key": "k3"}}
    scenarios = {scenario["id"]: scenario for scenario in SCENARIOS}
    injection = execute_episode(scenarios["prompt_injection"], OfflineAgent(), live=False)
    drift = execute_episode(scenarios["contract_drift"], OfflineAgent(), live=False)
    self.assertEqual(injection["harm"], 0)
    self.assertEqual(drift["coverage_loss"], 1)
    self.assertEqual(drift["final_state"]["version"], 1)
    self.assertTrue(drift["rows"][0]["prevented_contract_violation"])

  def test_repeated_call_is_denied(self):
    h = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), max_repeated_calls=2, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("read", {"key": "x"})
    self.assertEqual(h.decide(call, {}).action, "allow")
    self.assertEqual(h.decide(call, {}).action, "allow")
    self.assertEqual(h.decide(call, {}).reason, "repeated_action_no_progress")

  def test_irreversible_repeat_is_denied_after_one_proposal(self):
    h = Harness(CredalSet.uniform([Hypothesis("safe", lambda c, s: 0.0)]), max_repeated_calls=2, authorization_resolver=self.AUTHORIZED)
    call = ToolCall("send", {"message": "x"}, irreversible=True)
    self.assertEqual(h.decide(call, {}).action, "allow")
    self.assertEqual(h.decide(call, {}).reason, "repeated_action_no_progress")

  def test_unknown_tool_invalidates_coverage_in_agent_loop(self):
    class OfflineAgent:
      def propose(self, task, transcript):
        return {"action": "tool", "tool": "publish_note", "args": {"key": "k5"}}

    scenario = next(s for s in SCENARIOS if s["id"] == "unknown_tool_request")
    result = execute_episode(scenario, OfflineAgent(), live=False)
    self.assertEqual(result["coverage_loss"], 1)

  def test_agent_cannot_self_authorize_with_tool_arguments(self):
    scenario = next(s for s in SCENARIOS if s["id"] == "self_authorization_claim")
    result = execute_episode(scenario, object(), live=False)
    self.assertEqual(result["harm"], 0)
    self.assertEqual(result["coverage_loss"], 1)
    self.assertEqual(result["final_state"]["notes"]["k7"], "retain")
    self.assertEqual(result["rows"][0]["event"], "invalid_arguments")

  def test_context_digest_blocks_state_race_without_commit(self):
    scenario = next(s for s in SCENARIOS if s["id"] == "state_race")
    result = execute_episode(scenario, object(), live=False)
    self.assertEqual(result["harm"], 0)
    self.assertEqual(result["final_state"]["version"], 2)
    self.assertEqual(result["rows"][0]["event"], "stale_capability")

  def test_context_digest_blocks_authorization_revocation(self):
    scenario = next(s for s in SCENARIOS if s["id"] == "authorization_revocation_race")
    result = execute_episode(scenario, object(), live=False)
    self.assertEqual(result["harm"], 0)
    self.assertEqual(result["final_state"]["notes"]["k8"], "protected-until-approved")
    self.assertEqual(result["rows"][0]["event"], "stale_capability")

  def test_credal_routes_high_disagreement_but_allows_safe_writes(self):
    safe = (0.005, 0.010, 0.010)
    ambiguous = (0.010, 0.010, 0.800)
    self.assertLessEqual(score("credal", safe), 0.05)
    self.assertGreater(score("credal", ambiguous), 0.05)
    self.assertLessEqual(score("point", ambiguous), 0.05)

  def test_shadow_probing_reduces_omitted_world_harm(self):
    reactive = coverage_loss_seed(0, omitted_rate=0.10, probe_rate=0.0)
    probed = coverage_loss_seed(0, omitted_rate=0.10, probe_rate=0.20)
    self.assertLess(probed["harm_per_proposal"], reactive["harm_per_proposal"])


if __name__ == "__main__":
    unittest.main()

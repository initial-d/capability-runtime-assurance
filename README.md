# Capability-Mediated Runtime Assurance

Reference implementation and reproducible experiments for mediating
side-effectful tool calls proposed by untrusted agents.

The runtime separates proposal authority from commit authority. It provides:

- deterministic authorization outside the risk model;
- one-use capabilities bound to canonical calls, resources, and context;
- executor-side revalidation at commit;
- rollback isolation and contract-drift detection;
- uncertainty-aware allow, sandbox, confirm, and deny routing;
- sequence-level risk accounting and auditable invalidation.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make test
make all
```

Generated outputs are written to `experiments/results/` and are intentionally
ignored by Git.

## Layout

- `credal_harness/core.py`: reference monitor, ambiguity model, capability
  issuance, commit revalidation, rollback sandbox, and audit records.
- `credal_harness/agent.py`: optional OpenAI-compatible hosted-model adapter.
- `experiments/run_sqlite_transaction_benchmark.py`: transactional race and
  compare-and-swap validation.
- `experiments/run_policy_benchmark.py`: identical-proposal governance replay.
- `experiments/run_credal_harness.py`: controlled stateful evaluation.
- `experiments/run_extended_studies.py`: calibration, selective-routing,
  model-containment, and sequence-budget studies.
- `tests/`: deterministic regression tests for the enforcement boundary.

## Optional hosted evaluation

No credentials, provider endpoints, model responses, or local paths are stored
in this repository. Set the authorization header through the environment and
pass the endpoint and model explicitly:

```bash
export HARNESS_API_AUTHORIZATION='Bearer ...'
python -m experiments.run_agent_harness --live \
  --endpoint 'https://provider.example/v1/chat/completions' \
  --model 'provider/model'
```

AgentDojo integration is isolated because it requires Python 3.10 or later:

```bash
python3.12 -m venv .agentdojo-venv
.agentdojo-venv/bin/pip install -r requirements-agentdojo.txt
export OPENAI_API_KEY='...'
PYTHONPATH=. .agentdojo-venv/bin/python -m experiments.run_agentdojo_benchmark \
  --base-url 'https://provider.example/v1' \
  --models 'provider/model' --policies open credal
```

## Scope

This repository contains software and experiment code only. It excludes
identity metadata, proprietary datasets, and organization-specific configuration.

## License

MIT

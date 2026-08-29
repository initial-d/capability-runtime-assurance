# Locally Adaptive Credal Risk Routing

Reference implementation and reproducible experiments for locally adaptive
risk routing and capability-mediated execution of side-effectful tool calls.

The runtime separates proposal authority from commit authority. It provides:

- deterministic authorization outside the risk model;
- one-use capabilities bound to canonical calls, resources, and context;
- executor-side revalidation at commit;
- rollback isolation and contract-drift detection;
- uncertainty-aware allow, sandbox, confirm, and deny routing;
- multi-scale local residual calibration and query-specific credal radii;
- independent split-selective risk control;
- sequence-level risk accounting and auditable invalidation.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make test
make all
```

Generated outputs are written to `experiments/results/`. The complete adaptive
benchmark result is committed; other regenerable outputs remain ignored.

## Layout

- `credal_harness/core.py`: reference monitor, ambiguity model, capability
  issuance, commit revalidation, rollback sandbox, and audit records.
- `credal_harness/adaptive.py`: locally adaptive credal certificates,
  split-selective risk control, and sequence-budget routing.
- `credal_harness/agent.py`: optional OpenAI-compatible hosted-model adapter.
- `experiments/run_sqlite_transaction_benchmark.py`: transactional race and
  compare-and-swap validation.
- `experiments/run_policy_benchmark.py`: identical-proposal governance replay.
- `experiments/run_credal_harness.py`: controlled stateful evaluation.
- `experiments/run_extended_studies.py`: calibration, selective-routing,
  model-containment, and sequence-budget studies.
- `experiments/run_adaptive_benchmark.py`: frozen 30-seed protocol with
  matched-coverage, independent split-control, and shift evaluations.
- `experiments/results/adaptive_benchmark.json`: complete records,
  protocol hash, paired comparisons, and risk-control summaries.
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

## Adaptive benchmark

```bash
make adaptive
```

The run uses 30 seeds, seven policies, three protocol-specified shift levels,
independent fit/design/policy/test splits, and 1.8 million held-out proposals.
For a fast installation check, run
`python -m experiments.run_adaptive_benchmark --quick`; quick output must
not replace the committed full result.

## License

MIT

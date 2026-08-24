.PHONY: test reproduce extensions sqlite sensitivity offline replay all

PYTHON ?= python3

test:
	$(PYTHON) tests/run_tests.py

reproduce:
	$(PYTHON) -m experiments.run_credal_harness

extensions:
	$(PYTHON) -m experiments.run_extended_studies

sqlite:
	$(PYTHON) -m experiments.run_sqlite_transaction_benchmark

sensitivity:
	$(PYTHON) -m experiments.run_sensitivity

offline:
	$(PYTHON) -m experiments.run_agent_harness

replay: offline
	$(PYTHON) -m experiments.run_policy_benchmark --traces experiments/results/agent_harness_results.json

all: test reproduce extensions sqlite sensitivity offline replay

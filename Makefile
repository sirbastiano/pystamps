PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag inputs_and_outputs/InSAR_dataset_small_baseline_stage7
AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json
VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10
VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test
BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag

.PHONY: setup test test-impl build twine-check audit verify benchmark parity-loop

setup:
	uv sync

test:
	uv run pytest -q

test-impl:
	uv run pytest -q tests/test_cli.py tests/test_verify.py tests/test_validate_audit.py tests/test_stage7_ported.py tests/test_kernels_accelerated.py tests/test_dataset.py

build:
	uv run --with build python -m build --sdist --wheel

twine-check:
	uv run --with twine python -m twine check dist/*

audit:
	$(PARITY_ENV) uv run python scripts/validate_audit.py \
		--datasets $(AUDIT_DATASETS) \
		--output $(AUDIT_OUTPUT)

verify:
	$(PARITY_ENV) uv run pystamps verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)

benchmark:
	uv run python scripts/benchmark_backends.py \
		--dataset $(BENCHMARK_DATASET) \
		--start-step 1 --end-step 8 \
		--repeat 3 --warmup 1

parity-loop:
	$(PARITY_ENV) uv run python scripts/parity_bug_loop.py \
		--datasets $(AUDIT_DATASETS) \
		--audit-output $(AUDIT_OUTPUT) \
		--output inputs_and_outputs/validation_runs/latest_parity_loop.json

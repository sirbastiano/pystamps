PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag inputs_and_outputs/InSAR_dataset_test inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag inputs_and_outputs/InSAR_dataset_small_baseline_stage7
AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json
VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10
VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test

# Native VM reproduction:
#   make deps-ubuntu && make deps-check
#   uv run python -c "import h5py, mat73"
#   cargo build --release -p pystamps-core --bin pystamps-native
#   make native-full-chain-verify
#
# DATASET is copied to RUN before execution; GOLDEN is the parity reference.
# RUN is removed/recreated and must stay outside DATASET. Reports are written to
# $(RUN)/_native_gate_reports/{native-coverage-report.json,native-run-report.json,
# native-run-timings.json,native-verify-report.json}. THREADS=0 lets the native
# runner use all available processing threads; THREADS=N pins --cpu-workers and
# --stage2-native-threads. START_STEP/END_STEP scope focused gates.
#
# MAT/HDF5 support: native Rust uses the vendored pure-Rust HDF5/MAT support and
# Python verification uses uv-managed h5py/mat73. Performance waivers are checked
# in $(PERFORMANCE_BUDGETS) as temporary_waiver objects with reason, owner, and
# future expires_at_utc.
DATASET ?= inputs_and_outputs/InSAR_dataset_test
RUN ?= inputs_and_outputs/validation_runs/native-full-chain
GOLDEN ?= inputs_and_outputs/InSAR_dataset_test
THREADS ?= 0
START_STEP ?= 1
END_STEP ?= 8
NATIVE_BIN ?= target/release/pystamps-native
PERFORMANCE_BUDGETS ?= pystamps/data/native_performance_budgets.json
BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag
PYPI_PROJECT ?= pystamps-insar
VERSION ?=
DIST_DIR ?= dist
RELEASE_DIST_DIR ?= /tmp/pystamps-insar-dist
WHEELHOUSE ?= /tmp/pystamps-insar-wheelhouse
PUBLISH_DIST_DIR ?= $(DIST_DIR)
TWINE_USERNAME ?= __token__
TWINE_PASSWORD ?= $(UV_PUBLISH_TOKEN)
TWINE_REPOSITORY ?=
TWINE_REPOSITORY_ARGS = $(if $(TWINE_REPOSITORY),--repository $(TWINE_REPOSITORY),)
UV_INSTALL_URL ?= https://astral.sh/uv/install.sh
RUSTUP_INSTALL_URL ?= https://sh.rustup.rs
APT_NATIVE_DEPS ?= build-essential curl pkg-config python3 python3-dev

.PHONY: deps deps-python deps-rust deps-ubuntu deps-check setup test test-impl build clean-dist require-version require-publish-token require-publish-files next-patch bump release-build repair-wheel release-check publish publish-dist-check publish-dist publish-testpypi release release-patch audit verify benchmark parity-loop web native-release-bin native-full-chain-run native-full-chain-verify

deps: deps-rust deps-python

deps-python:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv"; \
		curl -LsSf "$(UV_INSTALL_URL)" | sh; \
	fi
	@UV_BIN=$$(command -v uv || printf '%s' "$$HOME/.local/bin/uv"); \
	"$$UV_BIN" sync

deps-rust:
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "Installing Rust with rustup"; \
		curl --proto '=https' --tlsv1.2 -sSf "$(RUSTUP_INSTALL_URL)" | sh -s -- -y; \
	fi
	@. "$$HOME/.cargo/env" 2>/dev/null || true; \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "cargo was not found after Rust install"; \
		exit 2; \
	fi; \
	if ! command -v rustup >/dev/null 2>&1; then \
		echo "rustup is required to install rustfmt and clippy components"; \
		exit 2; \
	fi; \
	RUSTUP_PERMIT_COPY_RENAME=1 rustup component add rustfmt clippy

deps-ubuntu:
	sudo apt-get update
	sudo apt-get install -y $(APT_NATIVE_DEPS)
	$(MAKE) deps

deps-check:
	@command -v cargo >/dev/null 2>&1 || { echo "missing cargo"; exit 2; }
	@command -v rustfmt >/dev/null 2>&1 || { echo "missing rustfmt"; exit 2; }
	@cargo clippy --version >/dev/null 2>&1 || { echo "missing clippy"; exit 2; }
	@UV_BIN=$$(command -v uv || printf '%s' "$$HOME/.local/bin/uv"); \
	"$$UV_BIN" --version >/dev/null 2>&1 || { echo "missing uv"; exit 2; }

setup: deps-python

test:
	uv run pytest -q

test-impl:
	uv run pytest -q tests/test_cli.py tests/test_verify.py tests/test_validate_audit.py tests/test_stage7_ported.py tests/test_kernels_accelerated.py tests/test_dataset.py

build:
	uv run --with build python -m build --sdist --wheel

twine-check:
	uv run --with twine python -m twine check dist/*

clean-dist:
	rm -rf $(RELEASE_DIST_DIR) $(WHEELHOUSE)

require-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=0.1.3"; \
		exit 2; \
	fi

require-publish-token:
	@if [ -z "$(TWINE_PASSWORD)" ]; then \
		echo "Set UV_PUBLISH_TOKEN or TWINE_PASSWORD before publishing"; \
		exit 2; \
	fi

require-publish-files:
	@python -c 'import glob, sys; files=[f for p in sys.argv[1:] for f in glob.glob(p)]; sys.exit(0 if files else 2)' "$(PUBLISH_DIST_DIR)/*.whl" "$(PUBLISH_DIST_DIR)/*.tar.gz" || { \
		echo "No distributions found in $(PUBLISH_DIST_DIR)"; \
		exit 2; \
	}

next-patch:
	@python -c 'import json, urllib.request; v=json.load(urllib.request.urlopen("https://pypi.org/pypi/$(PYPI_PROJECT)/json", timeout=15))["info"]["version"].split("."); print(f"{v[0]}.{v[1]}.{int(v[2]) + 1}")'

bump: require-version
	@echo "Preparing release version $(VERSION)"
	@git tag -a "v$(VERSION)" -m "Release $(VERSION)"

release-build: clean-dist require-version
	SETUPTOOLS_SCM_PRETEND_VERSION=$(VERSION) \
		uv run --with build --with setuptools-scm --with setuptools-rust \
		python -m build --sdist --wheel -o $(RELEASE_DIST_DIR)

repair-wheel: release-build
	mkdir -p $(WHEELHOUSE)
	uv run --no-project --with auditwheel --with patchelf auditwheel repair \
		$(RELEASE_DIST_DIR)/*-linux_x86_64.whl \
		-w $(WHEELHOUSE)

release-check: repair-wheel
	uv run --no-project --with twine python -m twine check \
		$(WHEELHOUSE)/*.whl \
		$(RELEASE_DIST_DIR)/*.tar.gz

publish: release-check require-publish-token
	@TWINE_USERNAME=$(TWINE_USERNAME) TWINE_PASSWORD=$(TWINE_PASSWORD) \
		uv run --no-project --with twine python -m twine upload \
		$(WHEELHOUSE)/*.whl \
		$(RELEASE_DIST_DIR)/*.tar.gz

publish-dist-check: require-publish-files
	@files=$$(python -c 'import glob, sys; print(" ".join(f for p in sys.argv[1:] for f in glob.glob(p)))' "$(PUBLISH_DIST_DIR)/*.whl" "$(PUBLISH_DIST_DIR)/*.tar.gz"); \
	uv run --no-project --with twine python -m twine check $$files

publish-dist: publish-dist-check require-publish-token
	@files=$$(python -c 'import glob, sys; print(" ".join(f for p in sys.argv[1:] for f in glob.glob(p)))' "$(PUBLISH_DIST_DIR)/*.whl" "$(PUBLISH_DIST_DIR)/*.tar.gz"); \
	TWINE_USERNAME=$(TWINE_USERNAME) TWINE_PASSWORD=$(TWINE_PASSWORD) \
		uv run --no-project --with twine python -m twine upload \
		$(TWINE_REPOSITORY_ARGS) \
		$$files

publish-testpypi:
	$(MAKE) publish-dist TWINE_REPOSITORY=testpypi

release: publish

release-patch:
	@VERSION=$$($(MAKE) --no-print-directory next-patch); \
	echo "Releasing $(PYPI_PROJECT) $$VERSION"; \
	$(MAKE) release VERSION=$$VERSION

audit:
	$(PARITY_ENV) uv run python scripts/validate_audit.py \
		--datasets $(AUDIT_DATASETS) \
		--output $(AUDIT_OUTPUT)

verify:
	$(PARITY_ENV) uv run pystamps verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)

native-release-bin:
	cargo build --release -p pystamps-core --bin pystamps-native

native-full-chain-run: native-release-bin
	$(PARITY_ENV) uv run python scripts/native_full_chain_gate.py run \
		--dataset "$(DATASET)" \
		--run "$(RUN)" \
		--native-bin "$(NATIVE_BIN)" \
		--threads "$(THREADS)" \
		--start-step "$(START_STEP)" \
		--end-step "$(END_STEP)" \
		--budget-manifest "$(PERFORMANCE_BUDGETS)"

native-full-chain-verify: native-release-bin
	$(PARITY_ENV) uv run python scripts/native_full_chain_gate.py verify \
		--dataset "$(DATASET)" \
		--run "$(RUN)" \
		--golden "$(GOLDEN)" \
		--native-bin "$(NATIVE_BIN)" \
		--threads "$(THREADS)" \
		--start-step "$(START_STEP)" \
		--end-step "$(END_STEP)" \
		--budget-manifest "$(PERFORMANCE_BUDGETS)"

benchmark:
	uv run python scripts/benchmark_backends.py \
		--dataset $(BENCHMARK_DATASET) \
		--start-step 1 --end-step 8 \
		--repeat 3 --warmup 1

web:
	cargo run -p pystamps-web

parity-loop:
	$(PARITY_ENV) uv run python scripts/parity_bug_loop.py \
		--datasets $(AUDIT_DATASETS) \
		--audit-output $(AUDIT_OUTPUT) \
		--output inputs_and_outputs/validation_runs/latest_parity_loop.json

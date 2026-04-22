from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
TESTING_DOC = (REPO_ROOT / "docs" / "testing.html").read_text(encoding="utf-8")
RELEASE_DOC = (REPO_ROOT / "docs" / "release.md").read_text(encoding="utf-8")
MANIFEST = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_readme_documents_fresh_clone_validation_separately() -> None:
    assert "Fresh-clone validation commands:" in README
    assert "uv run pytest -q" in README
    assert "uv run --with build python -m build --sdist --wheel" in README
    assert "uv run --with twine python -m twine check dist/*" in README
    assert "Rust toolchain" in README
    assert "platform wheels for the Rust extension" in README
    assert "uv run python scripts/validate_audit.py \\\n  --datasets inputs_and_outputs/InSAR_dataset_test" not in README
    assert "inputs_and_outputs/InSAR_dataset_test_stage8diag" in README
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag" in README
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7" in README
    assert "optional repo assets" in README


def test_readme_and_makefile_expose_the_same_local_entrypoints() -> None:
    assert "make setup" in README
    assert "make test" in README
    assert "make build" in README
    assert "make twine-check" in README
    assert "make audit" in README
    assert "make verify" in README
    assert "make benchmark" in README

    assert ".PHONY: setup test test-impl build twine-check audit verify benchmark" in MAKEFILE
    assert "PARITY_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=." in MAKEFILE
    assert (
        "AUDIT_DATASETS = inputs_and_outputs/InSAR_dataset_test_stage8diag "
        "inputs_and_outputs/InSAR_dataset_test "
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag "
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7"
    ) in MAKEFILE
    assert "AUDIT_OUTPUT = inputs_and_outputs/validation_runs/latest_audit.json" in MAKEFILE
    assert "VERIFY_RUN = inputs_and_outputs/RUN_FULL_GATE_1e10" in MAKEFILE
    assert "VERIFY_GOLDEN = inputs_and_outputs/InSAR_dataset_test" in MAKEFILE
    assert "BENCHMARK_DATASET = inputs_and_outputs/InSAR_dataset_test_stage8diag" in MAKEFILE
    assert "uv sync" in MAKEFILE
    assert "uv run pytest -q" in MAKEFILE
    assert "uv run --with build python -m build --sdist --wheel" in MAKEFILE
    assert "uv run --with twine python -m twine check dist/*" in MAKEFILE
    assert "uv run python scripts/validate_audit.py" in MAKEFILE
    assert "uv run pystamps verify --run $(VERIFY_RUN) --golden $(VERIFY_GOLDEN)" in MAKEFILE
    assert "uv run python scripts/benchmark_backends.py" in MAKEFILE


def test_release_docs_reference_the_supported_parity_gate() -> None:
    assert "make audit" not in RELEASE_DOC
    assert "make verify" not in RELEASE_DOC
    assert "Do not substitute a Makefile target" in RELEASE_DOC
    assert "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=." in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_test_stage8diag" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_test" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag" in RELEASE_DOC
    assert "inputs_and_outputs/InSAR_dataset_small_baseline_stage7" in RELEASE_DOC
    assert "--output inputs_and_outputs/validation_runs/latest_audit.json" in RELEASE_DOC
    assert "run_root" in RELEASE_DOC
    assert "manual restart" in RELEASE_DOC
    assert "Rust toolchain" in RELEASE_DOC
    assert "cibuildwheel" in RELEASE_DOC
    assert "platform wheels" in RELEASE_DOC


def test_testing_docs_call_out_optional_dataset_workflows() -> None:
    assert "skip cleanly when the local validation datasets are absent" in TESTING_DOC
    assert "do not guess a Makefile, CI wrapper, or reduced audit dataset list" in TESTING_DOC
    assert "documented audited dataset set" in TESTING_DOC
    assert "uv run --with build python -m build --sdist --wheel" in TESTING_DOC
    assert "uv run --with twine python -m twine check dist/*" in TESTING_DOC
    assert "uv run --with cibuildwheel python -m cibuildwheel --platform" in TESTING_DOC
    assert "uv run pystamps verify --run RUN_COPY --golden ./inputs_and_outputs/InSAR_dataset_test" in TESTING_DOC
    assert "latest_audit.json" in TESTING_DOC
    assert "stale-output reuse keeps the validation gate red" in TESTING_DOC


def test_manifest_excludes_generated_release_artifacts() -> None:
    assert "prune dist" in MANIFEST
    assert "prune build" in MANIFEST
    assert "prune .codex" in MANIFEST
    assert "prune .github" in MANIFEST


def test_packaging_contract_prefers_rust_sources_and_excludes_cython_package_data() -> None:
    assert "include Cargo.toml" in MANIFEST
    assert "recursive-include src *.rs" in MANIFEST
    assert "recursive-include pystamps/data *.json" in MANIFEST
    assert "recursive-include pystamps *.pyx" not in MANIFEST
    assert "include-package-data = false" in PYPROJECT
    assert '[tool.setuptools.package-data]\npystamps = ["data/*.json"]' in PYPROJECT
    assert "setuptools-rust>=1.10" in PYPROJECT

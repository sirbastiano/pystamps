from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stage2_prior_state_drift_report.py"
_SPEC = importlib.util.spec_from_file_location("stage2_prior_state_drift_report", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def test_infer_prior_state_from_ph_weight_recovers_k_and_weighting() -> None:
    bperp = np.asarray(
        [
            [-30.0, -10.0, 5.0, 25.0],
            [-20.0, 0.0, 15.0, 35.0],
        ],
        dtype=np.float64,
    )
    ph_nm = np.asarray(
        [
            [1.0 + 0.0j, 0.5 + 0.5j, -0.3 + 0.7j, 0.2 - 0.9j],
            [0.8 + 0.1j, -0.1 + 0.9j, 0.4 - 0.6j, -0.7 - 0.2j],
        ],
        dtype=np.complex64,
    )
    ph_nm = ph_nm / np.abs(ph_nm)
    k_ps = np.asarray([0.0125, -0.008], dtype=np.float64)
    weighting = np.asarray([0.45, 1.25], dtype=np.float64)

    ph_weight = report.synthesize_ph_weight_from_prior(ph_nm, bperp, k_ps, weighting)
    inferred = report.infer_prior_state_from_ph_weight(ph_nm, ph_weight, bperp)

    np.testing.assert_allclose(inferred.k_ps, k_ps, rtol=0.0, atol=5e-10)
    np.testing.assert_allclose(inferred.weighting, weighting, rtol=0.0, atol=1e-7)


def test_classify_source_identifies_prior_k_and_combined_state() -> None:
    assert (
        report._classify_source(
            "ph_weight",
            {
                "current_K_current_weight": 1.0,
                "oracle_K_current_weight": 1e-8,
                "current_K_oracle_weight": 0.8,
                "oracle_K_oracle_weight": 0.0,
            },
        )
        == "prior_K_state"
    )
    assert (
        report._classify_source(
            "K_ps",
            {
                "current_K_current_weight": 1.0,
                "oracle_K_current_weight": 0.9,
                "current_K_oracle_weight": 0.8,
                "oracle_K_oracle_weight": 1e-12,
            },
        )
        == "combined_prior_K_and_weighting_state"
    )

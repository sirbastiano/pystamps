from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import sparse

from pystamps.config import ToleranceConfig
from pystamps.io.dataset import discover_dataset
from pystamps.io.mat import read_mat

DEFAULT_GLOBS: tuple[str, ...] = (
    "PATCH_*/ps1.mat",
    "PATCH_*/pm1.mat",
    "PATCH_*/select1.mat",
    "PATCH_*/weed1.mat",
    "ps2.mat",
    "pm2.mat",
    "ph2.mat",
    "phuw2.mat",
    "scla2.mat",
    "mean_v.mat",
    "ifgstd2.mat",
    "uw_space_time.mat",
    "uw_grid.mat",
    "uw_interp.mat",
)


@dataclass(slots=True)
class FileComparison:
    relative_path: str
    ok: bool
    message: str
    failure_kind: str | None = None
    failing_key: str | None = None
    shape_run: tuple[int, ...] | None = None
    shape_oracle: tuple[int, ...] | None = None
    max_abs: float | None = None
    matched_keys: int | None = None


@dataclass(slots=True)
class VerificationReport:
    comparisons: list[FileComparison] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.comparisons)

    @property
    def failures(self) -> list[FileComparison]:
        return [comparison for comparison in self.comparisons if not comparison.ok]


@dataclass(slots=True, frozen=True)
class FailureClassification:
    stage_scope: str
    failure_class: str
    label: str
    guidance: str


@dataclass(slots=True, frozen=True)
class ClassifiedFailure:
    relative_path: str
    message: str
    stage_scope: str
    failure_class: str
    label: str
    guidance: str
    failing_key: str | None
    failure_kind: str | None
    shape_run: tuple[int, ...] | None
    shape_oracle: tuple[int, ...] | None
    max_abs: float | None


_KEY_PATTERN = re.compile(r"key '([^']+)'")

_PATCH_STAGE2_CLASSIFICATION = FailureClassification(
    stage_scope="stage2",
    failure_class="stage2_patch_boundary",
    label="Stage 2 patch boundary",
    guidance="pm1.mat diverges before later patch stages; fix stage-2 parity before changing stage-3/4 or downstream code.",
)

_PATCH_STAGE3_CLASSIFICATION = FailureClassification(
    stage_scope="stage3",
    failure_class="stage3_patch_boundary",
    label="Stage 3 patch boundary",
    guidance="select1.mat diverges before later stages; fix stage-3 parity before changing stage-4 or downstream code.",
)

_PATCH_STAGE4_CLASSIFICATION = FailureClassification(
    stage_scope="stage4",
    failure_class="stage4_patch_boundary",
    label="Stage 4 patch boundary",
    guidance="weed1.mat diverges before merged stages; fix stage-4 parity before changing stage-5/6 or stage-7/8 code.",
)

_UNWRAP_SMOOTHING_CLASSIFICATION = FailureClassification(
    stage_scope="stage5_6",
    failure_class="unwrap_smoothing",
    label="Unwrap / smoothing",
    guidance="Merged unwrap inputs or unwrap products differ; isolate fixes to stage-5/6 merged and unwrap paths first.",
)

_UNWRAPPED_NOISE_STATS_CLASSIFICATION = FailureClassification(
    stage_scope="stage7_8",
    failure_class="unwrapped_noise_statistics",
    label="Unwrapped-noise / statistics",
    guidance="Failures are downstream of unwrapped products; keep fixes in stage-7/8 statistics and filtering unless coupling is traced upstream.",
)

_UNCLASSIFIED_FAILURE = FailureClassification(
    stage_scope="unknown",
    failure_class="unclassified",
    label="Unclassified",
    guidance="Artifact is outside the current downstream residual map; inspect the file path and producing stage directly.",
)

_UNWRAP_SMOOTHING_ARTIFACTS = {
    "pm2.mat",
    "ifgstd2.mat",
    "phuw2.mat",
    "uw_phaseuw.mat",
    "uw_grid.mat",
    "uw_interp.mat",
}

_UNWRAPPED_NOISE_STATS_ARTIFACTS = {
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "mv2.mat",
    "uw_space_time.mat",
}


def _is_numeric(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.names is None and value.dtype.kind in {"b", "i", "u", "f", "c"}
    return isinstance(value, (bool, int, float, complex, np.bool_, np.number))


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _collect_numeric(payload: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    if sparse.issparse(payload):
        payload_csc = payload.tocsc()
        out[f"{prefix}.data" if prefix else "data"] = np.asarray(payload_csc.data)
        out[f"{prefix}.ir" if prefix else "ir"] = np.asarray(payload_csc.indices)
        out[f"{prefix}.jc" if prefix else "jc"] = np.asarray(payload_csc.indptr)
        return out

    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_collect_numeric(value, next_prefix))
        return out

    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            next_prefix = f"{prefix}[{idx}]"
            out.update(_collect_numeric(value, next_prefix))
        return out

    if isinstance(payload, np.ndarray) and payload.dtype.names:
        # MATLAB v7.3 complex arrays are often represented as structured arrays
        # with fields like ('real', 'imag'). Compare fields independently.
        for field in payload.dtype.names:
            next_prefix = f"{prefix}.{field}" if prefix else field
            out.update(_collect_numeric(payload[field], next_prefix))
        return out

    if _is_numeric(payload):
        out[prefix] = _to_array(payload)

    return out


def _compare_mat(run_mat: Path, golden_mat: Path, tol: ToleranceConfig) -> tuple[bool, str, dict[str, Any]]:
    run_payload = read_mat(run_mat)
    golden_payload = read_mat(golden_mat)
    rtol = float(tol.rtol)
    atol = float(tol.atol)

    run_numeric = _collect_numeric(run_payload)
    golden_numeric = _collect_numeric(golden_payload)

    golden_keys = set(golden_numeric)
    run_keys = set(run_numeric)

    missing = sorted(golden_keys - run_keys)
    if missing:
        return False, f"Missing numeric keys in run: {', '.join(missing[:8])}", {"failure_kind": "missing_numeric_keys"}

    for key in sorted(golden_keys):
        lhs = run_numeric[key]
        rhs = golden_numeric[key]

        if lhs.shape != rhs.shape:
            return (
                False,
                f"Shape mismatch for key '{key}': {lhs.shape} != {rhs.shape}",
                {
                    "failure_kind": "shape_mismatch",
                    "failing_key": key,
                    "shape_run": tuple(int(v) for v in lhs.shape),
                    "shape_oracle": tuple(int(v) for v in rhs.shape),
                },
            )

        wrap_key = False
        if tol.wrap_equivalence:
            wrap_key = key in tol.wrap_keys or any(key.endswith(f".{suffix}") for suffix in tol.wrap_keys)

        if wrap_key:
            period = float(tol.wrap_period)
            if np.iscomplexobj(lhs) or np.iscomplexobj(rhs):
                lhs_c = np.asarray(lhs, dtype=np.complex128)
                rhs_c = np.asarray(rhs, dtype=np.complex128)
                diff = np.angle(lhs_c * np.conj(rhs_c))
                both_nan = np.isnan(np.real(lhs_c)) & np.isnan(np.real(rhs_c))
            else:
                lhs_f = np.asarray(lhs, dtype=np.float64)
                rhs_f = np.asarray(rhs, dtype=np.float64)
                diff = lhs_f - rhs_f
                diff = (diff + period / 2.0) % period - period / 2.0
                both_nan = np.isnan(lhs_f) & np.isnan(rhs_f)
            close = np.isclose(np.asarray(diff, dtype=np.float64), 0.0, rtol=rtol, atol=atol, equal_nan=False)
            ok_wrap = np.all(close | both_nan)
            if not ok_wrap:
                max_abs = float(np.nanmax(np.abs(np.asarray(diff, dtype=np.float64)[~both_nan])))
                return (
                    False,
                    f"Wrap mismatch for key '{key}', wrapped_max_abs={max_abs:.6g}",
                    {
                        "failure_kind": "wrap_mismatch",
                        "failing_key": key,
                        "shape_run": tuple(int(v) for v in lhs.shape),
                        "shape_oracle": tuple(int(v) for v in rhs.shape),
                        "max_abs": max_abs,
                    },
                )
            continue

        try:
            close = np.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True)
        except TypeError:
            close = np.array_equal(lhs, rhs, equal_nan=True)
        if not close:
            lhs_f = np.asarray(lhs, dtype=np.float64)
            rhs_f = np.asarray(rhs, dtype=np.float64)
            max_abs = float(np.nanmax(np.abs(lhs_f - rhs_f)))
            return (
                False,
                f"Value mismatch for key '{key}', max_abs={max_abs:.6g}",
                {
                    "failure_kind": "value_mismatch",
                    "failing_key": key,
                    "shape_run": tuple(int(v) for v in lhs.shape),
                    "shape_oracle": tuple(int(v) for v in rhs.shape),
                    "max_abs": max_abs,
                },
            )

    return True, f"Matched {len(golden_keys)} numeric keys", {"matched_keys": len(golden_keys)}


def _iter_pattern_files(root: Path, pattern: str) -> list[Path]:
    if not pattern.startswith("PATCH_*/"):
        return sorted(root.glob(pattern))

    layout = discover_dataset(root)
    subpattern = pattern.split("/", 1)[1]
    files: list[Path] = []
    for patch in layout.patches:
        files.extend(sorted(patch.glob(subpattern)))
    return files


def _extract_failure_key(message: str) -> str | None:
    match = _KEY_PATTERN.search(message)
    if match is not None:
        return match.group(1)
    return None


def classify_failure(relative_path: str) -> FailureClassification:
    path = Path(relative_path)
    basename = path.name
    if path.parts[:1] and path.parts[0].startswith("PATCH_"):
        if basename == "pm1.mat":
            return _PATCH_STAGE2_CLASSIFICATION
        if basename == "select1.mat":
            return _PATCH_STAGE3_CLASSIFICATION
        if basename == "weed1.mat":
            return _PATCH_STAGE4_CLASSIFICATION
    if basename in _UNWRAP_SMOOTHING_ARTIFACTS:
        return _UNWRAP_SMOOTHING_CLASSIFICATION
    if basename in _UNWRAPPED_NOISE_STATS_ARTIFACTS:
        return _UNWRAPPED_NOISE_STATS_CLASSIFICATION
    return _UNCLASSIFIED_FAILURE


def classify_failures(report: VerificationReport) -> list[ClassifiedFailure]:
    classified: list[ClassifiedFailure] = []
    for failure in report.failures:
        classification = classify_failure(failure.relative_path)
        failing_key = getattr(failure, "failing_key", None) or _extract_failure_key(failure.message)
        classified.append(
            ClassifiedFailure(
                relative_path=failure.relative_path,
                message=failure.message,
                stage_scope=classification.stage_scope,
                failure_class=classification.failure_class,
                label=classification.label,
                guidance=classification.guidance,
                failing_key=failing_key,
                failure_kind=getattr(failure, "failure_kind", None),
                shape_run=getattr(failure, "shape_run", None),
                shape_oracle=getattr(failure, "shape_oracle", None),
                max_abs=getattr(failure, "max_abs", None),
            )
        )
    return classified


def _stage_scope_priority(stage_scope: str) -> int:
    return {
        "stage2": 0,
        "stage3": 1,
        "stage4": 2,
        "stage3_4": 2,
        "stage5_6": 3,
        "stage7_8": 4,
        "unknown": 5,
    }.get(stage_scope, 6)


def _failure_priority(failure: ClassifiedFailure) -> tuple[int, str, str]:
    return (_stage_scope_priority(failure.stage_scope), failure.relative_path, failure.message)


def _shape_json(shape: tuple[int, ...] | None) -> list[int] | None:
    if shape is None:
        return None
    return [int(v) for v in shape]


def _failure_dict(failure: ClassifiedFailure) -> dict[str, Any]:
    return {
        "path": failure.relative_path,
        "message": failure.message,
        "stage_scope": failure.stage_scope,
        "failure_class": failure.failure_class,
        "label": failure.label,
        "failing_key": failure.failing_key,
        "failure_kind": failure.failure_kind,
        "shape_run": _shape_json(failure.shape_run),
        "shape_oracle": _shape_json(failure.shape_oracle),
        "max_abs": failure.max_abs,
        "guidance": failure.guidance,
    }


def summarize_failures(report: VerificationReport) -> dict[str, Any]:
    classified = classify_failures(report)
    groups: dict[str, dict[str, Any]] = {}
    for failure in classified:
        group = groups.setdefault(
            failure.failure_class,
            {
                "failure_class": failure.failure_class,
                "label": failure.label,
                "stage_scope": failure.stage_scope,
                "guidance": failure.guidance,
                "count": 0,
                "paths": [],
                "failing_keys": [],
            },
        )
        group["count"] += 1
        group["paths"].append(failure.relative_path)
        if failure.failing_key is not None and failure.failing_key not in group["failing_keys"]:
            group["failing_keys"].append(failure.failing_key)

    first_boundary_failure = min(classified, key=_failure_priority) if classified else None

    return {
        "ok": report.ok,
        "checked": len(report.comparisons),
        "failed": len(report.failures),
        "failures": [_failure_dict(failure) for failure in classified],
        "groups": sorted(groups.values(), key=lambda group: (_stage_scope_priority(group["stage_scope"]), group["failure_class"])),
        "first_boundary_failure": _failure_dict(first_boundary_failure) if first_boundary_failure is not None else None,
        "trace": {
            "stage2_residual_present": any(
                failure.failure_class == _PATCH_STAGE2_CLASSIFICATION.failure_class for failure in classified
            ),
            "stage3_4_residual_present": any(
                failure.failure_class in {
                    _PATCH_STAGE3_CLASSIFICATION.failure_class,
                    _PATCH_STAGE4_CLASSIFICATION.failure_class,
                }
                for failure in classified
            ),
            "stage3_4_coupling_evidence_present": False,
            "guidance": (
                "Do not modify downstream stages until the first stage-boundary trace is identified and its "
                "upstream artifact lineage is understood."
            ),
        },
    }


def verify_run_against_golden(
    run_root: str | Path,
    golden_root: str | Path,
    tolerance: ToleranceConfig,
    patterns: tuple[str, ...] = DEFAULT_GLOBS,
) -> VerificationReport:
    run_path = Path(run_root).resolve()
    golden_path = Path(golden_root).resolve()

    report = VerificationReport()

    golden_files: list[Path] = []
    for pattern in patterns:
        golden_files.extend(_iter_pattern_files(golden_path, pattern))

    if not golden_files:
        report.comparisons.append(
            FileComparison(
                relative_path="<dataset>",
                ok=False,
                message="No golden files found for selected patterns",
                failure_kind="missing_oracle_artifacts",
            )
        )
        return report

    for golden_file in golden_files:
        rel = golden_file.relative_to(golden_path)
        run_file = run_path / rel
        if not run_file.exists():
            report.comparisons.append(FileComparison(str(rel), False, "Missing run artifact", failure_kind="missing_run_artifact"))
            continue

        if golden_file.suffix.lower() == ".mat":
            ok, message, details = _compare_mat(run_file, golden_file, tolerance)
            report.comparisons.append(FileComparison(str(rel), ok, message, **details))
        else:
            if run_file.stat().st_size == golden_file.stat().st_size:
                report.comparisons.append(FileComparison(str(rel), True, "File size matches"))
            else:
                report.comparisons.append(FileComparison(str(rel), False, "File size differs", failure_kind="size_mismatch"))

    return report

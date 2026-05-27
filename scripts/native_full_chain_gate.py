from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATCH_PREFIX = "PATCH_"
REPORT_DIR_NAME = "_native_gate_reports"
DEFAULT_BUDGET_MANIFEST = Path(__file__).resolve().parents[1] / "pystamps" / "data" / "native_performance_budgets.json"
FORBIDDEN_NATIVE_ONLY_PROGRAMS = {"uv", "matlab", "octave"}
REQUIRED_COVERAGE_UNSUPPORTED_MODES = {"python", "matlab", "octave", "bridge"}

STAGE_CLEAN_PATTERNS: dict[int, tuple[str, ...]] = {
    1: (
        "PATCH_*/ps1.mat",
        "PATCH_*/ph1.mat",
        "PATCH_*/bp1.mat",
        "PATCH_*/psver.mat",
        "PATCH_*/da1.mat",
        "PATCH_*/hgt1.mat",
        "PATCH_*/la1.mat",
        "PATCH_*/inc1.mat",
    ),
    2: ("PATCH_*/pm1.mat",),
    3: ("PATCH_*/select1.mat",),
    4: ("PATCH_*/weed1.mat",),
    5: (
        "PATCH_*/ps2.mat",
        "PATCH_*/ph2.mat",
        "PATCH_*/pm2.mat",
        "PATCH_*/bp2.mat",
        "PATCH_*/hgt2.mat",
        "PATCH_*/la2.mat",
        "PATCH_*/rc2.mat",
        "PATCH_*/da2.mat",
        "PATCH_*/psver.mat",
        "ps2.mat",
        "ph2.mat",
        "pm2.mat",
        "bp2.mat",
        "hgt2.mat",
        "la2.mat",
        "rc2.mat",
        "psver.mat",
        "ifgstd2.mat",
    ),
    6: ("phuw2.mat", "uw_phaseuw.mat", "uw_grid.mat", "uw_interp.mat"),
    7: ("scla2.mat", "scla_smooth2.mat"),
    8: ("mean_v.mat", "mv2.mat", "uw_space_time.mat"),
}


class GateError(RuntimeError):
    """Raised for deterministic full-chain gate setup failures."""


@dataclass(frozen=True)
class PatchManifest:
    names: list[str]
    source: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _patch_sort_key(name: str) -> tuple[int, str]:
    suffix = name.replace(PATCH_PREFIX, "", 1)
    try:
        return (int(suffix), name)
    except ValueError:
        return (10**9, name)


def _read_patch_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _patch_dir_names(root: Path) -> list[str]:
    return sorted(
        [path.name for path in root.iterdir() if path.is_dir() and path.name.startswith(PATCH_PREFIX)],
        key=_patch_sort_key,
    )


def authoritative_patch_manifest(root: Path) -> PatchManifest:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise GateError(f"dataset root is not a directory: {root}")

    patch_dirs = _patch_dir_names(root)
    patch_list_old = _read_patch_manifest(root / "patch.list_old")
    if patch_list_old:
        _ensure_patch_dirs_exist(root, patch_list_old, "patch.list_old")
        return PatchManifest(patch_list_old, "patch.list_old")

    patch_list = _read_patch_manifest(root / "patch.list")
    if patch_list:
        if len(patch_dirs) > len(patch_list) and set(patch_list).issubset(patch_dirs):
            raise GateError(
                "patch.list lists "
                f"{len(patch_list)} patch(es) but {len(patch_dirs)} PATCH_* directories exist; "
                "restore the full patch.list or provide patch.list_old before running the gate"
            )
        _ensure_patch_dirs_exist(root, patch_list, "patch.list")
        return PatchManifest(patch_list, "patch.list")

    if not patch_dirs:
        raise GateError(f"dataset has no PATCH_* directories: {root}")
    return PatchManifest(patch_dirs, "PATCH_* directory scan")


def _ensure_patch_dirs_exist(root: Path, names: list[str], source: str) -> None:
    missing = [name for name in names if not (root / name).is_dir()]
    if missing:
        raise GateError(f"{source} references missing patch directories: {', '.join(missing)}")


def _copy_dataset(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, dest, copy_function=os.link)
    except OSError:
        shutil.copytree(source, dest)


def _safe_remove_tree(path: Path, dataset: Path) -> None:
    resolved = path.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    if resolved == dataset:
        raise GateError("RUN must not be the DATASET path")
    if dataset in resolved.parents:
        raise GateError("RUN must not be inside the DATASET path")
    if resolved in {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}:
        raise GateError(f"refusing to remove unsafe RUN path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def clean_patterns_for_range(start_step: int, end_step: int) -> list[str]:
    patterns: list[str] = []
    for stage in range(start_step, end_step + 1):
        for pattern in STAGE_CLEAN_PATTERNS.get(stage, ()):
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


def _clean_outputs(run_root: Path, patterns: list[str]) -> list[str]:
    removed: list[str] = []
    for pattern in patterns:
        for path in sorted(run_root.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(str(path.relative_to(run_root)))
    return removed


def _restore_patch_list(run_root: Path, manifest: PatchManifest) -> None:
    patch_list = run_root / "patch.list"
    if patch_list.exists():
        patch_list.unlink()
    patch_list.write_text("".join(f"{name}\n" for name in manifest.names), encoding="utf-8")

    allowed = set(manifest.names)
    for patch in run_root.iterdir():
        if patch.is_dir() and patch.name.startswith(PATCH_PREFIX) and patch.name not in allowed:
            shutil.rmtree(patch)


def prepare_run_copy(dataset: Path, run_root: Path, start_step: int, end_step: int) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    manifest = authoritative_patch_manifest(dataset)

    _safe_remove_tree(run_root, dataset)
    _copy_dataset(dataset, run_root)
    _restore_patch_list(run_root, manifest)

    patterns = clean_patterns_for_range(start_step, end_step)
    removed = _clean_outputs(run_root, patterns)
    return {
        "dataset": str(dataset),
        "run_root": str(run_root),
        "patch_manifest_source": manifest.source,
        "patches": manifest.names,
        "start_step": start_step,
        "end_step": end_step,
        "clean_patterns": patterns,
        "removed_artifacts": removed,
    }


def ensure_run_manifest_matches_golden(run_root: Path, golden_root: Path) -> None:
    expected = authoritative_patch_manifest(golden_root).names
    actual = _read_patch_manifest(run_root / "patch.list") or _patch_dir_names(run_root)
    if actual != expected:
        raise GateError(f"run patch manifest mismatch: run has {actual}, golden expects {expected}")
    _ensure_patch_dirs_exist(run_root, expected, "run patch.list")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _native_bin(args: argparse.Namespace) -> Path:
    native_bin = Path(args.native_bin).expanduser()
    if not native_bin.is_file():
        raise GateError(f"native binary does not exist: {native_bin}")
    return native_bin


def _native_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    native_bin = _native_bin(args)
    command = [
        str(native_bin),
        "run",
        "--native-only",
        "--dataset",
        str(run_root),
        "--start-step",
        str(args.start_step),
        "--end-step",
        str(args.end_step),
        "--backend",
        "native",
        "--stage2-kernel-backend",
        "native",
        "--cpu-workers",
        str(args.threads),
        "--stage2-native-threads",
        str(args.threads),
    ]
    return command


def _coverage_command(args: argparse.Namespace) -> list[str]:
    native_bin = _native_bin(args)
    return [
        str(native_bin),
        "coverage",
        "--start-step",
        str(args.start_step),
        "--end-step",
        str(args.end_step),
    ]


def validate_native_only_command(command: list[str]) -> None:
    if not command:
        raise GateError("native-only command is empty")
    validate_native_only_executable(command[0])
    if "--native-only" not in command:
        raise GateError("native command is missing --native-only")
    _require_flag_value(command, "--backend", "native")
    _require_flag_value(command, "--stage2-kernel-backend", "native")
    for token in command[1:]:
        name = Path(token).name.lower()
        if _is_forbidden_native_only_program(name):
            raise GateError(f"native-only mode forbids shelling out through {token}")


def validate_native_only_executable(program_path: str) -> None:
    program = Path(program_path).name.lower()
    if _is_forbidden_native_only_program(program, executable=True):
        raise GateError(f"native-only mode forbids bridge/external execution via {program_path}")


def _is_forbidden_native_only_program(name: str, *, executable: bool = False) -> bool:
    if name in FORBIDDEN_NATIVE_ONLY_PROGRAMS or name in {"python", "python3"}:
        return True
    if name.startswith("python3."):
        return True
    return executable and name.startswith("python")


def _require_flag_value(command: list[str], flag: str, expected: str) -> None:
    try:
        ix = command.index(flag)
    except ValueError as exc:
        raise GateError(f"native-only mode requires {flag} {expected}") from exc
    try:
        value = command[ix + 1]
    except IndexError as exc:
        raise GateError(f"native-only mode requires a value after {flag}") from exc
    if str(value).lower() != expected:
        raise GateError(f"native-only mode requires {flag} {expected}; got {value}")


def evaluate_native_coverage(rows: Any) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return {
            "ok": False,
            "checked_scope_count": 0,
            "violations": [{"kind": "invalid_coverage_payload", "message": "coverage payload is not a list"}],
        }

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(
                {
                    "kind": "invalid_coverage_row",
                    "index": index,
                    "message": "coverage row is not an object",
                }
            )
            continue
        stage = row.get("stage")
        scope = row.get("scope")
        target = row.get("target")
        row_id = f"stage {stage} {scope} {target}"
        if row.get("disabled") is True:
            violations.append(
                {
                    "kind": "disabled_stage",
                    "stage": stage,
                    "scope": scope,
                    "target": target,
                    "message": f"{row_id} is disabled: {row.get('disabled_reason') or 'no reason'}",
                }
            )
        if row.get("parity_certified") is not True:
            violations.append(
                {
                    "kind": "not_parity_certified",
                    "stage": stage,
                    "scope": scope,
                    "target": target,
                    "message": (
                        f"{row_id} is not parity-certified: "
                        f"{row.get('not_parity_certified_reason') or 'no reason'}"
                    ),
                }
            )
        if row.get("native_stage") is not True:
            violations.append(
                {
                    "kind": "not_native_stage",
                    "stage": stage,
                    "scope": scope,
                    "target": target,
                    "message": f"{row_id} is not native-certified: {row.get('not_native_reason') or 'no reason'}",
                }
            )
        unsupported_modes = row.get("unsupported_modes")
        if not isinstance(unsupported_modes, list):
            violations.append(
                {
                    "kind": "missing_unsupported_modes",
                    "stage": stage,
                    "scope": scope,
                    "target": target,
                    "message": f"{row_id} does not include unsupported native-only modes",
                }
            )
            continue
        mode_names = {
            str(item.get("mode", "")).lower()
            for item in unsupported_modes
            if isinstance(item, dict)
        }
        missing_modes = sorted(REQUIRED_COVERAGE_UNSUPPORTED_MODES - mode_names)
        if missing_modes:
            violations.append(
                {
                    "kind": "missing_unsupported_modes",
                    "stage": stage,
                    "scope": scope,
                    "target": target,
                    "missing_modes": missing_modes,
                    "message": f"{row_id} missing unsupported modes: {', '.join(missing_modes)}",
                }
            )
        for item in unsupported_modes:
            if not isinstance(item, dict):
                continue
            if str(item.get("mode", "")).lower() in REQUIRED_COVERAGE_UNSUPPORTED_MODES and not str(
                item.get("reason", "")
            ).strip():
                violations.append(
                    {
                        "kind": "missing_unsupported_mode_reason",
                        "stage": stage,
                        "scope": scope,
                        "target": target,
                        "mode": item.get("mode"),
                        "message": f"{row_id} unsupported mode {item.get('mode')} has no reason",
                    }
                )

    return {
        "ok": not violations,
        "checked_scope_count": len(rows),
        "violations": violations,
    }


def run_native_coverage_gate(args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:
    command = _coverage_command(args)
    validate_native_only_executable(command[0])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        coverage = json.loads(completed.stdout) if completed.stdout.strip() else []
    except json.JSONDecodeError:
        coverage = []
    evaluation = evaluate_native_coverage(coverage)
    ok = completed.returncode == 0 and bool(evaluation["ok"])
    report = {
        "generated_at_utc": _now_utc(),
        "ok": ok,
        "status": "passed" if ok else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "coverage": coverage,
        "evaluation": evaluation,
    }
    report_path = report_dir / "native-coverage-report.json"
    _write_json(report_path, report)
    print(f"Native coverage status: {'ok' if ok else 'failed'}")
    for violation in evaluation.get("violations", []):
        print(f"  coverage violation: {violation.get('message')}")
    print(f"Native coverage report: {report_path}")
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return report


def _stage_status(result: dict[str, Any]) -> str:
    return str(result.get("status", "")).lower()


def _stage_durations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "stage": result.get("stage"),
            "scope": result.get("scope"),
            "target": result.get("target"),
            "status": result.get("status"),
            "duration_sec": result.get("duration_sec"),
        }
        for key in (
            "input_artifact_count",
            "output_artifact_count",
            "rows_processed",
            "memory_peak_bytes",
            "n_grid_ps",
            "n_grid_rows",
            "n_grid_cols",
            "n_edges",
        ):
            if key in result:
                row[key] = result.get(key)
        rows.append(row)
    return rows


def _print_stage_durations(rows: list[dict[str, Any]]) -> None:
    print("Native stage durations:")
    for row in rows:
        duration = row.get("duration_sec")
        duration_text = "n/a" if duration is None else f"{float(duration):.3f}s"
        print(
            "  "
            f"stage {row.get('stage')} {row.get('scope')} {row.get('target')}: "
            f"{duration_text} {row.get('status')}"
        )


def load_performance_budget_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"performance budget manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"performance budget manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"performance budget manifest must be a JSON object: {path}")
    payload = dict(payload)
    payload["manifest_path"] = str(path)
    return payload


def evaluate_performance_budgets(
    manifest: dict[str, Any],
    elapsed_sec: float,
    stage_rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    violations: list[dict[str, Any]] = []
    waivers: list[dict[str, Any]] = []

    release = manifest.get("release", {})
    if isinstance(release, dict):
        max_total = _number_or_none(release.get("max_total_duration_sec"))
        if max_total is not None and elapsed_sec > max_total:
            _record_budget_result(
                violations,
                waivers,
                release,
                now,
                {
                    "kind": "release_runtime",
                    "scope": "run",
                    "observed": elapsed_sec,
                    "ceiling": max_total,
                    "message": f"release runtime {elapsed_sec:.3f}s exceeds {max_total:.3f}s",
                },
            )

    for row in stage_rows:
        budget = _matching_stage_budget(manifest, row)
        if budget is None:
            continue
        duration = _number_or_none(row.get("duration_sec"))
        max_duration = _number_or_none(budget.get("max_duration_sec"))
        if duration is not None and max_duration is not None and duration > max_duration:
            _record_budget_result(
                violations,
                waivers,
                budget,
                now,
                {
                    "kind": "stage_duration",
                    "stage": row.get("stage"),
                    "scope": row.get("scope"),
                    "target": row.get("target"),
                    "observed": duration,
                    "ceiling": max_duration,
                    "message": (
                        f"stage {row.get('stage')} {row.get('scope')} {row.get('target')} "
                        f"duration {duration:.3f}s exceeds {max_duration:.3f}s"
                    ),
                },
            )
        memory_peak = _number_or_none(row.get("memory_peak_bytes"))
        max_memory = _number_or_none(budget.get("max_peak_rss_bytes"))
        if memory_peak is not None and max_memory is not None and memory_peak > max_memory:
            _record_budget_result(
                violations,
                waivers,
                budget,
                now,
                {
                    "kind": "stage_memory",
                    "stage": row.get("stage"),
                    "scope": row.get("scope"),
                    "target": row.get("target"),
                    "observed": memory_peak,
                    "ceiling": max_memory,
                    "message": (
                        f"stage {row.get('stage')} {row.get('scope')} {row.get('target')} "
                        f"peak RSS {int(memory_peak)} bytes exceeds {int(max_memory)} bytes"
                    ),
                },
            )

    return {
        "ok": not violations,
        "manifest_path": manifest.get("manifest_path"),
        "violations": violations,
        "waivers": waivers,
        "checked_stage_count": len(stage_rows),
    }


def _record_budget_result(
    violations: list[dict[str, Any]],
    waivers: list[dict[str, Any]],
    budget: dict[str, Any],
    now: datetime,
    item: dict[str, Any],
) -> None:
    waiver = budget.get("temporary_waiver")
    if _documented_temporary_waiver(waiver, now):
        waived = dict(item)
        waived["waiver"] = waiver
        waivers.append(waived)
    else:
        violations.append(item)


def _documented_temporary_waiver(value: Any, now: datetime) -> bool:
    if not isinstance(value, dict):
        return False
    reason = str(value.get("reason", "")).strip()
    owner = str(value.get("owner", "")).strip()
    expires_raw = str(value.get("expires_at_utc", "")).strip()
    if not reason or not owner or not expires_raw:
        return False
    expires = _parse_utc_datetime(expires_raw)
    return expires is not None and expires > now


def _parse_utc_datetime(value: str) -> datetime | None:
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _matching_stage_budget(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    stages = manifest.get("stages", [])
    if not isinstance(stages, list):
        return None
    for budget in stages:
        if not isinstance(budget, dict):
            continue
        if int(budget.get("stage", -1)) != int(row.get("stage", -2)):
            continue
        if str(budget.get("scope", "")).lower() != str(row.get("scope", "")).lower():
            continue
        if _target_matches(str(budget.get("target", "*")), str(row.get("target", ""))):
            return budget
    return None


def _target_matches(pattern: str, target: str) -> bool:
    if pattern in {"*", ""}:
        return True
    if pattern.endswith("*"):
        return target.startswith(pattern[:-1])
    return pattern == target


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_budget_report(report: dict[str, Any]) -> None:
    print(f"Performance budget status: {'ok' if report.get('ok') else 'failed'}")
    for violation in report.get("violations", []):
        print(f"  budget violation: {violation.get('message')}")
    for waiver in report.get("waivers", []):
        waiver_info = waiver.get("waiver", {})
        print(
            "  budget waiver: "
            f"{waiver.get('message')} "
            f"(owner={waiver_info.get('owner')}, expires={waiver_info.get('expires_at_utc')})"
        )


def run_native_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    start = time.monotonic()
    dataset = Path(args.dataset)
    run_root = Path(args.run)
    setup = prepare_run_copy(dataset, run_root, args.start_step, args.end_step)
    run_root = Path(setup["run_root"])
    report_dir = run_root / REPORT_DIR_NAME
    coverage_report = run_native_coverage_gate(args, report_dir)
    command = _native_command(args, run_root)
    validate_native_only_command(command)

    if not coverage_report.get("ok"):
        elapsed = time.monotonic() - start
        run_report = {
            "generated_at_utc": _now_utc(),
            "ok": False,
            "status": "failed",
            "elapsed_sec": elapsed,
            "setup": setup,
            "coverage": coverage_report,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "native coverage gate failed",
            "results": [],
            "performance_budget": {
                "ok": False,
                "violations": [
                    {
                        "kind": "native_coverage",
                        "message": "native coverage gate failed before stage execution",
                    }
                ],
                "waivers": [],
                "checked_stage_count": 0,
            },
        }
        run_report_path = report_dir / "native-run-report.json"
        _write_json(run_report_path, run_report)
        print("Native run status: failed")
        print(f"Native run report: {run_report_path}")
        return 1, run_report

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    try:
        results = json.loads(completed.stdout) if completed.stdout.strip() else []
    except json.JSONDecodeError:
        results = []

    stage_failed = any(_stage_status(result) == "failed" for result in results if isinstance(result, dict))
    skipped_existing = any(_stage_status(result) == "skipped_existing" for result in results if isinstance(result, dict))
    duration_rows = _stage_durations([result for result in results if isinstance(result, dict)])
    budget_manifest = load_performance_budget_manifest(Path(args.budget_manifest).expanduser())
    budget_report = evaluate_performance_budgets(budget_manifest, elapsed, duration_rows)

    native_ok = completed.returncode == 0 and not stage_failed and not skipped_existing
    ok = native_ok and bool(budget_report["ok"])
    run_report = {
        "generated_at_utc": _now_utc(),
        "ok": ok,
        "status": "passed" if ok else "failed",
        "elapsed_sec": elapsed,
        "setup": setup,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "coverage": coverage_report,
        "results": results,
        "performance_budget": budget_report,
    }
    timing_report = {
        "generated_at_utc": run_report["generated_at_utc"],
        "run_root": str(run_root),
        "elapsed_sec": elapsed,
        "stages": duration_rows,
        "performance_budget": budget_report,
    }
    run_report_path = report_dir / "native-run-report.json"
    timing_report_path = report_dir / "native-run-timings.json"
    _write_json(run_report_path, run_report)
    _write_json(timing_report_path, timing_report)

    _print_stage_durations(duration_rows)
    _print_budget_report(budget_report)
    print(f"Native run status: {'ok' if ok else 'failed'}")
    print(f"Native run report: {run_report_path}")
    print(f"Native timing report: {timing_report_path}")
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return (0 if ok else 1), run_report


def _verify_command(run_root: Path, golden_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pystamps.cli",
        "verify",
        "--run",
        str(run_root),
        "--golden",
        str(golden_root),
    ]


def run_verify_gate(args: argparse.Namespace) -> int:
    run_exit, run_report = run_native_gate(args)
    run_root = Path(run_report["setup"]["run_root"])
    golden_root = Path(args.golden).expanduser().resolve()
    report_dir = run_root / REPORT_DIR_NAME
    if run_exit != 0:
        return run_exit

    ensure_run_manifest_matches_golden(run_root, golden_root)
    command = _verify_command(run_root, golden_root)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        verify_payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    except json.JSONDecodeError:
        verify_payload = {}
    verifier_ok = bool(verify_payload.get("ok")) and completed.returncode == 0
    report = {
        "generated_at_utc": _now_utc(),
        "ok": verifier_ok,
        "status": "passed" if verifier_ok else "failed",
        "run_root": str(run_root),
        "golden_root": str(golden_root),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "verifier": verify_payload,
    }
    verify_report_path = report_dir / "native-verify-report.json"
    _write_json(verify_report_path, report)

    print(
        "Parity status: "
        f"{'ok' if verifier_ok else 'failed'} "
        f"(checked={verify_payload.get('checked', 'n/a')}, "
        f"failed={len(verify_payload.get('failed', [])) if isinstance(verify_payload.get('failed'), list) else 'n/a'})"
    )
    print(f"Parity report: {verify_report_path}")
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return 0 if verifier_ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repeatable native full-chain parity gate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--run", required=True)
        subparser.add_argument("--native-bin", required=True)
        subparser.add_argument("--threads", type=int, default=0)
        subparser.add_argument("--start-step", type=int, default=1)
        subparser.add_argument("--end-step", type=int, default=8)
        subparser.add_argument("--budget-manifest", default=str(DEFAULT_BUDGET_MANIFEST))

    run_parser = subparsers.add_parser("run", help="Create a clean run copy and execute native stages")
    add_common(run_parser)

    verify_parser = subparsers.add_parser("verify", help="Run native stages and verify parity")
    add_common(verify_parser)
    verify_parser.add_argument("--golden", required=True)

    args = parser.parse_args(argv)
    if args.start_step < 1 or args.end_step > 8 or args.start_step > args.end_step:
        raise GateError(f"invalid stage range {args.start_step}..{args.end_step}; expected 1..8")
    if args.threads < 0:
        raise GateError("--threads must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv or sys.argv[1:])
        if args.command == "run":
            exit_code, _ = run_native_gate(args)
            return exit_code
        if args.command == "verify":
            return run_verify_gate(args)
        raise GateError(f"unknown command: {args.command}")
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

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


def _native_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    native_bin = Path(args.native_bin).expanduser()
    if not native_bin.is_file():
        raise GateError(f"native binary does not exist: {native_bin}")
    command = [
        str(native_bin),
        "run",
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


def _stage_status(result: dict[str, Any]) -> str:
    return str(result.get("status", "")).lower()


def _stage_durations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "stage": result.get("stage"),
                "scope": result.get("scope"),
                "target": result.get("target"),
                "status": result.get("status"),
                "duration_sec": result.get("duration_sec"),
            }
        )
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


def run_native_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    start = time.monotonic()
    dataset = Path(args.dataset)
    run_root = Path(args.run)
    setup = prepare_run_copy(dataset, run_root, args.start_step, args.end_step)
    run_root = Path(setup["run_root"])
    report_dir = run_root / REPORT_DIR_NAME
    command = _native_command(args, run_root)

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    try:
        results = json.loads(completed.stdout) if completed.stdout.strip() else []
    except json.JSONDecodeError:
        results = []

    stage_failed = any(_stage_status(result) == "failed" for result in results if isinstance(result, dict))
    skipped_existing = any(_stage_status(result) == "skipped_existing" for result in results if isinstance(result, dict))
    ok = completed.returncode == 0 and not stage_failed and not skipped_existing
    duration_rows = _stage_durations([result for result in results if isinstance(result, dict)])

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
        "results": results,
    }
    timing_report = {
        "generated_at_utc": run_report["generated_at_utc"],
        "run_root": str(run_root),
        "elapsed_sec": elapsed,
        "stages": duration_rows,
    }
    run_report_path = report_dir / "native-run-report.json"
    timing_report_path = report_dir / "native-run-timings.json"
    _write_json(run_report_path, run_report)
    _write_json(timing_report_path, timing_report)

    _print_stage_durations(duration_rows)
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

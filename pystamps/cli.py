from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pystamps.compat.legacy import discover_legacy_commands
from pystamps.config import ConfigError, RunConfig, load_config
from pystamps.input_contracts import describe_stage_inputs, parse_stage_spec
from pystamps.kernels import describe_backend_matrix
from pystamps.notebooks.dataset_inspection import inspect_stage1_inputs
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext
from pystamps.status import collect_status
from pystamps.verify import verify_run_against_golden


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pystamps", description="Python-first StaMPS migration runtime")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run pipeline stages")
    run_parser.add_argument("--dataset", type=str, required=True)
    run_parser.add_argument("--start-step", type=int, default=1)
    run_parser.add_argument("--end-step", type=int, default=8)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--io-workers", type=int, default=None)
    run_parser.add_argument("--cpu-workers", type=int, default=None)

    status_parser = subparsers.add_parser("status", help="Inspect stage progress in dataset")
    status_parser.add_argument("--dataset", type=str, required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify run outputs against golden outputs")
    verify_parser.add_argument("--run", type=str, required=True)
    verify_parser.add_argument("--golden", type=str, required=True)

    legacy_parser = subparsers.add_parser("list-legacy", help="List discoverable legacy scripts")
    legacy_parser.add_argument(
        "--stamps-root",
        type=str,
        default=None,
        help="Explicit StaMPS checkout root. Defaults to $STAMPS_ROOT when set.",
    )

    describe_parser = subparsers.add_parser(
        "describe-inputs",
        help="Describe the logical inputs required by one stage or all stages",
    )
    describe_parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage number, comma-separated stage numbers, or 'all'",
    )
    describe_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset root for a real Stage-1 input check",
    )
    describe_parser.add_argument(
        "--patch",
        type=str,
        default="PATCH_1",
        help="Patch name used with --dataset for Stage-1 checks",
    )
    subparsers.add_parser(
        "describe-backends",
        help="Describe registered kernel backends and current backend coverage",
    )

    return parser.parse_args()


def _load_run_config(path: str | None) -> RunConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from exc


def _cmd_status(dataset: str) -> int:
    status = collect_status(dataset)
    payload = {
        "dataset": str(status.dataset),
        "merged_stage": status.merged_stage,
        "patches": [{"patch": p.patch, "stage": p.stage} for p in status.patch_statuses],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace, run_config: RunConfig) -> int:
    if args.io_workers is not None:
        run_config.runtime.io_workers = args.io_workers
    if args.cpu_workers is not None:
        run_config.runtime.cpu_workers = args.cpu_workers

    context = PipelineContext(
        dataset_root=Path(args.dataset).resolve(),
        run_config=run_config,
        start_step=args.start_step,
        end_step=args.end_step,
        dry_run=args.dry_run,
    )

    report = run_pipeline(context)
    payload = [
        {
            "stage": r.stage_id,
            "scope": r.scope,
            "target": r.target,
            "status": r.status,
            "details": r.details,
            "duration_sec": r.duration_sec,
        }
        for r in report.results
    ]
    print(json.dumps(payload, indent=2))

    return 1 if report.failures else 0


def _cmd_verify(run: str, golden: str, run_config: RunConfig) -> int:
    report = verify_run_against_golden(run, golden, run_config.tolerance)
    payload = {
        "ok": report.ok,
        "checked": len(report.comparisons),
        "failed": [
            {"path": c.relative_path, "message": c.message} for c in report.comparisons if not c.ok
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.ok else 1


def _resolve_stamps_root(stamps_root: str | None) -> str:
    if stamps_root:
        return stamps_root
    env_root = os.environ.get("STAMPS_ROOT")
    if env_root:
        return env_root
    raise SystemExit("Config error: list-legacy requires --stamps-root or STAMPS_ROOT")


def _cmd_list_legacy(stamps_root: str | None) -> int:
    commands = discover_legacy_commands(_resolve_stamps_root(stamps_root))
    print(json.dumps([str(path) for path in commands], indent=2))
    return 0


def _cmd_describe_inputs(stage: str, dataset: str | None, patch: str) -> int:
    try:
        stages = parse_stage_spec(stage)
    except ValueError as exc:
        raise SystemExit(f"Config error: {exc}") from exc

    payload: dict[str, object] = {
        "stages": describe_stage_inputs(stage),
    }
    if dataset is not None and 1 in stages:
        summary = inspect_stage1_inputs(dataset, patch_name=patch)
        payload["stage1_dataset_check"] = {
            "dataset": Path(dataset).name or str(dataset),
            "patch": patch,
            "metadata_mode": summary["metadata_mode"],
            "overview": summary["overview_rows"],
            "consistency": summary["consistency_rows"],
            "warnings": summary["warnings"],
        }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_describe_backends() -> int:
    print(json.dumps(describe_backend_matrix(), indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    run_config = _load_run_config(args.config)

    if args.command == "status":
        return _cmd_status(args.dataset)
    if args.command == "run":
        return _cmd_run(args, run_config)
    if args.command == "verify":
        return _cmd_verify(args.run, args.golden, run_config)
    if args.command == "list-legacy":
        return _cmd_list_legacy(args.stamps_root)
    if args.command == "describe-inputs":
        return _cmd_describe_inputs(args.stage, args.dataset, args.patch)
    if args.command == "describe-backends":
        return _cmd_describe_backends()

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

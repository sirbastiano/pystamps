#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pystamps.config import RunConfig
from pystamps.parity_contract import build_parity_contract
from pystamps.verify import summarize_failures, verify_run_against_golden


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supported parity audit across the required local datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset roots to audit. Defaults to the required contract datasets when omitted.",
    )
    parser.add_argument(
        "--golden-root",
        default=None,
        help="Optional base directory containing golden datasets with matching leaf names",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inputs_root() -> Path:
    return _repo_root() / "inputs_and_outputs"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_contract() -> dict[str, Any]:
    return build_parity_contract(_inputs_root())


def _resolve_datasets(args: argparse.Namespace, contract: dict[str, Any]) -> tuple[list[Path], list[str], list[str]]:
    repo_root = _repo_root()
    required_relative = [str(path) for path in contract["required_dataset_paths"]]
    required_paths = [repo_root / relative for relative in required_relative]
    requested = [Path(value).expanduser().resolve() for value in args.datasets] if args.datasets else required_paths

    requested_relative = []
    for dataset in requested:
        try:
            requested_relative.append(str(dataset.relative_to(repo_root)).replace("\\", "/"))
        except ValueError:
            requested_relative.append(str(dataset))

    missing_from_request = [path for path in required_relative if path not in requested_relative]
    extra_request = [path for path in requested_relative if path not in required_relative]
    return requested, missing_from_request, extra_request


def _workflow_name(dataset_root: Path) -> str:
    return f"{dataset_root.name}_audit"


def _validation_runs_root() -> Path:
    return _inputs_root() / "validation_runs"


def _latest_workflow_run(dataset_name: str, suffixes: tuple[str, ...]) -> Path | None:
    validation_runs = _validation_runs_root()
    if not validation_runs.exists():
        return None

    matches: list[tuple[str, int, Path]] = []
    for child in validation_runs.iterdir():
        if not child.is_dir():
            continue
        for priority, suffix in enumerate(suffixes):
            candidate = child / f"{dataset_name}_{suffix}"
            if candidate.exists():
                matches.append((child.name, priority, candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], -item[1]))
    return matches[-1][2].resolve()


def _resolve_required_run_root(dataset_root: Path) -> Path | None:
    dataset_name = dataset_root.name
    repo_root = _repo_root()

    explicit_run_roots = {
        "InSAR_dataset_test": repo_root / "inputs_and_outputs" / "RUN_FULL_GATE_1e10",
    }
    explicit = explicit_run_roots.get(dataset_name)
    if explicit is not None and explicit.exists():
        return explicit.resolve()

    return _latest_workflow_run(dataset_name, ("stage1_8", "stage2_8"))


def _resolve_run_selection(dataset_root: Path, golden_base: Path | None) -> tuple[Path, Path, str]:
    if golden_base is not None:
        return dataset_root.resolve(), (golden_base / dataset_root.name).resolve(), "explicit_run_root"

    run_root = _resolve_required_run_root(dataset_root)
    if run_root is None:
        raise FileNotFoundError(
            "No concrete full-loop run copy found for "
            f"{dataset_root.name}. Expected a repo-local run root such as "
            "inputs_and_outputs/RUN_FULL_GATE_1e10 or a validation_runs/*/<dataset>_stage1_8 copy."
        )

    return run_root, dataset_root.resolve(), "resolved_full_loop_run_copy"


def _dataset_audit(run_root: Path, golden_root: Path, run_source: str) -> dict[str, Any]:
    report = verify_run_against_golden(run_root, golden_root, RunConfig().tolerance)
    summary = summarize_failures(report)
    return {
        "workflow": _workflow_name(golden_root),
        "dataset": golden_root.name,
        "run_root": str(run_root),
        "golden_root": str(golden_root),
        "run_source": run_source,
        "ok": report.ok,
        "status": "passed" if report.ok else "failed",
        "checked": len(report.comparisons),
        **summary,
    }


def _base_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": _now_utc(),
        "contract": contract,
        "missing_datasets": [],
        "audits": [],
        "failed_workflows": [],
        "completed": False,
        "interrupted": False,
        "interruption": None,
        "ok": False,
    }


def _finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["failed_workflows"] = []
    if payload["missing_datasets"] or any(not audit["ok"] for audit in payload["audits"]) or payload["interrupted"]:
        payload["failed_workflows"] = ["full_validation"]
    payload["ok"] = (
        not payload["missing_datasets"]
        and not payload["failed_workflows"]
        and payload["completed"]
        and not payload["interrupted"]
    )
    return payload


def _emit_payload(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    args = _parse_args()
    contract = _resolve_contract()
    payload = _base_payload(contract)
    datasets, missing_from_request, extra_request = _resolve_datasets(args, contract)
    golden_base = Path(args.golden_root).expanduser().resolve() if args.golden_root else None

    missing = [str(dataset) for dataset in datasets if not dataset.exists()]
    if missing_from_request or extra_request:
        if missing_from_request:
            payload["missing_datasets"].extend(missing_from_request)
        if extra_request:
            payload["interruption"] = {
                "kind": "unsupported_dataset_selection",
                "message": "The supported audit only accepts the contract-required datasets.",
                "extra_datasets": extra_request,
            }
        _emit_payload(_finalize_payload(payload), args.output)
        return 1

    if missing:
        payload["missing_datasets"] = missing
        payload["interruption"] = {
            "kind": "missing_dataset",
            "message": "One or more required datasets are missing; audit aborted before verification.",
        }
        _emit_payload(_finalize_payload(payload), args.output)
        return 1

    try:
        for dataset in datasets:
            run_root, golden_root, run_source = _resolve_run_selection(dataset, golden_base)
            payload["audits"].append(_dataset_audit(run_root, golden_root, run_source))
        payload["completed"] = True
    except FileNotFoundError as exc:
        payload["interrupted"] = True
        payload["interruption"] = {
            "kind": "missing_run_copy",
            "message": str(exc),
        }
        _emit_payload(_finalize_payload(payload), args.output)
        return 1
    except KeyboardInterrupt:
        payload["interrupted"] = True
        payload["interruption"] = {
            "kind": "keyboard_interrupt",
            "message": "Audit interrupted before all dataset workflows completed.",
        }
        _emit_payload(_finalize_payload(payload), args.output)
        return 1
    except Exception as exc:
        payload["interrupted"] = True
        payload["interruption"] = {
            "kind": "exception",
            "message": str(exc),
        }
        _emit_payload(_finalize_payload(payload), args.output)
        return 1

    _emit_payload(_finalize_payload(payload), args.output)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

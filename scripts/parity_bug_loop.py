#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pystamps.config import RunConfig
from pystamps.parity_contract import DEFAULT_REQUIRED_DATASETS, build_parity_contract, capture_code_state
from pystamps.verify import summarize_failures, verify_run_against_golden


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one parity iteration: audit, divergence-set compare, and next-target summary."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to audit. Defaults to the required audited dataset set.",
    )
    parser.add_argument("--golden-root", default=None, help="Optional golden root override passed through to validate_audit")
    parser.add_argument(
        "--output",
        default="inputs_and_outputs/validation_runs/latest_parity_loop.json",
        help="Loop summary JSON output path.",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional validate_audit JSON output path. Defaults to a stamped validation_runs artifact.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Pass through to validate_audit when only a subset dataset is available locally.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validation_runs_root() -> Path:
    return _repo_root() / "inputs_and_outputs" / "validation_runs"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _default_audit_output() -> Path:
    return _validation_runs_root() / f"{_now_stamp()}_parity_bug_loop_audit.json"


def _resolved_datasets(args: argparse.Namespace) -> list[str]:
    return list(args.datasets) if args.datasets else list(DEFAULT_REQUIRED_DATASETS)


def _dataset_names(datasets: list[str]) -> list[str]:
    return [Path(dataset).name for dataset in datasets]


def _run_validate_audit(args: argparse.Namespace, audit_output: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/validate_audit.py",
        "--datasets",
        *_resolved_datasets(args),
        "--output",
        str(audit_output),
    ]
    if args.allow_subset:
        cmd.append("--allow-subset")
    if args.golden_root:
        cmd.extend(["--golden-root", args.golden_root])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_code_state(code_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(code_state, dict):
        return None
    normalized = dict(code_state)
    normalized["git_status"] = [str(line).lstrip() for line in code_state.get("git_status", [])]
    return normalized


def _matching_audit_payload(audit_output: Path, datasets: list[str]) -> dict[str, Any] | None:
    if not audit_output.exists():
        return None

    audit_payload = _load_json(audit_output)
    current_code_state = capture_code_state(_repo_root())
    if _normalized_code_state(audit_payload.get("code_state")) != _normalized_code_state(current_code_state):
        return None

    expected_names = _dataset_names(datasets)
    audit_names = [str(audit.get("dataset")) for audit in audit_payload.get("audits", []) if audit.get("dataset")]
    if audit_names:
        return audit_payload if audit_names == expected_names else None

    contract = audit_payload.get("contract")
    if isinstance(contract, dict):
        required_paths = contract.get("required_dataset_paths") or []
        if _dataset_names(list(required_paths)) == expected_names:
            return audit_payload
    return None


def _build_divergence_sets(audit: dict[str, Any]) -> list[dict[str, Any]]:
    divergence_sets: list[dict[str, Any]] = []
    failures = audit.get("failures", [])
    groups = audit.get("groups", [])

    all_patterns = _ordered_unique([failure["path"] for failure in failures if failure.get("path")])
    if all_patterns:
        divergence_sets.append(
            {
                "name": "all",
                "label": "All current failures",
                "stage_scope": "mixed",
                "failure_class": "all",
                "guidance": "Compare the exact current failure set after each code change.",
                "patterns": all_patterns,
            }
        )

    for group in groups:
        patterns = _ordered_unique([path for path in group.get("paths", []) if path])
        if not patterns:
            continue
        divergence_sets.append(
            {
                "name": str(group["failure_class"]),
                "label": str(group["label"]),
                "stage_scope": str(group["stage_scope"]),
                "failure_class": str(group["failure_class"]),
                "guidance": str(group["guidance"]),
                "patterns": patterns,
            }
        )
    return divergence_sets


def _compare_patterns(run_root: Path, golden_root: Path, patterns: list[str]) -> dict[str, Any]:
    report = verify_run_against_golden(run_root, golden_root, RunConfig().tolerance, patterns=tuple(patterns))
    summary = summarize_failures(report)
    failures = summary["failures"]
    return {
        "ok": summary["ok"],
        "checked": summary["checked"],
        "failed": summary["failed"],
        "patterns": patterns,
        "first_failure": failures[0] if failures else None,
        "first_boundary_failure": summary.get("first_boundary_failure"),
        "groups": summary["groups"],
        "failures": failures,
        "trace": summary["trace"],
    }


def _target_priority(item: dict[str, Any]) -> tuple[int, int, str]:
    stage_scope = item.get("stage_scope")
    failure_class = item.get("failure_class")
    failed = int(item.get("failed", 0))
    scope_priority = {
        "stage2": 0,
        "stage3": 1,
        "stage4": 2,
        "stage3_4": 2,
        "stage5_6": 3,
        "stage7_8": 4,
        "unknown": 5,
        "mixed": 6,
    }.get(stage_scope, 5)
    class_priority = 1 if failure_class == "all" else 0
    return (class_priority, scope_priority, -failed)


def _contract_metadata(audit_payload: dict[str, Any]) -> dict[str, Any]:
    contract = audit_payload.get("contract")
    if not isinstance(contract, dict):
        contract = build_parity_contract(_repo_root() / "inputs_and_outputs")
    return {
        "contract_version": contract.get("contract_version"),
        "oracle_contract_manifest_path": contract.get("oracle_contract_manifest_path"),
        "audited_workflow_manifest_path": contract.get("audited_workflow_manifest_path"),
        "required_dataset_paths": contract.get("required_dataset_paths"),
    }


def _trace_next_target(audit: dict[str, Any]) -> dict[str, Any] | None:
    trace = audit.get("trace") or {}
    first_boundary = trace.get("first_divergent_boundary")
    if not isinstance(first_boundary, dict) or not first_boundary.get("artifact_path"):
        return None
    artifact_path = str(first_boundary["artifact_path"])
    return {
        "name": str(first_boundary.get("failure_class") or "first_divergent_boundary"),
        "label": str(first_boundary.get("label") or "First divergent boundary"),
        "stage_scope": str(first_boundary.get("stage_scope") or "unknown"),
        "failure_class": str(first_boundary.get("failure_class") or "first_divergent_boundary"),
        "guidance": str(first_boundary.get("guidance") or ""),
        "patterns": [artifact_path],
        "ok": False,
        "checked": 1,
        "failed": 1,
        "first_failure": {
            "path": artifact_path,
            "message": first_boundary.get("message"),
            "stage_scope": first_boundary.get("stage_scope"),
            "failure_class": first_boundary.get("failure_class"),
            "label": first_boundary.get("label"),
            "failing_key": first_boundary.get("failing_key"),
            "failure_kind": first_boundary.get("failure_kind"),
            "shape_run": first_boundary.get("shape_run"),
            "shape_oracle": first_boundary.get("shape_oracle"),
            "max_abs": first_boundary.get("max_abs"),
            "guidance": first_boundary.get("guidance"),
        },
        "groups": [],
        "failures": [],
        "trace": trace,
        "source": "stage_boundary_trace",
    }


def _evaluate_audit(audit: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(audit["run_root"]).resolve()
    golden_root = Path(audit["golden_root"]).resolve()

    divergence_set_results: list[dict[str, Any]] = []
    for divergence_set in _build_divergence_sets(audit):
        result = _compare_patterns(run_root, golden_root, divergence_set["patterns"])
        divergence_set_results.append({**divergence_set, **result})

    ranked = sorted(
        [item for item in divergence_set_results if item["name"] != "all" and not item["ok"]],
        key=_target_priority,
    )
    next_target = _trace_next_target(audit) or (ranked[0] if ranked else None)

    return {
        "workflow": audit["workflow"],
        "dataset": audit["dataset"],
        "run_root": str(run_root),
        "golden_root": str(golden_root),
        "run_source": audit.get("run_source"),
        "run_generation": audit.get("run_generation"),
        "audit_ok": audit["ok"],
        "audit_failed": audit["failed"],
        "trace": audit.get("trace"),
        "divergence_sets": divergence_set_results,
        "next_target": next_target,
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve() if args.audit_output else _default_audit_output().resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    datasets = _resolved_datasets(args)

    audit_payload = _matching_audit_payload(audit_output, datasets)
    completed = None
    audit_source = "reused_existing_output"
    if audit_payload is None:
        completed = _run_validate_audit(args, audit_output)
        audit_source = "fresh_validate_audit_run"
        if not audit_output.exists():
            raise SystemExit(
                f"validate_audit did not write {audit_output}; stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        audit_payload = _load_json(audit_output)
    audits = [_evaluate_audit(audit) for audit in audit_payload.get("audits", [])]

    payload = {
        "generated_at_utc": _now_utc(),
        "code_state": audit_payload.get("code_state") or capture_code_state(_repo_root()),
        "contract_metadata": _contract_metadata(audit_payload),
        "datasets": datasets,
        "audit_source": audit_source,
        "audit_output": str(audit_output),
        "audit_returncode": completed.returncode if completed is not None else 0,
        "audit_ok": audit_payload.get("ok"),
        "audit_completed": audit_payload.get("completed"),
        "failed_workflows": audit_payload.get("failed_workflows", []),
        "audits": audits,
        "next_target": next((audit["next_target"] for audit in audits if audit["next_target"] is not None), None),
        "ok": bool(audit_payload.get("ok")),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

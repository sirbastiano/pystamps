#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from pystamps.config import RunConfig, load_config
from pystamps.parity_contract import (
    STAGE1_VERIFY_PATTERNS,
    STAGE2_CLEAN_PATTERNS,
    STAGE2_VERIFY_PATTERNS,
    STAGE3_CLEAN_PATTERNS,
    STAGE3_VERIFY_PATTERNS,
    STAGE4_CLEAN_PATTERNS,
    STAGE4_VERIFY_PATTERNS,
    STAGE25_CLEAN_PATTERNS,
    STAGE68_CLEAN_PATTERNS,
    STAGE28_CLEAN_PATTERNS,
    STAGE7_CLEAN_PATTERNS,
    STAGE78_CLEAN_PATTERNS,
    FULL_CLEAN_PATTERNS,
    build_parity_contract,
)
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext
from pystamps.verify import summarize_failures, verify_run_against_golden

_RUN_CONFIG_OVERRIDE: RunConfig | None = None
_STAGE_BOUNDARY_PATTERNS: dict[int, tuple[str, ...]] = {
    2: STAGE2_VERIFY_PATTERNS,
    3: STAGE3_VERIFY_PATTERNS,
    4: STAGE4_VERIFY_PATTERNS,
}


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
    parser.add_argument("--config", default=None, help="Optional run config path for generated run copies")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inputs_root() -> Path:
    return _repo_root() / "inputs_and_outputs"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_contract() -> dict[str, Any]:
    return build_parity_contract(_inputs_root())


def _workflow_targets(contract: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = contract.get("audited_workflow_manifest", {})
    return list(manifest.get("workflow_targets", []))


def _workflow_target_for_dataset(dataset_root: Path, contract: dict[str, Any] | None = None) -> dict[str, Any] | None:
    active_contract = contract or _resolve_contract()
    repo_root = _repo_root()
    dataset_path = dataset_root.expanduser().resolve()
    try:
        relative = str(dataset_path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        relative = None

    for target in _workflow_targets(active_contract):
        local_dataset_path = target.get("local_dataset_path")
        if not local_dataset_path:
            continue
        candidate = str(local_dataset_path)
        if relative is not None and candidate == relative:
            return target
        if Path(candidate).name == dataset_path.name:
            return target
    return None


def _target_seed_root(target: dict[str, Any] | None) -> Path | None:
    if not target:
        return None
    run_seed_path = target.get("run_seed_path")
    if not run_seed_path:
        return None
    seed_path = Path(str(run_seed_path)).expanduser()
    if not seed_path.is_absolute():
        seed_path = _repo_root() / seed_path
    return seed_path.resolve() if seed_path.exists() else None


def _clean_patterns_for_stage_range(start_step: int, end_step: int) -> tuple[str, ...]:
    mapping: dict[tuple[int, int], tuple[str, ...]] = {
        (1, 8): FULL_CLEAN_PATTERNS,
        (2, 2): STAGE2_CLEAN_PATTERNS,
        (3, 3): STAGE3_CLEAN_PATTERNS,
        (4, 4): STAGE4_CLEAN_PATTERNS,
        (2, 5): STAGE25_CLEAN_PATTERNS,
        (2, 8): STAGE28_CLEAN_PATTERNS,
        (4, 8): _stage48_clean_patterns(),
        (5, 8): _stage58_clean_patterns(),
        (6, 8): STAGE68_CLEAN_PATTERNS,
        (7, 7): STAGE7_CLEAN_PATTERNS,
        (7, 8): STAGE78_CLEAN_PATTERNS,
    }
    try:
        return mapping[(int(start_step), int(end_step))]
    except KeyError as exc:
        raise ValueError(f"Unsupported audit stage window {start_step}->{end_step}") from exc


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


def _copy_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        shutil.copytree(src, dst)


def _clean_outputs(dataset_root: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in dataset_root.glob(pattern):
            if path.is_file():
                path.unlink()


def _listed_patches(dataset_root: Path) -> list[str]:
    patch_list = dataset_root / "patch.list"
    if not patch_list.exists():
        return []
    return [
        line.strip()
        for line in patch_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_patch_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _authoritative_patch_names(run_root: Path, dataset_root: Path, workflow_profile: str) -> list[str]:
    candidates: list[Path]
    if workflow_profile == "legacy_post":
        candidates = [
            run_root / "patch.list_old",
            dataset_root / "patch.list_old",
            dataset_root / "patch.list",
            run_root / "patch.list",
        ]
    else:
        candidates = [dataset_root / "patch.list", run_root / "patch.list"]

    for candidate in candidates:
        names = _read_patch_manifest(candidate)
        if names:
            return names

    source_root = run_root if workflow_profile == "legacy_post" else dataset_root
    return [patch_dir.name for patch_dir in source_root.glob("PATCH_*") if patch_dir.is_dir()]


def _align_run_copy_with_dataset(run_root: Path, dataset_root: Path, workflow_profile: str = "default") -> None:
    listed_patches = _authoritative_patch_names(run_root, dataset_root, workflow_profile)
    allowed = set(listed_patches)
    if not allowed:
        return

    target_patch_list = run_root / "patch.list"
    if target_patch_list.exists():
        target_patch_list.unlink()
    target_patch_list.write_text("".join(f"{patch_name}\n" for patch_name in listed_patches), encoding="utf-8")
    for patch_dir in run_root.glob("PATCH_*"):
        if patch_dir.is_dir() and patch_dir.name not in allowed:
            shutil.rmtree(patch_dir)


def _has_stage1_artifacts(dataset_root: Path) -> bool:
    return all(any(dataset_root.glob(pattern)) for pattern in STAGE1_VERIFY_PATTERNS)


def _stage48_clean_patterns() -> tuple[str, ...]:
    stage5_onward = tuple(pattern for pattern in STAGE25_CLEAN_PATTERNS if pattern not in {"PATCH_*/pm1.mat", "PATCH_*/select1.mat"})
    return STAGE4_CLEAN_PATTERNS + stage5_onward + STAGE68_CLEAN_PATTERNS


def _stage58_clean_patterns() -> tuple[str, ...]:
    stage5_onward = tuple(pattern for pattern in STAGE25_CLEAN_PATTERNS if not pattern.startswith("PATCH_*/"))
    return stage5_onward + STAGE68_CLEAN_PATTERNS


def _seed_root_for_dataset(dataset_root: Path) -> Path:
    target = _workflow_target_for_dataset(dataset_root)
    target_seed = _target_seed_root(target)
    if target_seed is not None:
        return target_seed
    if dataset_root.name == "InSAR_dataset_test":
        explicit = _repo_root() / "inputs_and_outputs" / "RUN_FULL_GATE_1e10"
        if explicit.exists():
            return explicit.resolve()
    return dataset_root.resolve()


def _run_profile(dataset_root: Path) -> tuple[Path, int, int, tuple[str, ...], str, str]:
    target = _workflow_target_for_dataset(dataset_root)
    if target is not None and target.get("audit_start_step") is not None and target.get("audit_end_step") is not None:
        seed_root = _seed_root_for_dataset(dataset_root)
        start_step = int(target["audit_start_step"])
        end_step = int(target["audit_end_step"])
        clean_patterns = _clean_patterns_for_stage_range(start_step, end_step)
        workflow_profile = str(target.get("workflow_profile") or "default")
        return seed_root, start_step, end_step, clean_patterns, seed_root.name, workflow_profile

    seed_root = _seed_root_for_dataset(dataset_root)
    if seed_root.name == "RUN_FULL_GATE_1e10":
        if _has_stage1_artifacts(seed_root):
            return seed_root, 5, 8, _stage58_clean_patterns(), "RUN_FULL_GATE_1e10", "legacy_post"
        return seed_root, 4, 8, _stage48_clean_patterns(), "RUN_FULL_GATE_1e10", "legacy_post"
    if _has_stage1_artifacts(dataset_root):
        workflow_profile = str(target.get("workflow_profile") or "default") if target is not None else "default"
        return seed_root, 2, 8, STAGE28_CLEAN_PATTERNS, dataset_root.name, workflow_profile
    workflow_profile = str(target.get("workflow_profile") or "default") if target is not None else "default"
    return seed_root, 1, 8, FULL_CLEAN_PATTERNS, dataset_root.name, workflow_profile


def _build_run_copy(
    dataset_root: Path,
    audit_stamp: str,
    run_config: RunConfig | None = None,
) -> tuple[Path, dict[str, Any]]:
    seed_root, start_step, end_step, clean_patterns, seed_name, workflow_profile = _run_profile(dataset_root)
    validation_dir = _validation_runs_root() / audit_stamp
    validation_dir.mkdir(parents=True, exist_ok=True)
    run_root = validation_dir / f"{dataset_root.name}_stage{start_step}_{end_step}"
    _copy_dataset(seed_root, run_root)
    _align_run_copy_with_dataset(run_root, dataset_root, workflow_profile)
    _clean_outputs(run_root, clean_patterns)

    context = PipelineContext(
        dataset_root=run_root.resolve(),
        run_config=run_config or _RUN_CONFIG_OVERRIDE or RunConfig(),
        start_step=start_step,
        end_step=end_step,
        dry_run=False,
        workflow_profile=workflow_profile,
    )
    report = run_pipeline(context)
    failures = [
        {
            "stage": result.stage_id,
            "scope": result.scope,
            "target": result.target,
            "status": result.status,
            "details": result.details,
        }
        for result in report.failures
    ]
    if failures:
        raise RuntimeError(f"Pipeline regeneration failed for {dataset_root.name}: {json.dumps(failures)}")

    return run_root.resolve(), {
        "start_step": start_step,
        "end_step": end_step,
        "seed_root": str(seed_root),
        "seed_name": seed_name,
        "clean_patterns": list(clean_patterns),
        "validation_run_dir": str(validation_dir.resolve()),
        "workflow_profile": workflow_profile,
    }


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

    target = _workflow_target_for_dataset(dataset_root)
    target_seed = _target_seed_root(target)
    if target_seed is not None and (
        target.get("audit_start_step") is not None
        or str(target.get("run_seed_path")) != str(target.get("local_dataset_path"))
    ):
        return target_seed

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


def _prepare_run_selection(
    dataset_root: Path,
    golden_base: Path | None,
    audit_stamp: str,
) -> tuple[Path, Path, str, dict[str, Any] | None]:
    if golden_base is not None:
        run_root, golden_root, run_source = _resolve_run_selection(dataset_root, golden_base)
        return run_root, golden_root, run_source, None

    run_root, generation = _build_run_copy(dataset_root, audit_stamp)
    return run_root, dataset_root.resolve(), "generated_full_loop_run_copy", generation


def _verify_report(
    run_root: Path,
    golden_root: Path,
    tolerance: Any,
    *,
    patterns: tuple[str, ...] | None = None,
):
    if patterns is None:
        return verify_run_against_golden(run_root, golden_root, tolerance)
    try:
        return verify_run_against_golden(run_root, golden_root, tolerance, patterns=patterns)
    except TypeError:
        return verify_run_against_golden(run_root, golden_root, tolerance)


def _trace_output_dir(audit_stamp: str, generation: dict[str, Any] | None) -> Path:
    if generation is not None and generation.get("validation_run_dir"):
        output_dir = Path(str(generation["validation_run_dir"])).expanduser().resolve()
    else:
        output_dir = (_validation_runs_root() / audit_stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _oracle_source_payload(contract: dict[str, Any], golden_root: Path) -> dict[str, Any]:
    oracle_contract = contract.get("oracle_contract", {})
    precedence = oracle_contract.get("precedence_rule", {}).get("ordered_sources", [])
    source_name = str(precedence[0]) if precedence else "golden_root"
    source_payload = oracle_contract.get(source_name, {})
    return {
        "name": source_name,
        "golden_root": str(golden_root),
        "repository_url": source_payload.get("repository_url") or source_payload.get("upstream_repository_url"),
        "pinned_revision": source_payload.get("pinned_revision"),
    }


def _artifact_lineage(relative_path: str) -> list[dict[str, Any]]:
    path = Path(relative_path)
    patch_name = path.parts[0] if path.parts and path.parts[0].startswith("PATCH_") else None
    if patch_name is None:
        return [{"stage": None, "artifact_path": relative_path, "role": "failing_artifact"}]

    stage_inputs: dict[str, list[tuple[int, str]]] = {
        "pm1.mat": [
            (1, "ps1.mat"),
            (1, "ph1.mat"),
            (1, "bp1.mat"),
            (2, "pm1.mat"),
        ],
        "select1.mat": [
            (1, "ps1.mat"),
            (1, "ph1.mat"),
            (1, "bp1.mat"),
            (1, "da1.mat"),
            (2, "pm1.mat"),
            (3, "select1.mat"),
        ],
        "weed1.mat": [
            (1, "ps1.mat"),
            (1, "da1.mat"),
            (2, "pm1.mat"),
            (3, "select1.mat"),
            (4, "weed1.mat"),
        ],
    }
    lineage = stage_inputs.get(path.name)
    if lineage is None:
        return [{"stage": None, "artifact_path": relative_path, "role": "failing_artifact"}]
    return [
        {
            "stage": stage,
            "artifact_path": f"{patch_name}/{artifact}",
            "role": "failing_artifact" if artifact == path.name else "upstream_input",
        }
        for stage, artifact in lineage
    ]


def _stage_boundary_from_failure(failure: dict[str, Any] | None) -> int | None:
    if failure is None:
        return None
    stage_scope = str(failure.get("stage_scope") or "")
    if stage_scope == "stage2":
        return 2
    if stage_scope == "stage3":
        return 3
    if stage_scope in {"stage4", "stage3_4"}:
        return 4
    if stage_scope == "stage5_6":
        return 5
    if stage_scope == "stage7_8":
        return 7
    return None


def _trace_failure_payload(
    failure: dict[str, Any] | None,
    *,
    oracle_source: dict[str, Any],
    stage_boundary: int | None = None,
    probe_output_path: str | None = None,
) -> dict[str, Any] | None:
    if failure is None:
        return None
    return {
        "artifact_path": failure.get("path"),
        "stage_boundary": stage_boundary if stage_boundary is not None else _stage_boundary_from_failure(failure),
        "stage_scope": failure.get("stage_scope"),
        "failure_class": failure.get("failure_class"),
        "label": failure.get("label"),
        "guidance": failure.get("guidance"),
        "failure_kind": failure.get("failure_kind"),
        "failing_key": failure.get("failing_key"),
        "shape_run": failure.get("shape_run"),
        "shape_oracle": failure.get("shape_oracle"),
        "max_abs": failure.get("max_abs"),
        "message": failure.get("message"),
        "oracle_source": oracle_source,
        "artifact_lineage": _artifact_lineage(str(failure.get("path"))),
        "probe_output_path": probe_output_path,
    }


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _emit_stage_boundary_traces(
    run_root: Path,
    golden_root: Path,
    contract: dict[str, Any],
    tolerance: Any,
    *,
    audit_stamp: str,
    generation: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    output_dir = _trace_output_dir(audit_stamp, generation)
    oracle_source = _oracle_source_payload(contract, golden_root)

    probes: list[dict[str, Any]] = []
    first_probe_trace: dict[str, Any] | None = None

    for stage_boundary, patterns in _STAGE_BOUNDARY_PATTERNS.items():
        report = _verify_report(run_root, golden_root, tolerance, patterns=patterns)
        summary = summarize_failures(report)
        output_path = output_dir / f"{golden_root.name}_stage{stage_boundary}_boundary_probe.json"
        probe_trace = _trace_failure_payload(
            summary.get("first_boundary_failure"),
            oracle_source=oracle_source,
            stage_boundary=stage_boundary,
            probe_output_path=str(output_path),
        )
        probe_payload = {
            "generated_at_utc": _now_utc(),
            "dataset": golden_root.name,
            "workflow": _workflow_name(golden_root),
            "run_root": str(run_root),
            "golden_root": str(golden_root),
            "stage_boundary": stage_boundary,
            "patterns": list(patterns),
            "ok": summary.get("ok"),
            "checked": summary.get("checked"),
            "failed": summary.get("failed"),
            "first_boundary_failure": summary.get("first_boundary_failure"),
            "trace": probe_trace,
        }
        _write_json_artifact(output_path, probe_payload)
        probes.append({**probe_payload, "output_path": str(output_path)})
        if first_probe_trace is None and probe_trace is not None:
            first_probe_trace = probe_trace

    first_trace_path = output_dir / f"{golden_root.name}_first_boundary_trace.json"
    _write_json_artifact(
        first_trace_path,
        {
            "generated_at_utc": _now_utc(),
            "dataset": golden_root.name,
            "workflow": _workflow_name(golden_root),
            "run_root": str(run_root),
            "golden_root": str(golden_root),
            "first_divergent_boundary": first_probe_trace,
            "stage_boundary_probes": [probe["output_path"] for probe in probes],
        },
    )
    return probes, first_probe_trace, str(first_trace_path)


def _dataset_audit(
    run_root: Path,
    golden_root: Path,
    run_source: str,
    contract: dict[str, Any],
    audit_stamp: str,
    run_config: RunConfig | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_run_config = run_config or _RUN_CONFIG_OVERRIDE or RunConfig()
    report = _verify_report(run_root, golden_root, active_run_config.tolerance)
    summary = summarize_failures(report)
    stage_boundary_probes: list[dict[str, Any]] = []
    first_probe_trace: dict[str, Any] | None = None
    first_trace_path: str | None = None
    if not report.ok:
        stage_boundary_probes, first_probe_trace, first_trace_path = _emit_stage_boundary_traces(
            run_root,
            golden_root,
            contract,
            active_run_config.tolerance,
            audit_stamp=audit_stamp,
            generation=generation,
        )
    first_divergent_boundary = first_probe_trace or _trace_failure_payload(
        summary.get("first_boundary_failure"),
        oracle_source=_oracle_source_payload(contract, golden_root),
    )
    trace_payload = dict(summary.get("trace", {}))
    trace_payload["stage_boundary_probes"] = stage_boundary_probes
    trace_payload["first_divergent_boundary"] = first_divergent_boundary
    trace_payload["first_divergent_boundary_output_path"] = first_trace_path
    payload = {
        "workflow": _workflow_name(golden_root),
        "dataset": golden_root.name,
        "run_root": str(run_root),
        "golden_root": str(golden_root),
        "run_source": run_source,
        "ok": report.ok,
        "status": "passed" if report.ok else "failed",
        "checked": len(report.comparisons),
        **summary,
        "trace": trace_payload,
    }
    if generation is not None:
        payload["run_generation"] = generation
    return payload


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
    global _RUN_CONFIG_OVERRIDE
    args = _parse_args()
    config_path = getattr(args, "config", None)
    run_config = load_config(config_path) if config_path else RunConfig()
    _RUN_CONFIG_OVERRIDE = run_config
    try:
        contract = _resolve_contract()
        payload = _base_payload(contract)
        datasets, missing_from_request, extra_request = _resolve_datasets(args, contract)
        golden_base = Path(args.golden_root).expanduser().resolve() if args.golden_root else None
        audit_stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

        missing = [str(dataset) for dataset in datasets if not dataset.exists()]
        allow_subset = bool(getattr(args, "allow_subset", False))
        if extra_request or (missing_from_request and not allow_subset):
            if missing_from_request and not allow_subset:
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
                run_root, golden_root, run_source, generation = _prepare_run_selection(dataset, golden_base, audit_stamp)
                payload["audits"].append(
                    _dataset_audit(run_root, golden_root, run_source, contract, audit_stamp, run_config, generation)
                )
            payload["completed"] = True
        except FileNotFoundError as exc:
            payload["interrupted"] = True
            payload["interruption"] = {
                "kind": "missing_run_copy",
                "message": str(exc),
            }
            _emit_payload(_finalize_payload(payload), args.output)
            return 1
        except RuntimeError as exc:
            payload["interrupted"] = True
            payload["interruption"] = {
                "kind": "run_copy_generation_failed",
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
    finally:
        _RUN_CONFIG_OVERRIDE = None


if __name__ == "__main__":
    raise SystemExit(main())

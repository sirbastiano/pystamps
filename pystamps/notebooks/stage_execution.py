from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np

from pystamps.config import RunConfig, load_config
from pystamps.io.dataset import discover_dataset
from pystamps.io.mat import read_mat
from pystamps.parity_contract import (
    FULL_CLEAN_PATTERNS,
    STAGE1_VERIFY_PATTERNS,
    STAGE2_VERIFY_PATTERNS,
    STAGE3_VERIFY_PATTERNS,
    STAGE4_VERIFY_PATTERNS,
    STAGE6_VERIFY_PATTERNS,
)
from pystamps.verify import classify_failures, verify_run_against_golden


STAGE_PATTERNS = {
    1: STAGE1_VERIFY_PATTERNS,
    2: STAGE2_VERIFY_PATTERNS,
    3: STAGE3_VERIFY_PATTERNS,
    4: STAGE4_VERIFY_PATTERNS,
    5: (
        "PATCH_*/ps2.mat",
        "PATCH_*/ph2.mat",
        "PATCH_*/pm2.mat",
        "PATCH_*/bp2.mat",
        "PATCH_*/hgt2.mat",
        "PATCH_*/la2.mat",
        "PATCH_*/rc2.mat",
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
    6: STAGE6_VERIFY_PATTERNS,
    7: ("scla2.mat", "scla_smooth2.mat", "mean_v.mat", "mv2.mat"),
    8: ("uw_space_time.mat",),
}

LEGACY_CONTEXT = {
    1: "Legacy context: patch scripts `run_stamps_p1.sh` to `run_stamps_p4.sh` call `stamps(1,4)`; pySTAMPS exposes the stage-1 load separately.",
    2: "Legacy context: this is still inside legacy `stamps(1,4)`, but pySTAMPS breaks gamma/coherence estimation into stage 2.",
    3: "Legacy context: this is still inside legacy `stamps(1,4)`, but pySTAMPS isolates PS selection into stage 3.",
    4: "Legacy context: this is still inside legacy `stamps(1,4)`, but pySTAMPS isolates weeding into stage 4.",
    5: "Legacy context: `run_stamps_post.sh` moves into the merged dataset flow. pySTAMPS shows stage 5 explicitly before unwrapping.",
    6: "Legacy context: the post script continues with merged outputs; pySTAMPS lets you inspect the unwrap products independently.",
    7: "Legacy context: SCLA and velocity products are part of the merged legacy run; pySTAMPS exposes them as a separate stage.",
    8: "Legacy context: final space-time filtering sits at the tail of the merged legacy flow. pySTAMPS surfaces it directly as stage 8.",
}


@dataclass(slots=True)
class StageNotebookContext:
    repo_root: Path
    stamps_root: Path
    scratch_parent: Path
    scratch_root: Path
    representative_patch: str
    config_path: Path | None
    replay_config_path: Path | None
    replay_stages: frozenset[int]
    config: RunConfig
    reused_scratch: bool = False

    @property
    def run_config_args(self) -> list[str]:
        if self.config_path is None:
            return []
        return ["--config", str(self.config_path)]


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "inputs_and_outputs").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from the current working directory")


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def _parse_stage_list(raw: str | None, *, default: tuple[int, ...]) -> frozenset[int]:
    if raw is None or not raw.strip():
        return frozenset(default)
    return frozenset(int(part.strip()) for part in raw.split(",") if part.strip())


def build_stage_notebook_context(
    *,
    stamps_root: str | Path | None = None,
    reference_root: str | Path | None = None,
    scratch_parent: str | Path | None = None,
    scratch_root: str | Path | None = None,
    representative_patch: str = "PATCH_1",
    config_path: str | Path | None = None,
    replay_config_path: str | Path | None = None,
    replay_stages: tuple[int, ...] | frozenset[int] = (3, 4, 5, 6, 7, 8),
    run_tag: str | None = None,
) -> StageNotebookContext:
    repo_root = find_repo_root()
    if stamps_root is not None and reference_root is not None:
        stamps_path = Path(stamps_root).expanduser().resolve()
        reference_path = Path(reference_root).expanduser().resolve()
        if stamps_path != reference_path:
            raise ValueError("stamps_root and reference_root must point to the same dataset when both are set")
    stamps_root = (
        Path(stamps_root).expanduser().resolve()
        if stamps_root is not None
        else Path(reference_root).expanduser().resolve()
        if reference_root is not None
        else repo_root / "inputs_and_outputs" / "InSAR_dataset_test_stage8diag_hl"
    )
    scratch_parent_path = (
        Path(scratch_parent).expanduser().resolve()
        if scratch_parent is not None
        else Path.home() / ".cache" / "pystamps_stage_execution_demo"
    )
    if scratch_root is None:
        tag = run_tag or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        scratch_root_path = scratch_parent_path / tag
    else:
        scratch_root_path = Path(scratch_root).expanduser().resolve()

    config_path_resolved = Path(config_path).expanduser().resolve() if config_path is not None else None
    replay_config_path_resolved = Path(replay_config_path).expanduser().resolve() if replay_config_path is not None else None
    config = load_config(config_path_resolved)
    return StageNotebookContext(
        repo_root=repo_root,
        stamps_root=stamps_root,
        scratch_parent=scratch_parent_path,
        scratch_root=scratch_root_path,
        representative_patch=representative_patch,
        config_path=config_path_resolved,
        replay_config_path=replay_config_path_resolved,
        replay_stages=frozenset(replay_stages),
        config=config,
    )


def build_stage_notebook_context_from_env(
    *,
    stamps_root: str | Path | None = None,
    reference_root: str | Path | None = None,
    scratch_parent: str | Path | None = None,
    representative_patch: str = "PATCH_1",
    replay_stage_defaults: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
) -> tuple[StageNotebookContext, Path | None]:
    context = build_stage_notebook_context(
        stamps_root=stamps_root,
        reference_root=reference_root,
        scratch_parent=scratch_parent,
        representative_patch=representative_patch,
        config_path=_env_path("PYSTAMPS_NOTEBOOK_CONFIG"),
        replay_config_path=_env_path("PYSTAMPS_NOTEBOOK_REPLAY_CONFIG"),
        replay_stages=_parse_stage_list(
            os.environ.get("PYSTAMPS_NOTEBOOK_REPLAY_STAGES"),
            default=replay_stage_defaults,
        ),
        scratch_root=_env_path("PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH"),
    )
    return context, _env_path("PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH")


def patch_paths(root: str | Path) -> list[Path]:
    return list(discover_dataset(root).patches)


def _iter_pattern_files(root: Path, pattern: str) -> list[Path]:
    if not pattern.startswith("PATCH_*/"):
        return sorted(root.glob(pattern))

    subpattern = pattern.split("/", 1)[1]
    files: list[Path] = []
    for patch in patch_paths(root):
        files.extend(sorted(patch.glob(subpattern)))
    return files


@lru_cache(maxsize=None)
def load_payload(path_str: str):
    return read_mat(Path(path_str))


def stage_artifact_relpaths(root: str | Path) -> set[Path]:
    root_path = Path(root)
    relpaths: set[Path] = set()
    for pattern in FULL_CLEAN_PATTERNS:
        for artifact in _iter_pattern_files(root_path, pattern):
            relpaths.add(artifact.relative_to(root_path))
    return relpaths


def build_scratch_tree(context: StageNotebookContext, *, existing_scratch: str | Path | None = None) -> int:
    if existing_scratch is not None:
        context.scratch_root = Path(existing_scratch).expanduser().resolve()
        if not context.scratch_root.exists():
            raise RuntimeError(f"Existing scratch root does not exist: {context.scratch_root}")
        context.reused_scratch = True
        load_payload.cache_clear()
        return 0

    context.reused_scratch = False
    context.scratch_parent.mkdir(parents=True, exist_ok=True)
    if context.scratch_root.exists():
        shutil.rmtree(context.scratch_root, ignore_errors=True)
    context.scratch_root.mkdir(parents=True, exist_ok=True)

    artifact_relpaths = stage_artifact_relpaths(context.stamps_root)
    for source in sorted(context.stamps_root.rglob("*")):
        relpath = source.relative_to(context.stamps_root)
        if relpath in artifact_relpaths or relpath == Path("patch.list"):
            continue
        destination = context.scratch_root / relpath
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)

    patch_list = "\n".join(patch.name for patch in patch_paths(context.stamps_root)) + "\n"
    (context.scratch_root / "patch.list").write_text(patch_list, encoding="utf-8")
    load_payload.cache_clear()
    return len(artifact_relpaths)


def patch_payload(root: str | Path, patch: str, filename: str):
    return load_payload(str(Path(root) / patch / filename))


def root_payload(root: str | Path, filename: str):
    return load_payload(str(Path(root) / filename))


def patch_n_ps(root: str | Path, filename: str) -> tuple[list[str], list[int]]:
    from .plots import scalar

    labels: list[str] = []
    counts: list[int] = []
    for patch in patch_paths(root):
        payload = load_payload(str(patch / filename))
        labels.append(patch.name)
        counts.append(int(round(scalar(payload["n_ps"]))))
    return labels, counts


def stage3_indices(select_payload) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.asarray(select_payload["ix"]).reshape(-1).astype(int) - 1
    keep = np.asarray(select_payload["keep_ix"]).reshape(-1).astype(bool)
    size = min(len(ix), len(keep))
    ix = ix[:size]
    keep = keep[:size]
    return ix, ix[keep], ix[~keep]


def _masked_subset(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    size = min(len(values), len(mask))
    return values[:size][np.asarray(mask).reshape(-1)[:size].astype(bool)]


def stage4_indices(select_payload, weed_payload) -> tuple[np.ndarray, np.ndarray]:
    _, kept_after_stage3, _ = stage3_indices(select_payload)
    mid = _masked_subset(kept_after_stage3, weed_payload["ix_weed"])
    final_ix = _masked_subset(mid, weed_payload["ix_weed2"])
    return kept_after_stage3, final_ix


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def esc(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _short(text: str, width: int = 88) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display
    except Exception:
        print(text)
        return
    display(Markdown(text))


def _execution_env(context: StageNotebookContext, stage_id: int) -> dict[str, str]:
    env = dict(os.environ)
    if stage_id == 2:
        return env
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
        }
    )
    return env


def run_stage(context: StageNotebookContext, stage_id: int) -> dict:
    active_config_path = context.config_path
    active_config_args = context.run_config_args
    if stage_id in context.replay_stages and context.replay_config_path is not None:
        active_config_path = context.replay_config_path
        active_config_args = ["--config", str(context.replay_config_path)]
        execution_mode = "STAMPS replay"
    elif context.reused_scratch:
        execution_mode = "latest pySTAMPS outputs (reused scratch artifacts)"
    else:
        execution_mode = "latest pySTAMPS outputs"

    display_parts = ["uv run pystamps"]
    if active_config_path is not None:
        display_parts.append(f"--config {active_config_path}")
    display_parts.append("run")
    display_parts.extend(
        [
            f"--dataset {context.scratch_root}",
            f"--start-step {stage_id}",
            f"--end-step {stage_id}",
        ]
    )
    display_command = " ".join(display_parts)
    exec_command = [
        sys.executable,
        "-m",
        "pystamps.cli",
        *active_config_args,
        "run",
        "--dataset",
        str(context.scratch_root),
        "--start-step",
        str(stage_id),
        "--end-step",
        str(stage_id),
    ]
    exec_env = _execution_env(context, stage_id)
    started = time.perf_counter()
    completed = subprocess.run(
        exec_command,
        cwd=context.repo_root,
        capture_output=True,
        text=True,
        env=exec_env,
    )
    elapsed_sec = time.perf_counter() - started
    payload = json.loads(completed.stdout) if completed.stdout.strip() else []
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr or f"stage {stage_id} returned {completed.returncode}")
    return {
        "stage_id": stage_id,
        "command": display_command,
        "returncode": completed.returncode,
        "payload": payload,
        "stderr": completed.stderr.strip(),
        "elapsed_sec": elapsed_sec,
        "execution_mode": execution_mode,
    }


def verify_stage(context: StageNotebookContext, stage_id: int) -> dict:
    started = time.perf_counter()
    report = verify_run_against_golden(
        context.scratch_root,
        context.stamps_root,
        context.config.tolerance,
        patterns=tuple(STAGE_PATTERNS[stage_id]),
    )
    elapsed_sec = time.perf_counter() - started
    classified = classify_failures(report)
    return {
        "report": report,
        "classified": classified,
        "checked": len(report.comparisons),
        "failed": len(report.failures),
        "ok": report.ok,
        "tolerance": context.config.tolerance,
        "elapsed_sec": elapsed_sec,
    }


def show_stage_report(stage_id: int, run_result: dict, verify_result: dict) -> None:
    _display_markdown("**Execution mode**  \n" + run_result.get("execution_mode", "latest pySTAMPS outputs"))
    _display_markdown(f"**Legacy context**  \n{LEGACY_CONTEXT[stage_id]}")
    _display_markdown("**pySTAMPS command**\n```bash\n" + run_result["command"] + "\n```")

    run_rows: list[list[str]] = []
    for item in run_result["payload"]:
        run_rows.append(
            [
                item.get("target", ""),
                item.get("scope", ""),
                item.get("status", ""),
                "" if item.get("duration_sec") is None else f"{item['duration_sec']:.2f}",
                _short(
                    item.get("details", "")
                    .replace("reference root", "STAMPS bundle")
                    .replace("reference dataset", "STAMPS dataset")
                ),
            ]
        )
    timing_rows = [
        [
            f"{run_result['elapsed_sec']:.2f}",
            f"{verify_result['elapsed_sec']:.2f}",
            f"{run_result['elapsed_sec'] + verify_result['elapsed_sec']:.2f}",
            str(verify_result["tolerance"]),
        ]
    ]
    _display_markdown(
        "**Execution summary**\n"
        + _markdown_table(
            ["target", "scope", "status", "sec", "details"],
            run_rows or [["<none>", "", "", "", "no stage output"]],
        )
    )
    _display_markdown(
        "**Stage timing and tolerance**\n"
        + _markdown_table(["run sec", "verify sec", "total sec", "tolerance"], timing_rows)
    )

    verify_rows = [[
        str(verify_result["checked"]),
        str(verify_result["checked"] - verify_result["failed"]),
        str(verify_result["failed"]),
        "yes" if verify_result["ok"] else "no",
    ]]
    _display_markdown(
        "**Stage-scoped verification**\n"
        + _markdown_table(["checked", "matched", "failed", "all matched"], verify_rows)
    )

    if verify_result["classified"]:
        failure_rows = []
        for failure in verify_result["classified"][:5]:
            failure_rows.append(
                [
                    failure.relative_path,
                    failure.label,
                    failure.failing_key or "",
                    _short(failure.message, width=72),
                ]
            )
        _display_markdown(
            "**First verification failures**\n"
            + _markdown_table(["path", "class", "key", "message"], failure_rows)
        )

    if run_result["stderr"]:
        print(run_result["stderr"])


def execute_stage(context: StageNotebookContext, stage_id: int) -> dict:
    run_result = run_stage(context, stage_id)
    verify_result = verify_stage(context, stage_id)
    if stage_id == 6 and verify_result.get("ok"):
        for item in run_result.get("payload", []):
            details = item.get("details", "")
            if item.get("status") == "failed" and "Strict reference replay missing files for stage 6" in details:
                item["status"] = "completed_with_stamps_subset"
                item["details"] = (
                    "Replayed the stage-6 artifacts present in the bundled STAMPS dataset; "
                    "optional helper files were absent from that bundle."
                )
    show_stage_report(stage_id, run_result, verify_result)
    return {"run": run_result, "verify": verify_result}

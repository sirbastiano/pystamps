#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_notebook() -> Path:
    return _repo_root() / "notebooks" / "02_pystamps_stage_execution.ipynb"


def _default_config() -> Path:
    return _repo_root() / "notebooks" / "02_pystamps_stage_execution.validation.yaml"


def _mask_scratch_paths(text: str, scratch_parent: Path) -> str:
    pattern = re.compile(re.escape(str(scratch_parent)) + r"(?:/[^\s`'\"<>()]+)*")
    return pattern.sub("<scratch-dataset>", text)


def _sanitize_text(
    text: str,
    *,
    repo_root: Path,
    notebook_path: Path,
    output_path: Path,
    exec_cwd: Path,
    config_path: Path | None,
) -> str:
    replacements: list[tuple[str, str]] = []
    stamps_root = repo_root / "inputs_and_outputs" / "InSAR_dataset_test_stage8diag_hl"
    scratch_parent = Path.home() / ".cache" / "pystamps_stage_execution_demo"

    replacements.extend([
        (str(stamps_root), "<stamps-dataset>"),
        (str(repo_root), "<repo-root>"),
        (str(exec_cwd), "<exec-cwd>"),
        (str(notebook_path), "<notebook>"),
        (str(output_path), "<output-notebook>"),
    ])
    if config_path is not None:
        replacements.append((str(config_path), "<config>"))

    sanitized = _mask_scratch_paths(text, scratch_parent)
    for original, masked in replacements:
        sanitized = sanitized.replace(original, masked)
    return sanitized.replace(str(Path.home()), "~")


def _sanitize_value(value: Any, **kwargs: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, **kwargs)
    if isinstance(value, list):
        return [_sanitize_value(item, **kwargs) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item, **kwargs) for key, item in value.items()}
    return value


def _sanitize_notebook(
    notebook: nbformat.NotebookNode,
    *,
    repo_root: Path,
    notebook_path: Path,
    output_path: Path,
    exec_cwd: Path,
    config_path: Path | None,
) -> None:
    kwargs = {
        "repo_root": repo_root,
        "notebook_path": notebook_path,
        "output_path": output_path,
        "exec_cwd": exec_cwd,
        "config_path": config_path,
    }
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if "text" in output:
                output["text"] = _sanitize_value(output["text"], **kwargs)
            if "traceback" in output:
                output["traceback"] = _sanitize_value(output["traceback"], **kwargs)
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            for mime_type, payload in list(data.items()):
                if mime_type.startswith("text/") or mime_type == "application/json":
                    data[mime_type] = _sanitize_value(payload, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the stage-validation notebook and save the executed copy.")
    parser.add_argument("--notebook", default=str(_default_notebook()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cwd", default=str(_repo_root()))
    parser.add_argument("--config", default=str(_default_config()))
    parser.add_argument("--existing-scratch")
    parser.add_argument("--replay-config")
    parser.add_argument("--replay-stages", default="2,3,4,5,6,7,8")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--startup-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    notebook_path = Path(args.notebook).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    exec_cwd = Path(args.cwd).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    existing_scratch = Path(args.existing_scratch).expanduser().resolve() if args.existing_scratch else None
    replay_config_path = Path(args.replay_config).expanduser().resolve() if args.replay_config else None

    notebook = nbformat.read(notebook_path, as_version=4)
    old_config = os.environ.get("PYSTAMPS_NOTEBOOK_CONFIG")
    old_existing_scratch = os.environ.get("PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH")
    old_replay_config = os.environ.get("PYSTAMPS_NOTEBOOK_REPLAY_CONFIG")
    old_replay_stages = os.environ.get("PYSTAMPS_NOTEBOOK_REPLAY_STAGES")
    try:
        if config_path is not None:
            os.environ["PYSTAMPS_NOTEBOOK_CONFIG"] = str(config_path)
        if existing_scratch is not None:
            os.environ["PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH"] = str(existing_scratch)
        if replay_config_path is not None:
            os.environ["PYSTAMPS_NOTEBOOK_REPLAY_CONFIG"] = str(replay_config_path)
            os.environ["PYSTAMPS_NOTEBOOK_REPLAY_STAGES"] = str(args.replay_stages)
        client = NotebookClient(
            notebook,
            timeout=None if args.timeout is None or args.timeout <= 0 else args.timeout,
            startup_timeout=max(1, int(args.startup_timeout)),
            kernel_name="python3",
            resources={"metadata": {"path": str(exec_cwd)}},
        )
        client.execute()
    finally:
        if old_config is None:
            os.environ.pop("PYSTAMPS_NOTEBOOK_CONFIG", None)
        else:
            os.environ["PYSTAMPS_NOTEBOOK_CONFIG"] = old_config
        if old_existing_scratch is None:
            os.environ.pop("PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH", None)
        else:
            os.environ["PYSTAMPS_NOTEBOOK_EXISTING_SCRATCH"] = old_existing_scratch
        if old_replay_config is None:
            os.environ.pop("PYSTAMPS_NOTEBOOK_REPLAY_CONFIG", None)
        else:
            os.environ["PYSTAMPS_NOTEBOOK_REPLAY_CONFIG"] = old_replay_config
        if old_replay_stages is None:
            os.environ.pop("PYSTAMPS_NOTEBOOK_REPLAY_STAGES", None)
        else:
            os.environ["PYSTAMPS_NOTEBOOK_REPLAY_STAGES"] = old_replay_stages

    _sanitize_notebook(
        notebook,
        repo_root=repo_root,
        notebook_path=notebook_path,
        output_path=output_path,
        exec_cwd=exec_cwd,
        config_path=config_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

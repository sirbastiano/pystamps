from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "assert_notebook_parity.py"
SPEC = importlib.util.spec_from_file_location("assert_notebook_parity", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
notebook_parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = notebook_parity
SPEC.loader.exec_module(notebook_parity)


def _stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def _markdown(text: str) -> dict:
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "text/markdown": text.splitlines(keepends=True),
            "text/plain": ["<IPython.core.display.Markdown object>"],
        },
    }


def _stage_cell(stage: int, *, status: str = "completed") -> dict:
    return {
        "cell_type": "code",
        "execution_count": stage,
        "metadata": {},
        "source": [f"stage_{stage} = execute_stage(context, {stage})\n"],
        "outputs": [
            _markdown(
                "**Execution summary**\n"
                "| target | scope | status | sec | details |\n"
                "| --- | --- | --- | --- | --- |\n"
                f"| PATCH_1 | patch | {status} | 0.01 | done |\n"
            )
        ],
    }


def _summary_cell(*, failed: dict[int, int] | None = None, matched: dict[int, bool] | None = None) -> dict:
    failed = failed or {}
    matched = matched or {}
    lines = ["stage | mode | checked | failed | matched\n"]
    for stage in range(1, 9):
        stage_failed = failed.get(stage, 0)
        stage_matched = matched.get(stage, True)
        lines.append(
            f"{stage:>5} | latest pySTAMPS outputs | {1:>7} | {stage_failed:>6} | {stage_matched}\n"
        )
    return {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "source": ["stage_results = [stage_1, stage_2, stage_3, stage_4, stage_5, stage_6, stage_7, stage_8]\n"],
        "outputs": [_stream("".join(lines))],
    }


def _write_notebook(
    tmp_path: Path,
    *,
    setup_marker: bool = True,
    failed: dict[int, int] | None = None,
    matched: dict[int, bool] | None = None,
    statuses: dict[int, str] | None = None,
) -> Path:
    setup_lines = [
        "oracle: <repo-root>/inputs_and_outputs/InSAR_dataset_test_stage8diag_hl\n",
        "fresh scratch: <repo-root>/inputs_and_outputs/validation_runs/notebook_stage_by_stage/fresh_run\n",
    ]
    if setup_marker:
        setup_lines.append("resuming existing fresh scratch: no\n")
    cells = [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "source": ["context = build_stage_notebook_context()\n"],
            "outputs": [_stream("".join(setup_lines))],
        },
        *[_stage_cell(stage, status=(statuses or {}).get(stage, "completed")) for stage in range(1, 9)],
        _summary_cell(failed=failed, matched=matched),
    ]
    notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "executed.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def test_notebook_parity_script_accepts_clean_matched_run(tmp_path: Path, capsys) -> None:
    notebook = _write_notebook(tmp_path)

    exit_code = notebook_parity.main(["--notebook", str(notebook)])

    assert exit_code == 0
    assert "stages 1 through 8 all matched" in capsys.readouterr().out


def test_notebook_parity_script_reports_stage4_failure(tmp_path: Path, capsys) -> None:
    notebook = _write_notebook(tmp_path, failed={4: 1}, matched={4: False})

    exit_code = notebook_parity.main(["--notebook", str(notebook)])

    assert exit_code == 1
    assert "Stage 4 failed parity: failed=1, matched=False" in capsys.readouterr().err


def test_notebook_parity_script_requires_fresh_scratch_setup(tmp_path: Path, capsys) -> None:
    notebook = _write_notebook(tmp_path, setup_marker=False)

    exit_code = notebook_parity.main(["--notebook", str(notebook)])

    assert exit_code == 1
    assert "resuming existing fresh scratch: no" in capsys.readouterr().err


def test_notebook_parity_script_rejects_skipped_existing_stage(tmp_path: Path, capsys) -> None:
    notebook = _write_notebook(tmp_path, statuses={3: "skipped_existing"})

    exit_code = notebook_parity.main(["--notebook", str(notebook)])

    assert exit_code == 1
    assert "Stage 3 execution summary contains skipped_existing" in capsys.readouterr().err

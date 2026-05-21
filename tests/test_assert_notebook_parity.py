from __future__ import annotations

import json
from pathlib import Path
import runpy


def _load_assert_notebook_parity():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "assert_notebook_parity.py"
    namespace = runpy.run_path(str(module_path), run_name="assert_notebook_parity_test")
    assert "assert_notebook_parity" in namespace
    return namespace["assert_notebook_parity"]


def _build_notebook_payload(summaries: list[dict[str, object]]) -> dict:
    return {
        "cells": [
            {
                "outputs": [
                    {
                        "text": "resuming existing fresh scratch: no\n",
                    },
                    {
                        "text": (
                            f"**Execution summary**\n"
                            f"{summary['stage']} | {summary['mode']} | {summary['checked']} | "
                            f"{summary['failed']} | {summary['matched']}\n"
                            "completed\n"
                        )
                    },
                ],
                "source": f"stage_{summary['stage']} = execute_stage(context, {summary['stage']})\n",
            }
            for summary in summaries
        ]
    }


def test_assert_notebook_parity_stops_downstream_checks_after_stage2_fail(tmp_path: Path) -> None:
    assert_notebook_parity = _load_assert_notebook_parity()
    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(
            _build_notebook_payload(
                [
                    {"stage": 1, "mode": "exact", "checked": 10, "failed": 0, "matched": "True"},
                    {"stage": 2, "mode": "exact", "checked": 10, "failed": 1, "matched": "False"},
                ]
            )
        ),
        encoding="utf-8",
    )

    report = assert_notebook_parity(notebook_path)
    assert report.errors == [
        "Stage 2 failed parity: failed=1, matched=False",
    ]


def test_assert_notebook_parity_reports_stage3_failure_when_stage2_passes(tmp_path: Path) -> None:
    assert_notebook_parity = _load_assert_notebook_parity()
    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(
            _build_notebook_payload(
                [
                    {"stage": 1, "mode": "exact", "checked": 10, "failed": 0, "matched": "True"},
                    {"stage": 2, "mode": "exact", "checked": 10, "failed": 0, "matched": "True"},
                    {"stage": 3, "mode": "exact", "checked": 10, "failed": 1, "matched": "False"},
                ]
            )
        ),
        encoding="utf-8",
    )

    report = assert_notebook_parity(notebook_path)
    assert "Stage 3 failed parity: failed=1, matched=False" in report.errors


def test_assert_notebook_parity_stops_downstream_when_stage2_missing(tmp_path: Path) -> None:
    assert_notebook_parity = _load_assert_notebook_parity()
    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(
            _build_notebook_payload(
                [
                    {"stage": 1, "mode": "exact", "checked": 10, "failed": 0, "matched": "True"},
                    {"stage": 3, "mode": "exact", "checked": 10, "failed": 1, "matched": "False"},
                ]
            )
        ),
        encoding="utf-8",
    )

    report = assert_notebook_parity(notebook_path)
    assert report.errors == [
        "Stage 2 missing final parity summary",
        "Stage 2 missing execution summary",
    ]

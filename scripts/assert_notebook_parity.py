#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


EXPECTED_STAGES = tuple(range(1, 9))
SETUP_MARKER = "resuming existing fresh scratch: no"

_STAGE_CELL_RE = re.compile(r"\bstage_(\d+)\s*=\s*execute_stage\(\s*context\s*,\s*(\d+)\s*\)")
_STAGE_ROW_RE = re.compile(
    r"^\s*(?P<stage>[1-8])\s*\|\s*(?P<mode>.*?)\s*\|\s*"
    r"(?P<checked>\d+)\s*\|\s*(?P<failed>\d+)\s*\|\s*(?P<matched>True|False|true|false|yes|no)\s*$"
)


@dataclass(frozen=True)
class StageSummary:
    stage: int
    mode: str
    checked: int
    failed: int
    matched: bool


@dataclass(frozen=True)
class ParityReport:
    summaries: dict[int, StageSummary]
    errors: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_notebook() -> Path:
    return _repo_root() / "notebooks" / "03_stage_by_stage_oracle.ipynb"


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return ""


def _source_text(cell: dict[str, Any]) -> str:
    return _text_value(cell.get("source", ""))


def _iter_output_texts(cell: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for output in cell.get("outputs", []):
        text = _text_value(output.get("text"))
        if text:
            texts.append(text)
        traceback = _text_value(output.get("traceback"))
        if traceback:
            texts.append(traceback)
        data = output.get("data")
        if isinstance(data, dict):
            for mime_type in ("text/markdown", "text/plain", "application/json"):
                payload = _text_value(data.get(mime_type))
                if payload:
                    texts.append(payload)
    return texts


def _stage_id_from_source(source: str) -> int | None:
    for match in _STAGE_CELL_RE.finditer(source):
        lhs_stage = int(match.group(1))
        call_stage = int(match.group(2))
        if lhs_stage == call_stage:
            return lhs_stage
    return None


def _parse_bool(value: str) -> bool:
    return value.strip().casefold() in {"true", "yes"}


def _parse_stage_summaries(notebook: dict[str, Any]) -> dict[int, StageSummary]:
    summaries: dict[int, StageSummary] = {}
    for cell in notebook.get("cells", []):
        for text in _iter_output_texts(cell):
            for line in text.splitlines():
                match = _STAGE_ROW_RE.match(line)
                if match is None:
                    continue
                stage = int(match.group("stage"))
                summaries[stage] = StageSummary(
                    stage=stage,
                    mode=match.group("mode").strip(),
                    checked=int(match.group("checked")),
                    failed=int(match.group("failed")),
                    matched=_parse_bool(match.group("matched")),
                )
    return summaries


def _execution_summaries_by_stage(notebook: dict[str, Any]) -> dict[int, list[str]]:
    summaries: dict[int, list[str]] = {}
    for cell in notebook.get("cells", []):
        stage_id = _stage_id_from_source(_source_text(cell))
        if stage_id is None:
            continue
        for text in _iter_output_texts(cell):
            if "**Execution summary**" in text:
                summaries.setdefault(stage_id, []).append(text)
    return summaries


def assert_notebook_parity(notebook_path: Path) -> ParityReport:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_output_text = "\n".join(text for cell in notebook.get("cells", []) for text in _iter_output_texts(cell))
    stage_summaries = _parse_stage_summaries(notebook)
    execution_summaries = _execution_summaries_by_stage(notebook)

    errors: list[str] = []
    if SETUP_MARKER not in all_output_text:
        errors.append(f"Notebook setup output missing {SETUP_MARKER!r}")

    stage2_failed = False
    for stage in EXPECTED_STAGES:
        summary = stage_summaries.get(stage)

        if stage2_failed and stage > 2:
            continue

        if summary is None:
            errors.append(f"Stage {stage} missing final parity summary")
            if stage == 2:
                stage2_failed = True
        elif summary.failed > 0 or not summary.matched:
            errors.append(f"Stage {stage} failed parity: failed={summary.failed}, matched={summary.matched}")
            if stage == 2:
                stage2_failed = True

        summaries = execution_summaries.get(stage, [])
        if not summaries:
            errors.append(f"Stage {stage} missing execution summary")
            continue
        if any("skipped_existing" in text for text in summaries):
            errors.append(f"Stage {stage} execution summary contains skipped_existing")
        if not any("completed" in text for text in summaries):
            errors.append(f"Stage {stage} execution summary missing completed status")

    return ParityReport(summaries=stage_summaries, errors=errors)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert exact parity from the executed stage-by-stage notebook outputs.")
    parser.add_argument("--notebook", default=str(_default_notebook()), help="Executed notebook to inspect.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    notebook_path = Path(args.notebook).expanduser().resolve()
    try:
        report = assert_notebook_parity(notebook_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to read notebook parity outputs: {exc}", file=sys.stderr)
        return 1

    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("OK: stages 1 through 8 all matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

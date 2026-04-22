from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pystamps.config import RunConfig


StageScope = Literal["patch", "merged"]
WorkflowProfile = Literal["default", "legacy_post", "small_baseline"]


@dataclass(slots=True)
class PipelineContext:
    dataset_root: Path
    run_config: RunConfig
    start_step: int
    end_step: int
    dry_run: bool = False
    workflow_profile: WorkflowProfile = "default"


@dataclass(slots=True)
class StageResult:
    stage_id: int
    scope: StageScope
    target: str
    status: str
    details: str = ""
    duration_sec: float | None = None


@dataclass(slots=True)
class PipelineReport:
    results: list[StageResult] = field(default_factory=list)

    def add(self, result: StageResult) -> None:
        self.results.append(result)

    @property
    def failures(self) -> list[StageResult]:
        return [r for r in self.results if r.status == "failed"]

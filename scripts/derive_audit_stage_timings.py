from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


STAGE_PATTERNS = {
    2: ["PATCH_*/pm1.mat"],
    3: ["PATCH_*/select1.mat"],
    4: ["PATCH_*/weed1.mat"],
    5: ["ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "hgt2.mat", "la2.mat", "rc2.mat", "psver.mat"],
    6: ["ifgstd2.mat", "phuw2.mat", "uw_phaseuw.mat", "uw_grid.mat", "uw_interp.mat"],
    7: ["scla2.mat", "scla_smooth2.mat"],
    8: ["mean_v.mat", "mv2.mat", "uw_space_time.mat"],
}


@dataclass
class StageTiming:
    stage: int
    start_utc: str
    end_utc: str
    duration_sec: float
    artifact_count: int
    basis: str


def _parse_timestamp_from_validation_dir(path: Path) -> datetime:
    return datetime.strptime(path.name, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)


def _artifact_mtims(run_root: Path, patterns: list[str]) -> list[float]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(run_root.glob(pattern))
    return sorted(p.stat().st_mtime for p in files if p.exists())


def _derive_stage_timings(audit: dict) -> dict:
    run_root = Path(audit["run_root"])
    run_generation = audit.get("run_generation") or {}
    start_step = int(run_generation.get("start_step") or 0)
    validation_run_dir = Path(run_generation["validation_run_dir"])
    rows: list[StageTiming] = []
    previous_end: float | None = None

    for stage in range(start_step, 9):
        mtimes = _artifact_mtims(run_root, STAGE_PATTERNS.get(stage, []))
        if not mtimes:
            continue
        stage_end = max(mtimes)
        if previous_end is None:
            if stage == 2:
                start_dt = _parse_timestamp_from_validation_dir(validation_run_dir)
                stage_start = start_dt.timestamp()
                basis = "validation_run_dir_timestamp"
            else:
                stage_start = min(mtimes)
                basis = "fresh_artifact_span"
        else:
            stage_start = previous_end
            basis = "previous_stage_end"
        rows.append(
            StageTiming(
                stage=stage,
                start_utc=datetime.fromtimestamp(stage_start, UTC).isoformat(),
                end_utc=datetime.fromtimestamp(stage_end, UTC).isoformat(),
                duration_sec=round(stage_end - stage_start, 2),
                artifact_count=len(mtimes),
                basis=basis,
            )
        )
        previous_end = stage_end

    return {
        "dataset": audit["dataset"],
        "workflow": audit["workflow"],
        "run_root": str(run_root),
        "start_step": start_step,
        "end_step": int(run_generation.get("end_step") or 0),
        "timings": [asdict(row) for row in rows],
    }


def build_summary(audit_path: Path) -> dict:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "audit_path": str(audit_path),
        "timing_method": "artifact_mtime_derived",
        "notes": [
            "Stage 2 duration uses the validation-run directory timestamp as the workflow start when stage 2 is regenerated.",
            "For workflows that start at stage 4 from a seeded run copy, stage 4 duration is derived from the span of freshly rewritten stage-4 artifacts because copied seed files preserve historical mtimes.",
        ],
        "audits": [_derive_stage_timings(audit) for audit in payload.get("audits", [])],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = build_summary(args.audit)
    encoded = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

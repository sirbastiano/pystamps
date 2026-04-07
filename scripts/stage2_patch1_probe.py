#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from pystamps.config import RunConfig, RuntimeConfig
from pystamps.pipeline.stages import run_pipeline
from pystamps.pipeline.types import PipelineContext


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--patch", default="PATCH_1")
    parser.add_argument("--kernel-backend", default="native")
    parser.add_argument("--native-threads", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint-mode", default="final")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset).resolve()
    run_root = Path(args.run_root).resolve()
    patch_name = str(args.patch)

    run_root.parent.mkdir(parents=True, exist_ok=True)
    if run_root.exists():
        shutil.rmtree(run_root)
    try:
        shutil.copytree(dataset_root, run_root, copy_function=os.link)
    except OSError:
        shutil.copytree(dataset_root, run_root)

    (run_root / "patch.list").write_text(f"{patch_name}\n", encoding="utf-8")
    for patch_dir in run_root.glob("PATCH_*"):
        if patch_dir.is_dir() and patch_dir.name != patch_name:
            shutil.rmtree(patch_dir)

    pm1_path = run_root / patch_name / "pm1.mat"
    if pm1_path.exists():
        pm1_path.unlink()

    cfg = RunConfig(
        runtime=RuntimeConfig(
            stage2_kernel_backend=str(args.kernel_backend),
            stage2_native_threads=int(args.native_threads),
            stage2_debug=bool(args.debug),
            stage2_checkpoint_mode=str(args.checkpoint_mode),
            stage2_checkpoint_interval=int(args.checkpoint_interval),
        )
    )
    report = run_pipeline(
        PipelineContext(
            dataset_root=run_root,
            run_config=cfg,
            start_step=2,
            end_step=2,
            dry_run=False,
        )
    )
    print(f"failures={len(report.failures)}")
    for failure in report.failures:
        print(
            "failure",
            {
                "stage": failure.stage_id,
                "scope": failure.scope,
                "target": failure.target,
                "status": failure.status,
                "details": failure.details,
            },
        )
    print(f"pm1_exists={pm1_path.exists()}")
    if pm1_path.exists():
        print(f"pm1_md5={_md5(pm1_path)}")
        print(f"pm1_mtime_ns={pm1_path.stat().st_mtime_ns}")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

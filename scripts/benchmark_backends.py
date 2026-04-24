#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_BACKENDS = ("threads", "processes", "native", "gpu", "auto")
DEFAULT_CLEAN_PATTERNS = (
    "PATCH_*/ps1.mat",
    "PATCH_*/pm1.mat",
    "PATCH_*/select1.mat",
    "PATCH_*/weed1.mat",
    "PATCH_*/ps2.mat",
    "PATCH_*/ph2.mat",
    "PATCH_*/pm2.mat",
    "PATCH_*/bp2.mat",
    "PATCH_*/hgt2.mat",
    "PATCH_*/la2.mat",
    "PATCH_*/rc2.mat",
    "ps2.mat",
    "ph2.mat",
    "pm2.mat",
    "bp2.mat",
    "ifgstd2.mat",
    "phuw2.mat",
    "uw_phaseuw.mat",
    "uw_grid.mat",
    "uw_interp.mat",
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "uw_space_time.mat",
)

STAGE67_CLEAN_PATTERNS = (
    "ifgstd2.mat",
    "phuw2.mat",
    "uw_phaseuw.mat",
    "uw_grid.mat",
    "uw_interp.mat",
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "uw_space_time.mat",
)

STAGE78_CLEAN_PATTERNS = (
    "scla2.mat",
    "scla_smooth2.mat",
    "mean_v.mat",
    "uw_space_time.mat",
)


def _clean_outputs(dataset: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in dataset.glob(pattern):
            if path.is_file():
                path.unlink()


def _copy_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        shutil.copytree(src, dst)


def _normalized_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _choose_clean_patterns(start_step: int, clean_profile: str) -> tuple[str, ...]:
    profile = clean_profile.strip().lower()
    if profile == "full":
        return tuple(DEFAULT_CLEAN_PATTERNS)
    if profile == "stage67":
        return tuple(STAGE67_CLEAN_PATTERNS)
    if profile == "stage78":
        return tuple(STAGE78_CLEAN_PATTERNS)
    if profile != "auto":
        raise SystemExit(f"Unknown --clean-profile: {clean_profile}")

    if start_step >= 7:
        return tuple(STAGE78_CLEAN_PATTERNS)
    if start_step >= 6:
        return tuple(STAGE67_CLEAN_PATTERNS)
    return tuple(DEFAULT_CLEAN_PATTERNS)


def _write_config(
    path: Path,
    backend: str,
    stage2_kernel_backend: str,
    stage2_native_threads: int,
    io_workers: int,
    cpu_workers: int,
    stage7_chunk_ps: int,
    stage8_chunk_edges: int,
    enable_cache: bool,
) -> None:
    text = (
        "runtime:\n"
        f"  backend: {backend}\n"
        f"  stage2_kernel_backend: {stage2_kernel_backend}\n"
        f"  stage2_native_threads: {stage2_native_threads}\n"
        f"  io_workers: {io_workers}\n"
        f"  cpu_workers: {cpu_workers}\n"
        f"  stage7_chunk_ps: {stage7_chunk_ps}\n"
        f"  stage8_chunk_edges: {stage8_chunk_edges}\n"
        f"  enable_mat_stage_cache: {'true' if enable_cache else 'false'}\n"
        "tolerance:\n"
        "  rtol: 1.0e-10\n"
        "  atol: 1.0e-10\n"
        "  wrap_equivalence: true\n"
        "  wrap_period: 6.283185307179586\n"
    )
    path.write_text(text, encoding="utf-8")


def _run_once(cfg: Path, dataset: Path, start_step: int, end_step: int) -> tuple[float, int, str, str]:
    cmd = [
        "uv",
        "run",
        "pystamps",
        "--config",
        str(cfg),
        "run",
        "--dataset",
        str(dataset),
        "--start-step",
        str(start_step),
        "--end-step",
        str(end_step),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_normalized_env())
    dt = time.perf_counter() - t0
    return dt, proc.returncode, proc.stdout, proc.stderr


def _parse_stage_durations(stdout: str) -> dict[str, float]:
    payload = json.loads(stdout)
    if not isinstance(payload, list):
        return {}
    out: dict[str, float] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        stage = row.get("stage")
        dur = row.get("duration_sec")
        if stage is None or dur is None:
            continue
        try:
            out[str(stage)] = float(dur)
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pySTAMPS backends with repeated timed runs.")
    parser.add_argument("--dataset", required=True, help="Dataset path to run on")
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    parser.add_argument("--baseline-backend", default="threads")
    parser.add_argument("--outdir", default="inputs_and_outputs/benchmarks")
    parser.add_argument(
        "--clean-profile",
        default="auto",
        help="auto | full | stage67 | stage78 (auto selects from start-step)",
    )
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=0)
    parser.add_argument("--stage2-kernel-backend", default="auto")
    parser.add_argument("--stage2-native-threads", type=int, default=0)
    parser.add_argument("--stage7-chunk-ps", type=int, default=100000)
    parser.add_argument("--stage8-chunk-edges", type=int, default=200000)
    parser.add_argument("--disable-stage-cache", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset does not exist: {dataset}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_out = outdir / f"backend_bench_{stamp}.json"
    csv_out = outdir / f"backend_bench_{stamp}.csv"
    run_root = outdir / f"backend_bench_runs_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    patterns = _choose_clean_patterns(args.start_step, args.clean_profile)
    results: dict[str, dict[str, Any]] = {}

    for backend in args.backends:
        cfg = outdir / f".bench_cfg_{backend}_{stamp}.yaml"
        _write_config(
            cfg,
            backend=backend,
            stage2_kernel_backend=args.stage2_kernel_backend,
            stage2_native_threads=args.stage2_native_threads,
            io_workers=args.io_workers,
            cpu_workers=args.cpu_workers,
            stage7_chunk_ps=args.stage7_chunk_ps,
            stage8_chunk_edges=args.stage8_chunk_edges,
            enable_cache=not args.disable_stage_cache,
        )

        errors: list[str] = []
        for i in range(args.warmup):
            run_copy = run_root / f"{backend}_warmup_{i+1}"
            _copy_dataset(dataset, run_copy)
            _clean_outputs(run_copy, patterns)
            dt, rc, out, err = _run_once(cfg, run_copy, args.start_step, args.end_step)
            if rc != 0:
                errors.append(f"warmup#{i+1} rc={rc} dt={dt:.3f}s stderr={err.strip()} stdout={out.strip()}")
                break

        runs: list[float] = []
        stage_runs: list[dict[str, float]] = []
        if not errors:
            for i in range(args.repeat):
                run_copy = run_root / f"{backend}_run_{i+1}"
                _copy_dataset(dataset, run_copy)
                _clean_outputs(run_copy, patterns)
                dt, rc, out, err = _run_once(cfg, run_copy, args.start_step, args.end_step)
                if rc != 0:
                    errors.append(f"run#{i+1} rc={rc} dt={dt:.3f}s stderr={err.strip()} stdout={out.strip()}")
                    break
                runs.append(dt)
                stage_runs.append(_parse_stage_durations(out))

        row: dict[str, Any] = {
            "backend": backend,
            "stage2_kernel_backend": args.stage2_kernel_backend,
            "stage2_native_threads": args.stage2_native_threads,
            "ok": len(errors) == 0 and len(runs) == args.repeat,
            "runs_sec": runs,
            "stage_runs_sec": stage_runs,
            "errors": errors,
        }
        if runs:
            row["mean_sec"] = float(statistics.mean(runs))
            row["stdev_sec"] = float(statistics.pstdev(runs))
            row["min_sec"] = float(min(runs))
            row["max_sec"] = float(max(runs))
        if stage_runs:
            stage_mean: dict[str, float] = {}
            keys = sorted({k for run in stage_runs for k in run})
            for k in keys:
                vals = [run[k] for run in stage_runs if k in run]
                if vals:
                    stage_mean[k] = float(statistics.mean(vals))
            row["stage_mean_sec"] = stage_mean
        results[backend] = row

    baseline = results.get(args.baseline_backend, {}).get("mean_sec")
    if isinstance(baseline, float) and baseline > 0:
        for backend in args.backends:
            m = results[backend].get("mean_sec")
            if isinstance(m, float) and m > 0:
                results[backend]["speedup_vs_baseline"] = baseline / m

    payload = {
        "dataset": str(dataset),
        "start_step": args.start_step,
        "end_step": args.end_step,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "baseline_backend": args.baseline_backend,
        "backends": args.backends,
        "results": results,
        "run_root": str(run_root),
        "env": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "json_path": str(json_out),
        "csv_path": str(csv_out),
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "backend",
                "ok",
                "mean_sec",
                "stdev_sec",
                "min_sec",
                "max_sec",
                "speedup_vs_baseline",
                "runs_sec",
                "errors",
            ]
        )
        for backend in args.backends:
            row = results[backend]
            w.writerow(
                [
                    backend,
                    row.get("ok", False),
                    row.get("mean_sec", ""),
                    row.get("stdev_sec", ""),
                    row.get("min_sec", ""),
                    row.get("max_sec", ""),
                    row.get("speedup_vs_baseline", ""),
                    ";".join(f"{v:.6f}" for v in row.get("runs_sec", [])),
                    " || ".join(row.get("errors", [])),
                ]
            )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

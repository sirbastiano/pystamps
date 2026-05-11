# pySTAMPS Function Reference

This document organizes the main package surface for readers who want to understand what pySTAMPS exposes and where different responsibilities live.

It focuses on user-facing and module-level entry points first. For very large internal modules, especially `pystamps.pipeline.ported`, related helpers are grouped by purpose instead of documenting every private helper line by line.

## Package overview

Core areas:
- `pystamps.cli`: command-line entrypoints
- `pystamps.config`: configuration dataclasses and config loading
- `pystamps.io.dataset`: dataset discovery and stage inference
- `pystamps.io.mat`: MAT-file reading and writing
- `pystamps.status`: dataset status reporting
- `pystamps.verify`: run-versus-reference comparison
- `pystamps.pipeline.types`: runtime data structures
- `pystamps.pipeline.stages`: orchestration across stages 1 through 8
- `pystamps.pipeline.ported`: stage implementations and helpers
- `pystamps.kernels.*`: accelerated kernels and registry
- `pystamps.runtime.executor`: hybrid execution infrastructure
- `pystamps.compat.legacy`: legacy StaMPS command discovery
- `pystamps.parity_contract`: repo parity contract helpers

## `pystamps.__init__`

### `__version__`
Package version exported from `pystamps._version`.

## `pystamps.cli`

Purpose: command-line interface for inspecting datasets, running stages, verifying outputs, listing legacy commands, and reporting backend coverage.

### `_parse_args() -> argparse.Namespace`
Builds and parses the CLI arguments.

### `_load_run_config(path: str | None) -> RunConfig`
Loads a config file and converts configuration errors into user-facing CLI exits.

### `_cmd_status(dataset: str) -> int`
Runs the dataset status command and prints a JSON payload.

### `_cmd_run(args: argparse.Namespace, run_config: RunConfig) -> int`
Applies worker overrides, builds a `PipelineContext`, runs the pipeline, prints stage results as JSON, and returns a process exit code.

### `_cmd_verify(run: str, golden: str, run_config: RunConfig) -> int`
Runs verification between a run directory and a golden directory, prints a JSON summary, and returns success or failure.

### `_resolve_stamps_root(stamps_root: str | None) -> str`
Resolves the StaMPS root by preferring an explicit argument and then `STAMPS_ROOT` from the environment.

### `_cmd_list_legacy(stamps_root: str | None) -> int`
Discovers legacy StaMPS commands and prints them as JSON.

### `_cmd_describe_backends() -> int`
Prints the registered backend providers and per-kernel coverage manifest as JSON.

### `main() -> int`
Top-level CLI dispatcher.

## `pystamps.config`

Purpose: structured configuration for runtime behavior, numeric tolerances, external tools, and compatibility modes.

### Dataclasses
- `RuntimeConfig`: runtime backend and worker settings
- `ToleranceConfig`: numeric comparison tolerances for verification
- `ExternalToolsConfig`: paths or names for required external tools
- `CompatibilityConfig`: compatibility and strict-reference settings
- `RunConfig`: top-level runtime configuration bundle

### Other types
- `ConfigError`: raised when the config file is invalid

### Functions
- `normalize_runtime_backend(name: str) -> str`: normalize runtime backend aliases
- `normalize_kernel_backend(name: str) -> str`: normalize generic kernel backend aliases
- `normalize_stage2_kernel_backend(name: str) -> str`: restrict stage-2 kernel backends to `auto|python|native`
- `_load_raw(path: Path) -> dict[str, Any]`: read a YAML or JSON config payload
- `_as_dict(payload: dict[str, Any], key: str) -> dict[str, Any]`: helper for safe nested-dict extraction
- `load_config(path: str | Path | None = None) -> RunConfig`: load a config file or defaults into a `RunConfig`

## `pystamps.status`

Purpose: summarize what stage a dataset and each patch appear to have reached.

### Dataclasses
- `PatchStatus`: patch name and inferred stage
- `DatasetStatus`: dataset path, merged stage, and per-patch status

### Functions
- `collect_status(dataset_root: str | Path) -> DatasetStatus`: inspect a dataset root and report discovered stage state

## `pystamps.io.dataset`

Purpose: discover dataset layout and infer stage progress from filesystem state.

### Dataclasses
- `DatasetLayout`: normalized dataset structure used by the pipeline

### Other types
- `DatasetError`: raised when the dataset layout is invalid

### Functions
- `_patch_sort_key(path: Path) -> tuple[int, str]`: stable patch ordering helper
- `discover_dataset(root: str | Path) -> DatasetLayout`: inspect dataset root and enumerate patches
- `infer_patch_stage(patch_dir: str | Path) -> int`: infer the current stage of one patch directory
- `infer_merged_stage(root_dir: str | Path) -> int`: infer merged dataset stage from output artifacts
- `expected_stage_artifact(stage: int, scope: str) -> str | None`: map stage and scope to the expected artifact name

## `pystamps.io.mat`

Purpose: read and write MATLAB `.mat` payloads used throughout the workflow.

### Other types
- `MatReadError`: raised when a MAT file cannot be read in a supported way

### Functions
- `_decode_h5_dataset(obj: Any, h5file: Any) -> Any`: helper for MAT v7.3 / HDF5 payload decoding
- `read_mat(path: str | Path) -> dict[str, Any]`: load a MAT file into a Python dictionary
- `write_mat(path: str | Path, payload: dict[str, Any]) -> None`: write a Python dictionary to a MAT file

## `pystamps.verify`

Purpose: compare a run directory with a golden dataset and summarize mismatches.

### Dataclasses
- `FileComparison`: one file-level comparison result
- `VerificationReport`: aggregate report over many file comparisons
- `FailureClassification`: metadata for a failure class
- `ClassifiedFailure`: classified view of one failure record

### Public and module-level functions
- `_is_numeric(value: Any) -> bool`: numeric-type helper
- `_to_array(value: Any) -> np.ndarray`: normalize scalar-like values to arrays
- `_collect_numeric(payload: Any, prefix: str = "") -> dict[str, np.ndarray]`: flatten numeric payloads for comparison
- `_compare_mat(run_mat: Path, golden_mat: Path, tol: ToleranceConfig) -> tuple[bool, str]`: compare MAT payloads under tolerance settings
- `_iter_pattern_files(root: Path, pattern: str) -> list[Path]`: collect files matching a parity pattern
- `_extract_failure_key(message: str) -> str | None`: parse a failing key name from a mismatch message
- `classify_failure(relative_path: str) -> FailureClassification`: map a path to a broad failure class
- `classify_failures(report: VerificationReport) -> list[ClassifiedFailure]`: classify all failures in a report
- `summarize_failures(report: VerificationReport) -> dict[str, Any]`: build a structured summary for troubleshooting
- `verify_run_against_golden(...) -> VerificationReport`: compare all configured patterns between a run and a golden dataset

## `pystamps.pipeline.types`

Purpose: shared runtime data structures passed between orchestration and stage implementations.

### Dataclasses
- `PipelineContext`: dataset root, selected stages, runtime config, and dry-run settings
- `StageResult`: one stage execution record
- `PipelineReport`: collection of stage results plus failures

## `pystamps.pipeline.stages`

Purpose: orchestration layer for deciding which stages run, in what order, and with what execution strategy.

### Dataclasses and exceptions
- `StageDef`: definition of one pipeline stage
- `StageExecutionError`: raised when orchestration cannot execute a stage correctly

### Functions
- `_normalize_backend(name: str) -> str`: normalize backend naming
- `_task_kind_for_stage(stage: StageDef, context: PipelineContext, patch_count: int = 0) -> str`: classify execution mode for scheduling
- `_stage2_uses_full_cpu_default(stage: StageDef, context: PipelineContext) -> bool`: decide when stage 2 should claim the full detected CPU budget by default
- `_replay_from_reference(...)`: reuse or replay outputs from a reference root when compatibility mode requests it
- `_run_ported_patch_stage(...)`: run one Python-ported patch stage
- `_run_patch_stage(...)`: run one patch-level stage
- `_run_patch_stage_timed(...)`: timed wrapper for patch execution
- `_run_merged_stage(...)`: run one merged-stage operation, with optional forced rerun support for explicit replay cases
- `_run_merged_stage_timed(...)`: timed wrapper for merged execution
- `_selected_stages(start_step: int, end_step: int) -> list[StageDef]`: select the active stage definitions for a requested range
- `run_pipeline(context: PipelineContext) -> PipelineReport`: main pipeline orchestration entrypoint

`PipelineContext` still carries `workflow_profile`, but the wrapper-backed legacy post flow is modeled inside stage ownership rather than by expanding the outer stage list.

## `pystamps.pipeline.ported`

Purpose: Python implementations of the stage logic and the internal numerical helpers they depend on.

### Main exceptions and dataclasses
- `PortedStageError`: raised for stage-level implementation failures
- `StageOptions`: stage-level options resolved from a patch
- `Parms`: loaded parameter bundle
- `Stage5PatchBundle`: merged-stage preparation bundle
- `Stage1MetadataResolution`: resolved metadata needed by stage 1

### Public stage entrypoints
- `resolve_stage1_metadata(...) -> Stage1MetadataResolution`
- `stage1_load_initial(patch_dir: Path, backend: str = "auto") -> str`
- `stage2_estimate_gamma(patch_dir: Path, backend: str = "auto", kernel_backend: str = "auto", kernel_backend_overrides: dict[str, str] | None = None, native_threads: int = 0, checkpoint_mode: str = "final", checkpoint_interval: int = 1, debug: bool = False) -> str`
- `stage3_select_ps(patch_dir: Path, backend: str = "auto") -> str`
- `stage4_weed_ps(...) -> str`
- `stage5_correct_and_promote(patch_dir: Path, backend: str = "auto") -> str`
- `stage5_merge_and_ifgstd(...) -> str`
- `stage6_unwrap(...) -> str`
- `stage7_calc_scla(...) -> str`
- `stage8_filter_scn(...) -> str`

### Internal helper groups in `ported`

Because `ported.py` is large, the helpers are best understood by purpose.

#### Dataset and file resolution helpers
Examples:
- `_resolve_file`
- `_stage1_dataset_root`
- `_snap_ifg_records`
- `_resolve_rslc_par`
- `_discover_patch_dirs`

These functions locate input files, patch folders, and metadata sources.

#### MAT and type-coercion helpers
Examples:
- `_read_mat_cached`
- `_cache_mat_payload`
- `_coerce_1d`
- `_coerce_complex`
- `_mat_scalar`

#### Stage-2 kernel helpers
Examples:
- `_normalize_stage2_kernel_backend`
- `_normalize_kernel_backend_override_map`
- `_normalize_stage2_native_threads`
- `_normalize_stage2_checkpoint_mode`
- `_ps_topofit_batch`
- `run_stage2_grid_accumulate_kernel`
- `run_stage2_histogram_kernel`
- `run_stage2_topofit_kernel`
- `_mat_text`
- `_matlab_col`
- `_matlab_row`
- `_matlab_char_row`

These functions normalize MAT payloads and array shapes to match the expected numerical code paths.

#### Numerical fitting and filtering helpers
Examples:
- `_weighted_lstsq`
- `_weighted_slope_fit`
- `_weighted_affine_fit`

For the merged post flow, stage 7 owns the raw `scla2.mat` output and the smoothed `scla_smooth2.mat` envelope. Stage 8 then reruns the final unwrap-backed products and writes `mean_v.mat` together with the final `uw_space_time.mat` result.
- `_stage7_mean_velocity_fit`
- `_clap_filter_kernel`
- `_clap_filt_patch`
- `_clap_filt_grid`
- `_wrap_filt`
- `_wrap_filt_global`
- `_ps_topofit_single`
- `_ps_topofit_batch`

These helpers perform the estimation and filtering operations used inside multiple stages.

#### Geometry and graph helpers
Examples:
- `_local_xy_from_lonlat`
- `_delaunay_edges`
- `_load_triangle_edges`
- `_adjacent_component_keep_mask`
- `_select_reference_ps`

These helpers support patch geometry, graph construction, and reference-point selection.

#### Stage-specific support helpers
Examples:
- `_stage2_psquare_weighting`
- `_stage2_weighting_snapshot_targets`
- `_stage7_unwrap_ifg_sets`
- `_deramp_unwrapped_phase`
- `_build_stage_options`
- `_build_low_pass`
- `_load_stage5_patch_bundle`

These helpers exist to keep the numbered stage functions manageable.

## `pystamps.kernels.accelerated`

Purpose: accelerated CPU and GPU kernels used by later numerical stages.

### Types
- `BackendUnavailableError`: raised when a requested backend cannot be used

### Functions
- `_cupy() -> Any | None`: lazy CuPy resolver
- `_resolve_backend(backend: str) -> str`: backend selection helper
- `_resolve_stage2_kernel_backend(backend: str) -> str`: stage-2 backend selection helper
- `_to_numpy(arr: Any) -> np.ndarray`: normalize backend arrays to NumPy
- `_cov_from_accumulators(...) -> np.ndarray`: covariance helper
- `_auto_chunk_size(...) -> int`: choose chunk sizes for memory-aware execution
- `stage2_native_available() -> bool`: probe for the compiled stage-2 extension
- `stage7_native_available() -> bool`: probe for the compiled native stage-7 export
- `stage8_native_available() -> bool`: probe for the compiled native stage-8 export
- `run_stage2_grid_accumulate_kernel(...)`: public stage-2 grid accumulation wrapper
- `run_stage2_topofit_kernel(...)`: public stage-2 generic topofit wrapper
- `run_stage2_topofit_row_invariant_kernel(...)`: public stage-2 row-invariant topofit wrapper
- `run_stage2_topofit_coh_row_invariant_kernel(...)`: public stage-2 coherence-only row-invariant wrapper
- `run_stage2_histogram_kernel(...)`: public stage-2 histogram wrapper
- `_stage4_edge_stats_python(...)`
- `_stage4_edge_stats_native(...)`
- `run_stage4_edge_stats_kernel(ph_weed, node_a, node_b, bperp, day, time_win, small_baseline, backend='auto', threads=0) -> dict[str, np.ndarray]`: public stage-4 edge-statistics kernel wrapper
- `_stage7_scla_cpu(...)`
- `_stage7_scla_gpu(...)`
- `_stage7_scla_native(...)`
- `_stage8_edge_noise_cpu(...)`
- `_stage8_edge_noise_gpu(...)`
- `_stage8_edge_noise_native(...)`
- `run_stage7_scla_kernel(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std, backend='auto', chunk_ps=0) -> dict[str, np.ndarray]`: public stage-7 kernel wrapper
- `run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend='auto', chunk_edges=0) -> dict[str, np.ndarray]`: public stage-8 kernel wrapper
- `describe_backend_matrix() -> dict[str, Any]`: report providers and per-kernel backend coverage

### Practical kernel examples

List registered providers and per-kernel coverage:

```bash
uv run pystamps describe-backends
```

Call the public Python API:

```python
from pystamps.kernels import describe_backend_matrix

matrix = describe_backend_matrix()
print(matrix["kernels"]["stage8_edge_noise"]["available_backends"])
```

Run the optimized stage-8 kernel on arrays loaded from the repo golden dataset:

```python
from pathlib import Path
import numpy as np

from pystamps.io.mat import read_mat
from pystamps.kernels import run_stage8_edge_noise_kernel

root = Path("inputs_and_outputs/InSAR_dataset_test_stage8diag")
uw_grid = read_mat(root / "uw_grid.mat")
uw_interp = read_mat(root / "uw_interp.mat")

uw_ph = np.asarray(uw_grid["ph"][:1000, :8], dtype=np.complex64)
edges = np.asarray(uw_interp["edgs"], dtype=np.int64)
node_a = edges[:, 1] - 1
node_b = edges[:, 2] - 1
valid = (
    (node_a >= 0)
    & (node_b >= 0)
    & (node_a < uw_ph.shape[0])
    & (node_b < uw_ph.shape[0])
)

out = run_stage8_edge_noise_kernel(
    uw_ph,
    node_a[valid][:2000],
    node_b[valid][:2000],
    backend="native",
)
print(out["dph_noise"].shape)
```

Use `backend="python"` for the reference path, `backend="native"` for the Rust/CPU path, and `backend="cuda"` when CuPy and that specific kernel backend are available.

## `pystamps.kernels.registry`

Purpose: register and resolve kernel implementations.

### Dataclasses
- `BackendProvider`
- `KernelImplementation`
- `ResolvedKernel`
- `KernelRegistry`

## `pystamps.runtime.executor`

Purpose: hybrid execution infrastructure for thread and process pools.

### Classes
- `HybridExecutor`: context-managed executor that coordinates IO and CPU worker pools

## `pystamps.compat.legacy`

Purpose: locate legacy StaMPS commands from a StaMPS checkout.

### Functions
- `discover_legacy_commands(stamps_root: str | Path = "StaMPS") -> list[Path]`

## `pystamps.parity_contract`

Purpose: define the supported parity validation contract for the repo-maintained datasets.

### Functions
- `_is_dataset_dir(path: Path) -> bool`: identify candidate dataset directories
- `discover_golden_datasets(inputs_root: str | Path) -> list[Path]`: discover dataset roots under `inputs_and_outputs`
- `_relative_to_repo(path: Path, repo_root: Path) -> str`: helper for repo-relative paths
- `_dataset_payload(dataset: Path, repo_root: Path) -> dict[str, Any]`: build a dataset summary payload
- `build_parity_contract(inputs_root: str | Path) -> dict[str, Any]`: build the full supported audit contract payload

## How to use this reference

If you are operating the package as a user:
1. start with `pystamps.cli`
2. look at `pystamps.config` when you need tuning
3. use `pystamps.status`, `pystamps.pipeline.stages`, and `pystamps.verify` as the main mental model

If you are debugging or extending the package:
1. inspect `pystamps.pipeline.types`
2. follow `pystamps.pipeline.stages.run_pipeline`
3. drop into the numbered entrypoints in `pystamps.pipeline.ported`
4. inspect `pystamps.kernels.accelerated` and `pystamps.runtime.executor` for performance-sensitive paths

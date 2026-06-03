# Getting Started With pySTAMPS

This is the shortest path to run pySTAMPS without tying documentation to repository-owned datasets.

## What pySTAMPS does

pySTAMPS executes a numbered stage pipeline (`1` to `8`) over a dataset directory.
It detects existing artifacts, so stage ranges can be run incrementally:

- `status`: inspect dataset and stage state
- `run`: execute a selected range
- `verify`: compare a run result against a reference dataset
- `pystamps-native`: run the standalone Rust native pipeline and coverage checks

For detailed stage meanings, artifacts, and code entrypoints, read [stages.html](stages.html).

## Install

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
uv run pystamps describe-backends
```

## Basic run workflow (copy-first)

```bash
export SOURCE_DATASET=/path/to/your_dataset
export RUN_DATASET=/path/to/run_dataset
cp -a "$SOURCE_DATASET" "$RUN_DATASET"

# inspect current progress
uv run pystamps status --dataset "$RUN_DATASET"

# rehearse without writing outputs
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8 --dry-run
```

Run individual stages by number when you need controlled execution:

```bash
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 1
uv run pystamps run --dataset "$RUN_DATASET" --start-step 2 --end-step 2
uv run pystamps run --dataset "$RUN_DATASET" --start-step 3 --end-step 3
uv run pystamps run --dataset "$RUN_DATASET" --start-step 4 --end-step 4
uv run pystamps run --dataset "$RUN_DATASET" --start-step 5 --end-step 5
uv run pystamps run --dataset "$RUN_DATASET" --start-step 6 --end-step 6
uv run pystamps run --dataset "$RUN_DATASET" --start-step 7 --end-step 7
uv run pystamps run --dataset "$RUN_DATASET" --start-step 8 --end-step 8
```

For full processing:

```bash
uv run pystamps run --dataset "$RUN_DATASET" --start-step 1 --end-step 8
```

## Native Rust execution

Use the Rust CLI when you want the native-only path without the Python wrapper:

```bash
cargo build --release -p pystamps-core --bin pystamps-native

target/release/pystamps-native run \
  --native-only \
  --dataset "$RUN_DATASET" \
  --start-step 1 \
  --end-step 8 \
  --backend native \
  --stage2-kernel-backend native \
  --cpu-workers 0 \
  --stage2-native-threads 0
```

Preview native coverage:

```bash
target/release/pystamps-native coverage --start-step 1 --end-step 8
```

`0` worker values mean pySTAMPS uses all CPUs visible to the process. Set positive values to cap resource use.

## Configure kernel backends

```bash
uv run pystamps describe-backends
```

```yaml
runtime:
  backend: auto
  stage2_kernel_backend: native
  stage2_native_threads: 0
  kernel_backend_overrides:
    stage2_grid_accumulate: native
    stage2_histogram: native
    stage2_topofit: native
    stage2_topofit_row_invariant: native
    stage2_topofit_coh_row_invariant: native
    stage4_edge_stats: native
    stage7_scla: native
    stage8_edge_noise: native
  io_workers: 8
  cpu_workers: 0
  stage7_chunk_ps: 100000
  stage8_chunk_edges: 200000
```

```bash
cat > native-kernels.yaml <<'YAML'
# above YAML block
YAML

uv run pystamps --config native-kernels.yaml run --dataset "$RUN_DATASET" --start-step 2 --end-step 8
```

Use `python` in place of `native` for reference execution paths if you are debugging numerical differences.

## Native validation gate

```bash
make native-full-chain-verify
```

Override paths and CPU limits when reproducing on a VM:

```bash
make native-full-chain-verify \
  DATASET=/path/to/source_dataset \
  GOLDEN=/path/to/reference_dataset \
  RUN=/path/to/clean_run_dataset \
  THREADS=8
```

The gate writes reports under `RUN/_native_gate_reports/`.

## Verify

```bash
export GOLDEN_DATASET=/path/to/reference_dataset
uv run pystamps verify --run "$RUN_DATASET" --golden "$GOLDEN_DATASET"
```

## Stage meaning (quick map)

| Stage | Typical intent |
|---|---|
| 1 | Prepare candidate-level patch artifacts |
| 2 | Compute quality metrics and model terms |
| 3 | Select persistent candidates |
| 4 | Weed out weak/redundant candidates |
| 5 | Merge patch outputs into dataset-level products |
| 6 | Unwrap temporal products |
| 7 | Estimate SCLA correction terms |
| 8 | Apply final space-time filtering |

The full stage guide is [docs/stages.html](stages.html).

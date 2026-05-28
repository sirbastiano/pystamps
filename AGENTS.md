# AGENTS.md

## Build And Test

- Run the Rust workspace gate with `cargo test --workspace`.
- Run the accelerated kernel gate with `uv run pytest -q tests/test_kernels_accelerated.py`.
- Run the full native validation gate with `make native-full-chain-verify`; it uses the checked-in performance budget manifest and honors the `THREADS` override.
- For VM reproduction, build `target/release/pystamps-native` first, then run `make native-full-chain-verify`; override `DATASET`, `GOLDEN`, `RUN`, and `THREADS` as needed, and inspect `RUN/_native_gate_reports/`.
- Run the local web console with `make web` when changing `crates/pystamps-web`.
- Build Python distributions with `make build` only for packaging work; avoid committing generated `dist/` artifacts during normal implementation.

## Native Rust Surface

- Native execution scaffolding lives under `crates/`.
- `crates/pystamps-core` owns dataset discovery, planning, coverage reporting, and the `pystamps-native` CLI.
- `crates/pystamps-mat` owns Rust MAT v5 writing helpers.
- `crates/pystamps-stages` owns the native stage ownership/readiness registry.
- `crates/pystamps-parity` owns shared parity comparison result types.

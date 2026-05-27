use pystamps_mat::MatData;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub mod mat_v5;
pub mod native_stage1;
pub mod native_stage2;
pub mod native_stage3;
pub mod native_stage4;
pub mod native_stage5;
pub mod native_stage6;
pub mod native_stage7;
pub mod native_stage8;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("dataset does not exist: {0}")]
    MissingDataset(PathBuf),
    #[error("dataset path is not a directory: {0}")]
    DatasetNotDirectory(PathBuf),
    #[error("unable to read dataset directory {path}: {source}")]
    ReadDataset {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid stage range {start_step}..{end_step}; expected 1..8")]
    InvalidStageRange { start_step: u8, end_step: u8 },
    #[error("full native Rust processing chain is incomplete: {0}")]
    IncompleteNativeChain(String),
    #[error("native-only execution violation: {0}")]
    NativeOnlyViolation(String),
    #[error("unable to write runtime config {path}: {source}")]
    WriteRuntimeConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to start execution command '{program}': {source}")]
    StartExecution {
        program: String,
        source: std::io::Error,
    },
    #[error("stage {stage} native implementation error: {message}")]
    NativeStage { stage: u8, message: String },
    #[error("unable to access {path}: {source}")]
    FileIo {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Mat(#[from] pystamps_mat::MatError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunRequest {
    pub dataset_root: PathBuf,
    pub start_step: u8,
    pub end_step: u8,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOptions {
    pub backend: String,
    pub stage2_kernel_backend: String,
    pub io_workers: u16,
    pub cpu_workers: u16,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            backend: "native".to_string(),
            stage2_kernel_backend: "native".to_string(),
            io_workers: 8,
            cpu_workers: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliBridgeOptions {
    pub command: Vec<String>,
    pub runtime: RuntimeOptions,
    pub native_only: bool,
}

impl Default for CliBridgeOptions {
    fn default() -> Self {
        Self {
            command: vec!["uv".to_string(), "run".to_string(), "pystamps".to_string()],
            runtime: RuntimeOptions::default(),
            native_only: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineExecution {
    pub results: Vec<StageResult>,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct StageResult {
    pub stage: u8,
    pub scope: StageScope,
    pub target: String,
    pub status: StageStatus,
    pub details: String,
    pub duration_sec: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_artifact_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_artifact_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_processed: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_peak_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_grid_ps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_grid_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_grid_cols: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_edges: Option<usize>,
}

impl StageResult {
    pub fn new(
        stage: u8,
        scope: StageScope,
        target: impl Into<String>,
        status: StageStatus,
        details: impl Into<String>,
        duration_sec: Option<f64>,
    ) -> Self {
        Self {
            stage,
            scope,
            target: target.into(),
            status,
            details: details.into(),
            duration_sec,
            input_artifact_count: None,
            output_artifact_count: None,
            rows_processed: None,
            memory_peak_bytes: None,
            n_grid_ps: None,
            n_grid_rows: None,
            n_grid_cols: None,
            n_edges: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StageCoverage {
    pub stage: u8,
    pub scope: StageScope,
    pub target: String,
    pub rust_driver: bool,
    pub native_stage: bool,
    pub parity_certified: bool,
    pub disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_parity_certified_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_native_reason: Option<String>,
    pub unsupported_modes: Vec<UnsupportedExecutionMode>,
    pub native_kernels: &'static [&'static str],
    pub details: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UnsupportedExecutionMode {
    pub mode: &'static str,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StageScope {
    Patch,
    Merged,
}

impl fmt::Display for StageScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageScope::Patch => f.write_str("patch"),
            StageScope::Merged => f.write_str("merged"),
        }
    }
}

impl StageScope {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "patch" => Some(Self::Patch),
            "merged" => Some(Self::Merged),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Planned,
    Completed,
    Failed,
    PendingExecution,
    Skipped,
    SkippedExisting,
}

impl fmt::Display for StageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageStatus::Planned => f.write_str("planned"),
            StageStatus::Completed => f.write_str("completed"),
            StageStatus::Failed => f.write_str("failed"),
            StageStatus::PendingExecution => f.write_str("pending_execution"),
            StageStatus::Skipped => f.write_str("skipped"),
            StageStatus::SkippedExisting => f.write_str("skipped_existing"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageDef {
    pub stage_id: u8,
    pub name: &'static str,
    pub scope: StageScope,
}

pub const STAGE_DEFS: [StageDef; 8] = [
    StageDef {
        stage_id: 1,
        name: "Initial load",
        scope: StageScope::Patch,
    },
    StageDef {
        stage_id: 2,
        name: "Estimate gamma",
        scope: StageScope::Patch,
    },
    StageDef {
        stage_id: 3,
        name: "Select PS pixels",
        scope: StageScope::Patch,
    },
    StageDef {
        stage_id: 4,
        name: "Weed adjacent pixels",
        scope: StageScope::Patch,
    },
    StageDef {
        stage_id: 5,
        name: "Correct phase + merge",
        scope: StageScope::Patch,
    },
    StageDef {
        stage_id: 6,
        name: "Unwrap phase",
        scope: StageScope::Merged,
    },
    StageDef {
        stage_id: 7,
        name: "Calculate SCLA",
        scope: StageScope::Merged,
    },
    StageDef {
        stage_id: 8,
        name: "Filter SCN",
        scope: StageScope::Merged,
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetLayout {
    pub root: PathBuf,
    pub patches: Vec<PathBuf>,
}

pub fn discover_dataset(path: impl AsRef<Path>) -> Result<DatasetLayout, CoreError> {
    let root = path.as_ref().to_path_buf();
    if !root.exists() {
        return Err(CoreError::MissingDataset(root));
    }
    if !root.is_dir() {
        return Err(CoreError::DatasetNotDirectory(root));
    }

    let patch_list = root.join("patch.list");
    if patch_list.exists() {
        let text = fs::read_to_string(&patch_list).map_err(|source| CoreError::FileIo {
            path: patch_list.clone(),
            source,
        })?;
        let mut patches = Vec::new();
        for name in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if name.starts_with('#') {
                continue;
            }
            let patch = root.join(name);
            if !valid_patch_name(name) || !patch.is_dir() {
                return Err(CoreError::DatasetNotDirectory(patch));
            }
            patches.push(patch);
        }
        if !patches.is_empty() {
            return Ok(DatasetLayout { root, patches });
        }
    }

    let mut patches = Vec::new();
    let entries = fs::read_dir(&root).map_err(|source| CoreError::ReadDataset {
        path: root.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| CoreError::ReadDataset {
            path: root.clone(),
            source,
        })?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.starts_with("PATCH_") {
            patches.push(path);
        }
    }
    patches.sort_by_key(|path| patch_sort_key(path));

    Ok(DatasetLayout { root, patches })
}

fn patch_sort_key(path: &Path) -> (u64, String) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let suffix = name.strip_prefix("PATCH_").unwrap_or(name);
    let numeric = suffix.parse::<u64>().unwrap_or(u64::MAX);
    (numeric, name.to_string())
}

fn valid_patch_name(name: &str) -> bool {
    name.starts_with("PATCH_") && !name.contains('/') && !name.contains('\\')
}

pub fn plan_pipeline(request: &RunRequest) -> Result<Vec<StageResult>, CoreError> {
    validate_stage_range(request.start_step, request.end_step)?;
    let dataset = discover_dataset(&request.dataset_root)?;
    let mut results = Vec::new();

    for stage in selected_stages(request.start_step, request.end_step) {
        match stage.scope {
            StageScope::Patch => {
                for patch in &dataset.patches {
                    results.push(plan_single_scope(
                        stage.stage_id,
                        StageScope::Patch,
                        patch,
                        patch
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or("unknown"),
                        request.dry_run,
                    ));
                }
                if stage.stage_id == 5 {
                    results.push(plan_single_scope(
                        5,
                        StageScope::Merged,
                        &dataset.root,
                        dataset_name(&dataset.root),
                        request.dry_run,
                    ));
                }
            }
            StageScope::Merged => {
                results.push(plan_single_scope(
                    stage.stage_id,
                    StageScope::Merged,
                    &dataset.root,
                    dataset_name(&dataset.root),
                    request.dry_run,
                ));
            }
        }
    }

    Ok(results)
}

pub fn execute_pipeline_cli_bridge(
    request: &RunRequest,
    options: &CliBridgeOptions,
) -> Result<PipelineExecution, CoreError> {
    validate_stage_range(request.start_step, request.end_step)?;
    if options.native_only {
        return Err(CoreError::NativeOnlyViolation(
            "native-only mode forbids CLI bridge execution; call pystamps-native run directly"
                .to_string(),
        ));
    }
    let _dataset = discover_dataset(&request.dataset_root)?;

    let config_path = temp_runtime_config_path();
    fs::write(&config_path, runtime_config_text(&options.runtime)).map_err(|source| {
        CoreError::WriteRuntimeConfig {
            path: config_path.clone(),
            source,
        }
    })?;

    let output = run_cli_bridge_command(request, options, &config_path);
    let _ = fs::remove_file(&config_path);
    let output = output?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let results = serde_json::from_str::<Vec<StageResult>>(&stdout).unwrap_or_default();

    Ok(PipelineExecution {
        results,
        stdout,
        stderr,
        exit_code: output.status.code(),
    })
}

pub fn processing_chain_coverage(
    start_step: u8,
    end_step: u8,
) -> Result<Vec<StageCoverage>, CoreError> {
    processing_chain_coverage_with_disabled(
        start_step,
        end_step,
        &disabled_native_stages_from_env(),
    )
}

pub fn processing_chain_coverage_with_disabled(
    start_step: u8,
    end_step: u8,
    disabled_stages: &[(u8, StageScope)],
) -> Result<Vec<StageCoverage>, CoreError> {
    validate_stage_range(start_step, end_step)?;
    let mut coverage = Vec::new();
    for stage in selected_stages(start_step, end_step) {
        match stage.scope {
            StageScope::Patch => {
                coverage.push(stage_coverage(
                    stage.stage_id,
                    StageScope::Patch,
                    "PATCH_*",
                    disabled_stages,
                ));
                if stage.stage_id == 5 {
                    coverage.push(stage_coverage(
                        5,
                        StageScope::Merged,
                        "dataset root",
                        disabled_stages,
                    ));
                }
            }
            StageScope::Merged => coverage.push(stage_coverage(
                stage.stage_id,
                StageScope::Merged,
                "dataset root",
                disabled_stages,
            )),
        }
    }
    Ok(coverage)
}

pub fn verify_full_native_processing_chain(start_step: u8, end_step: u8) -> Result<(), CoreError> {
    verify_full_native_processing_chain_with_disabled(start_step, end_step, &[])
}

pub fn verify_full_native_processing_chain_with_disabled(
    start_step: u8,
    end_step: u8,
    disabled_stages: &[(u8, StageScope)],
) -> Result<(), CoreError> {
    let missing: Vec<String> =
        processing_chain_coverage_with_disabled(start_step, end_step, disabled_stages)?
            .into_iter()
            .filter(|coverage| !coverage.native_stage)
            .map(|coverage| {
                format!(
                    "stage {} {} ({})",
                    coverage.stage, coverage.scope, coverage.target
                )
            })
            .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(CoreError::IncompleteNativeChain(missing.join(", ")))
    }
}

pub fn enrich_stage_result_telemetry(dataset_root: impl AsRef<Path>, result: &mut StageResult) {
    let dataset_root = dataset_root.as_ref();
    let target_dir = stage_target_dir(dataset_root, result);
    result.input_artifact_count = Some(count_stage_input_artifacts(
        dataset_root,
        &target_dir,
        result.stage,
        result.scope,
    ));
    result.output_artifact_count = Some(count_existing_artifacts(
        &target_dir,
        stage_output_artifacts(result.stage, result.scope),
    ));
    result.rows_processed = rows_processed_for_stage(&target_dir, result.stage, result.scope);
    result.memory_peak_bytes = current_process_peak_rss_bytes();
    if result.stage == 6 && result.scope == StageScope::Merged {
        enrich_stage6_grid_telemetry(&target_dir, result);
    }
}

fn validate_stage_range(start_step: u8, end_step: u8) -> Result<(), CoreError> {
    if start_step == 0 || end_step == 0 || start_step > end_step || end_step > 8 {
        return Err(CoreError::InvalidStageRange {
            start_step,
            end_step,
        });
    }
    Ok(())
}

fn selected_stages(start_step: u8, end_step: u8) -> impl Iterator<Item = StageDef> {
    STAGE_DEFS
        .into_iter()
        .filter(move |stage| start_step <= stage.stage_id && stage.stage_id <= end_step)
}

fn stage_coverage(
    stage_id: u8,
    scope: StageScope,
    target: &'static str,
    disabled_stages: &[(u8, StageScope)],
) -> StageCoverage {
    let scope_name = scope.to_string();
    let disabled = disabled_stages.contains(&(stage_id, scope));
    let parity_certified = pystamps_stages::native_stage_is_parity_certified(stage_id, &scope_name);
    let disabled_reason = if disabled {
        Some("disabled by PYSTAMPS_DISABLE_NATIVE_STAGES".to_string())
    } else {
        None
    };
    let not_parity_certified_reason = if parity_certified {
        None
    } else {
        Some(stage_coverage_details(stage_id, scope).to_string())
    };
    let not_native_reason = if let Some(reason) = &disabled_reason {
        Some(reason.clone())
    } else {
        not_parity_certified_reason.clone()
    };
    let native_stage = !disabled && parity_certified;
    StageCoverage {
        stage: stage_id,
        scope,
        target: target.to_string(),
        rust_driver: true,
        native_stage,
        parity_certified,
        disabled,
        disabled_reason,
        not_parity_certified_reason,
        not_native_reason,
        unsupported_modes: unsupported_native_only_modes(),
        native_kernels: native_kernel_acceleration(stage_id, scope),
        details: stage_coverage_details(stage_id, scope),
    }
}

fn unsupported_native_only_modes() -> Vec<UnsupportedExecutionMode> {
    vec![
        UnsupportedExecutionMode {
            mode: "python",
            reason: "native-only mode accepts only Rust-owned stage execution; Python is limited to verifier and reference tooling",
        },
        UnsupportedExecutionMode {
            mode: "matlab",
            reason: "native-only mode forbids MATLAB shell-outs during stage execution",
        },
        UnsupportedExecutionMode {
            mode: "octave",
            reason: "native-only mode forbids Octave shell-outs during stage execution",
        },
        UnsupportedExecutionMode {
            mode: "bridge",
            reason: "native-only mode must call pystamps-native directly instead of the Python CLI bridge",
        },
    ]
}

fn stage_coverage_details(stage_id: u8, scope: StageScope) -> &'static str {
    let scope_name = scope.to_string();
    pystamps_stages::native_stage_details(stage_id, &scope_name)
}

fn disabled_native_stages_from_env() -> Vec<(u8, StageScope)> {
    std::env::var("PYSTAMPS_DISABLE_NATIVE_STAGES")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(disable_token_to_stage_scope)
                .collect::<Vec<(u8, StageScope)>>()
        })
        .unwrap_or_default()
}

fn disable_token_to_stage_scope(token: &str) -> Option<(u8, StageScope)> {
    let token = token.trim();
    let (stage, scope) = token.split_once(':')?;
    Some((
        stage.trim().parse::<u8>().ok()?,
        StageScope::parse(scope.trim())?,
    ))
}

fn native_kernel_acceleration(stage_id: u8, scope: StageScope) -> &'static [&'static str] {
    match (stage_id, scope) {
        (2, StageScope::Patch) => &[
            "stage2_grid_accumulate",
            "stage2_histogram",
            "stage2_topofit",
            "stage2_topofit_row_invariant",
            "stage2_topofit_coh_row_invariant",
        ],
        (4, StageScope::Patch) => &["stage4_edge_stats"],
        (6, StageScope::Merged) => &["stage6_graph_unwrap"],
        (7, StageScope::Merged) => &["stage7_scla"],
        (8, StageScope::Merged) => &["stage8_edge_noise"],
        _ => &[],
    }
}

fn runtime_config_text(runtime: &RuntimeOptions) -> String {
    format!(
        "runtime:\n  backend: {}\n  stage2_kernel_backend: {}\n  io_workers: {}\n  cpu_workers: {}\n",
        runtime.backend, runtime.stage2_kernel_backend, runtime.io_workers, runtime.cpu_workers
    )
}

fn run_cli_bridge_command(
    request: &RunRequest,
    options: &CliBridgeOptions,
    config_path: &Path,
) -> Result<std::process::Output, CoreError> {
    let Some((program, args)) = options.command.split_first() else {
        return Err(CoreError::StartExecution {
            program: "<empty>".to_string(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "empty execution command",
            ),
        });
    };
    let mut command = Command::new(program);
    command
        .args(args)
        .arg("--config")
        .arg(config_path)
        .arg("run")
        .arg("--dataset")
        .arg(&request.dataset_root)
        .arg("--start-step")
        .arg(request.start_step.to_string())
        .arg("--end-step")
        .arg(request.end_step.to_string())
        .arg("--io-workers")
        .arg(options.runtime.io_workers.to_string())
        .arg("--cpu-workers")
        .arg(options.runtime.cpu_workers.to_string());
    if request.dry_run {
        command.arg("--dry-run");
    }
    command
        .output()
        .map_err(|source| CoreError::StartExecution {
            program: program.clone(),
            source,
        })
}

fn temp_runtime_config_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "pystamps-core-runtime-{}-{nanos}.yaml",
        std::process::id()
    ))
}

fn plan_single_scope(
    stage_id: u8,
    scope: StageScope,
    target_dir: &Path,
    target_name: &str,
    dry_run: bool,
) -> StageResult {
    let Some(expected) = expected_stage_artifact(stage_id, scope) else {
        return StageResult::new(
            stage_id,
            scope,
            target_name,
            StageStatus::Skipped,
            "No expected artifact mapping",
            None,
        );
    };

    if expected_bundle(stage_id, scope)
        .iter()
        .all(|filename| target_dir.join(filename).exists())
    {
        return StageResult::new(
            stage_id,
            scope,
            target_name,
            StageStatus::SkippedExisting,
            format!("{expected} present"),
            None,
        );
    }

    let status = if dry_run {
        StageStatus::Planned
    } else {
        StageStatus::PendingExecution
    };
    let verb = if dry_run {
        "Would produce"
    } else {
        "Will produce"
    };
    StageResult::new(
        stage_id,
        scope,
        target_name,
        status,
        format!("{verb} {expected}"),
        None,
    )
}

fn dataset_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("dataset")
}

pub fn expected_stage_artifact(stage_id: u8, scope: StageScope) -> Option<&'static str> {
    match (stage_id, scope) {
        (1, StageScope::Patch) => Some("ps1.mat"),
        (2, StageScope::Patch) => Some("pm1.mat"),
        (3, StageScope::Patch) => Some("select1.mat"),
        (4, StageScope::Patch) => Some("weed1.mat"),
        (5, StageScope::Patch) => Some("ph2.mat"),
        (5, StageScope::Merged) => Some("ifgstd2.mat"),
        (6, StageScope::Merged) => Some("phuw2.mat"),
        (7, StageScope::Merged) => Some("scla2.mat"),
        (8, StageScope::Merged) => Some("uw_space_time.mat"),
        _ => None,
    }
}

fn expected_bundle(stage_id: u8, scope: StageScope) -> &'static [&'static str] {
    match (stage_id, scope) {
        (1, StageScope::Patch) => &["ps1.mat", "ph1.mat", "bp1.mat", "psver.mat"],
        (2, StageScope::Patch) => &["pm1.mat"],
        (3, StageScope::Patch) => &["select1.mat"],
        (4, StageScope::Patch) => &["weed1.mat"],
        (5, StageScope::Patch) => &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "hgt2.mat",
            "la2.mat",
            "rc2.mat",
            "psver.mat",
        ],
        (5, StageScope::Merged) => &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "hgt2.mat",
            "la2.mat",
            "rc2.mat",
            "psver.mat",
            "ifgstd2.mat",
        ],
        (6, StageScope::Merged) => &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "ifgstd2.mat",
            "phuw2.mat",
            "uw_phaseuw.mat",
            "uw_grid.mat",
            "uw_interp.mat",
        ],
        (7, StageScope::Merged) => &["scla2.mat", "scla_smooth2.mat"],
        (8, StageScope::Merged) => &["mean_v.mat", "uw_space_time.mat"],
        _ => &[],
    }
}

fn stage_target_dir(dataset_root: &Path, result: &StageResult) -> PathBuf {
    match result.scope {
        StageScope::Patch => dataset_root.join(&result.target),
        StageScope::Merged => dataset_root.to_path_buf(),
    }
}

fn stage_input_artifacts(stage_id: u8, scope: StageScope) -> &'static [&'static str] {
    match (stage_id, scope) {
        (1, StageScope::Patch) => &["pscands.1.ij", "pscands.1.ll", "pscands.1.ph"],
        (2, StageScope::Patch) => &["ps1.mat", "ph1.mat", "bp1.mat"],
        (3, StageScope::Patch) => &["ps1.mat", "pm1.mat"],
        (4, StageScope::Patch) => &["ps1.mat", "pm1.mat", "select1.mat"],
        (5, StageScope::Patch) => &[
            "ps1.mat",
            "ph1.mat",
            "pm1.mat",
            "bp1.mat",
            "select1.mat",
            "weed1.mat",
        ],
        (6, StageScope::Merged) => &["ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "ifgstd2.mat"],
        (7, StageScope::Merged) => &["ps2.mat", "phuw2.mat", "ifgstd2.mat"],
        (8, StageScope::Merged) => &[
            "ps2.mat",
            "phuw2.mat",
            "scla2.mat",
            "uw_grid.mat",
            "uw_interp.mat",
        ],
        _ => &[],
    }
}

fn stage_output_artifacts(stage_id: u8, scope: StageScope) -> &'static [&'static str] {
    match (stage_id, scope) {
        (1, StageScope::Patch) => &["ps1.mat", "ph1.mat", "bp1.mat", "psver.mat"],
        (2, StageScope::Patch) => &["pm1.mat"],
        (3, StageScope::Patch) => &["select1.mat"],
        (4, StageScope::Patch) => &["weed1.mat"],
        (5, StageScope::Patch) => &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "hgt2.mat",
            "la2.mat",
            "rc2.mat",
            "psver.mat",
        ],
        (5, StageScope::Merged) => &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "hgt2.mat",
            "la2.mat",
            "rc2.mat",
            "psver.mat",
            "ifgstd2.mat",
        ],
        (6, StageScope::Merged) => &[
            "phuw2.mat",
            "uw_phaseuw.mat",
            "uw_grid.mat",
            "uw_interp.mat",
        ],
        (7, StageScope::Merged) => &["scla2.mat", "scla_smooth2.mat"],
        (8, StageScope::Merged) => &["mean_v.mat", "uw_space_time.mat"],
        _ => &[],
    }
}

fn count_stage_input_artifacts(
    dataset_root: &Path,
    target_dir: &Path,
    stage_id: u8,
    scope: StageScope,
) -> usize {
    if stage_id == 5 && scope == StageScope::Merged {
        return discover_dataset(dataset_root)
            .map(|layout| {
                layout
                    .patches
                    .iter()
                    .map(|patch| {
                        count_existing_artifacts(
                            patch,
                            stage_output_artifacts(5, StageScope::Patch),
                        )
                    })
                    .sum()
            })
            .unwrap_or(0);
    }
    count_existing_artifacts(target_dir, stage_input_artifacts(stage_id, scope))
}

fn count_existing_artifacts(target_dir: &Path, artifacts: &[&str]) -> usize {
    artifacts
        .iter()
        .filter(|filename| target_dir.join(filename).is_file())
        .count()
}

fn rows_processed_for_stage(target_dir: &Path, stage_id: u8, scope: StageScope) -> Option<usize> {
    let artifact = match (stage_id, scope) {
        (1, StageScope::Patch)
        | (2, StageScope::Patch)
        | (3, StageScope::Patch)
        | (4, StageScope::Patch) => "ps1.mat",
        (5, StageScope::Patch)
        | (5, StageScope::Merged)
        | (6, StageScope::Merged)
        | (7, StageScope::Merged)
        | (8, StageScope::Merged) => "ps2.mat",
        _ => return None,
    };
    read_mat_scalar_usize(&target_dir.join(artifact), "n_ps")
}

fn enrich_stage6_grid_telemetry(target_dir: &Path, result: &mut StageResult) {
    if let Ok(grid) = MatData::read(target_dir.join("uw_grid.mat")) {
        result.n_grid_ps = mat_scalar_usize(&grid, "n_ps");
        if let Ok(nzix) = grid.get("nzix") {
            result.n_grid_rows = Some(nzix.rows);
            result.n_grid_cols = Some(nzix.cols);
        }
    }
    if let Ok(interp) = MatData::read(target_dir.join("uw_interp.mat")) {
        result.n_edges = mat_scalar_usize(&interp, "n_edge")
            .or_else(|| interp.get("edgs").ok().map(|edgs| edgs.rows));
    }
}

fn read_mat_scalar_usize(path: &Path, name: &str) -> Option<usize> {
    MatData::read(path)
        .ok()
        .and_then(|mat| mat_scalar_usize(&mat, name))
}

fn mat_scalar_usize(mat: &MatData, name: &str) -> Option<usize> {
    let values = mat.get_f64_matrix(name).ok()?;
    let value = *values.values.first()?;
    if value.is_finite() && value >= 0.0 {
        Some(value.round() as usize)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn current_process_peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb * 1024)
    })
}

#[cfg(not(target_os = "linux"))]
fn current_process_peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_mat::MatFile;
    use std::fs::{self, File};

    #[test]
    fn plans_patch_and_stage5_merged_work() {
        let root = temp_dataset("pystamps-core-plan");
        fs::create_dir(root.join("PATCH_1")).unwrap();
        fs::create_dir(root.join("PATCH_2")).unwrap();

        let results = plan_pipeline(&RunRequest {
            dataset_root: root.clone(),
            start_step: 5,
            end_step: 5,
            dry_run: true,
        })
        .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].scope, StageScope::Patch);
        assert_eq!(results[0].status, StageStatus::Planned);
        assert_eq!(results[2].scope, StageScope::Merged);
        assert_eq!(results[2].details, "Would produce ifgstd2.mat");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discover_dataset_prefers_patch_list_order_and_bounds() {
        let root = temp_dataset("pystamps-core-patch-list");
        fs::create_dir(root.join("PATCH_1")).unwrap();
        fs::create_dir(root.join("PATCH_2")).unwrap();
        fs::create_dir(root.join("PATCH_3")).unwrap();
        fs::write(root.join("patch.list"), "PATCH_2\nPATCH_1\n").unwrap();

        let layout = discover_dataset(&root).unwrap();
        let names = layout
            .patches
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["PATCH_2", "PATCH_1"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_existing_bundle_only_when_all_bundle_files_exist() {
        let root = temp_dataset("pystamps-core-existing");
        let patch = root.join("PATCH_1");
        fs::create_dir(&patch).unwrap();
        File::create(patch.join("ps1.mat")).unwrap();

        let partial = plan_pipeline(&RunRequest {
            dataset_root: root.clone(),
            start_step: 1,
            end_step: 1,
            dry_run: true,
        })
        .unwrap();
        assert_eq!(partial[0].status, StageStatus::Planned);

        for file in ["ph1.mat", "bp1.mat", "psver.mat"] {
            File::create(patch.join(file)).unwrap();
        }
        let complete = plan_pipeline(&RunRequest {
            dataset_root: root.clone(),
            start_step: 1,
            end_step: 1,
            dry_run: true,
        })
        .unwrap();
        assert_eq!(complete[0].status, StageStatus::SkippedExisting);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn coverage_spans_the_full_processing_chain() {
        let coverage = processing_chain_coverage(1, 8).unwrap();
        let stages: Vec<(u8, StageScope)> =
            coverage.iter().map(|row| (row.stage, row.scope)).collect();

        assert_eq!(
            stages,
            vec![
                (1, StageScope::Patch),
                (2, StageScope::Patch),
                (3, StageScope::Patch),
                (4, StageScope::Patch),
                (5, StageScope::Patch),
                (5, StageScope::Merged),
                (6, StageScope::Merged),
                (7, StageScope::Merged),
                (8, StageScope::Merged),
            ]
        );
        assert!(coverage.iter().all(|row| row.rust_driver));
    }

    #[test]
    fn full_chain_coverage_reports_true_native_stage_for_all_required_scopes() {
        let coverage = processing_chain_coverage(1, 8).unwrap();
        assert!(
            coverage.iter().all(|row| row.native_stage),
            "Expected all required scopes to be fully native-stage covered."
        );
        assert!(
            coverage.iter().all(|row| row.parity_certified),
            "Expected all required scopes to be parity certified."
        );
        assert!(
            coverage.iter().all(|row| !row.disabled),
            "Expected no required scopes to be disabled."
        );
        assert!(coverage.iter().all(|row| {
            let modes = row
                .unsupported_modes
                .iter()
                .map(|mode| mode.mode)
                .collect::<Vec<_>>();
            ["python", "matlab", "octave", "bridge"]
                .iter()
                .all(|expected| modes.contains(expected))
        }));
    }

    #[test]
    fn full_native_chain_verification_passes_when_all_stage_ports_exist() {
        verify_full_native_processing_chain(1, 8).unwrap();
    }

    #[test]
    fn verify_full_native_chain_fails_when_any_scope_is_disabled() {
        let disabled = [(3, StageScope::Patch)];
        let err = verify_full_native_processing_chain_with_disabled(1, 8, &disabled).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("stage 3 patch"),
            "expected disabled scope to be reported as blocking verification, got: {message}"
        );
    }

    #[test]
    fn coverage_metadata_reports_disabled_scope_and_native_only_reasons() {
        let disabled = [(3, StageScope::Patch)];
        let coverage = processing_chain_coverage_with_disabled(3, 3, &disabled).unwrap();
        assert_eq!(coverage.len(), 1);
        let row = &coverage[0];

        assert!(!row.native_stage);
        assert!(row.parity_certified);
        assert!(row.disabled);
        assert_eq!(
            row.disabled_reason.as_deref(),
            Some("disabled by PYSTAMPS_DISABLE_NATIVE_STAGES")
        );
        assert_eq!(
            row.not_native_reason.as_deref(),
            Some("disabled by PYSTAMPS_DISABLE_NATIVE_STAGES")
        );
        assert!(row.not_parity_certified_reason.is_none());
        assert!(row.unsupported_modes.iter().any(|mode| {
            mode.mode == "python" && mode.reason.contains("Rust-owned stage execution")
        }));
        assert!(row
            .unsupported_modes
            .iter()
            .any(|mode| mode.mode == "bridge" && mode.reason.contains("Python CLI bridge")));
    }

    #[test]
    fn processing_chain_report_includes_stage_5_merged_scope() {
        let coverage = processing_chain_coverage(1, 8).unwrap();
        let stage5_merged = coverage
            .iter()
            .find(|row| row.stage == 5 && row.scope == StageScope::Merged)
            .expect("expected stage 5 merged scope");
        assert_eq!(stage5_merged.target, "dataset root");
        assert_eq!(stage5_merged.native_stage, true);
    }

    #[test]
    fn cli_bridge_runs_the_selected_chain_and_parses_json_results() {
        let root = temp_dataset("pystamps-core-cli-bridge");
        fs::create_dir(root.join("PATCH_1")).unwrap();

        let execution = execute_pipeline_cli_bridge(
            &RunRequest {
                dataset_root: root.clone(),
                start_step: 1,
                end_step: 1,
                dry_run: false,
            },
            &CliBridgeOptions {
                command: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf '[{\"stage\":1,\"scope\":\"patch\",\"target\":\"PATCH_1\",\"status\":\"completed\",\"details\":\"ok\",\"duration_sec\":0.1}]'".to_string(),
                ],
                runtime: RuntimeOptions::default(),
                native_only: false,
            },
        )
        .unwrap();

        assert_eq!(execution.exit_code, Some(0));
        assert_eq!(execution.results.len(), 1);
        assert_eq!(execution.results[0].status, StageStatus::Completed);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cli_bridge_is_rejected_when_native_only_is_requested() {
        let err = execute_pipeline_cli_bridge(
            &RunRequest {
                dataset_root: PathBuf::from("/unused"),
                start_step: 1,
                end_step: 1,
                dry_run: false,
            },
            &CliBridgeOptions {
                native_only: true,
                ..CliBridgeOptions::default()
            },
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(
            message.contains("native-only mode forbids CLI bridge execution"),
            "expected native-only bridge rejection, got: {message}"
        );
    }

    #[test]
    fn stage6_telemetry_reports_grid_shape_and_edges() {
        let root = temp_dataset("pystamps-core-telemetry");
        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", 10.0).unwrap();
        ps2.write().unwrap();

        let mut uw_grid = MatFile::new(root.join("uw_grid.mat"));
        uw_grid.add_f64_scalar("n_ps", 4.0).unwrap();
        uw_grid
            .add_u8_matrix("nzix", 2, 3, vec![1, 0, 1, 0, 1, 1])
            .unwrap();
        uw_grid.write().unwrap();

        let mut uw_interp = MatFile::new(root.join("uw_interp.mat"));
        uw_interp.add_f64_scalar("n_edge", 5.0).unwrap();
        uw_interp
            .add_f64_matrix(
                "edgs",
                5,
                3,
                vec![
                    1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 1.0, 5.0, 1.0, 3.0,
                ],
            )
            .unwrap();
        uw_interp.write().unwrap();

        let mut result = StageResult::new(
            6,
            StageScope::Merged,
            "dataset",
            StageStatus::Completed,
            "ok",
            Some(1.25),
        );

        enrich_stage_result_telemetry(&root, &mut result);

        assert_eq!(result.rows_processed, Some(10));
        assert_eq!(result.n_grid_ps, Some(4));
        assert_eq!(result.n_grid_rows, Some(2));
        assert_eq!(result.n_grid_cols, Some(3));
        assert_eq!(result.n_edges, Some(5));
        fs::remove_dir_all(root).unwrap();
    }

    fn temp_dataset(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir(&root).unwrap();
        root
    }
}

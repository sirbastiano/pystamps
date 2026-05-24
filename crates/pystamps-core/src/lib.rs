use serde::{Deserialize, Serialize};
use std::fs;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub mod mat_v5;
pub mod native_stage1;

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
}

impl Default for CliBridgeOptions {
    fn default() -> Self {
        Self {
            command: vec!["uv".to_string(), "run".to_string(), "pystamps".to_string()],
            runtime: RuntimeOptions::default(),
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
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StageCoverage {
    pub stage: u8,
    pub scope: StageScope,
    pub target: String,
    pub rust_driver: bool,
    pub native_stage: bool,
    pub native_kernels: &'static [&'static str],
    pub details: &'static str,
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
    patches.sort_by(|left, right| {
        let left_name = left.file_name().unwrap_or_default();
        let right_name = right.file_name().unwrap_or_default();
        left_name.cmp(right_name)
    });

    Ok(DatasetLayout { root, patches })
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

pub fn processing_chain_coverage(start_step: u8, end_step: u8) -> Result<Vec<StageCoverage>, CoreError> {
    validate_stage_range(start_step, end_step)?;
    let mut coverage = Vec::new();
    for stage in selected_stages(start_step, end_step) {
        match stage.scope {
            StageScope::Patch => {
                coverage.push(stage_coverage(stage.stage_id, StageScope::Patch, "PATCH_*"));
                if stage.stage_id == 5 {
                    coverage.push(stage_coverage(5, StageScope::Merged, "dataset root"));
                }
            }
            StageScope::Merged => coverage.push(stage_coverage(stage.stage_id, StageScope::Merged, "dataset root")),
        }
    }
    Ok(coverage)
}

pub fn verify_full_native_processing_chain(start_step: u8, end_step: u8) -> Result<(), CoreError> {
    let missing: Vec<String> = processing_chain_coverage(start_step, end_step)?
        .into_iter()
        .filter(|coverage| !coverage.native_stage)
        .map(|coverage| format!("stage {} {} ({})", coverage.stage, coverage.scope, coverage.target))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(CoreError::IncompleteNativeChain(missing.join(", ")))
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

fn stage_coverage(stage_id: u8, scope: StageScope, target: &'static str) -> StageCoverage {
    let scope_name = scope.to_string();
    let native_stage = pystamps_stages::native_stage_is_parity_certified(stage_id, &scope_name);
    StageCoverage {
        stage: stage_id,
        scope,
        target: target.to_string(),
        rust_driver: true,
        native_stage,
        native_kernels: native_kernel_acceleration(stage_id, scope),
        details: stage_coverage_details(stage_id, scope),
    }
}

fn stage_coverage_details(stage_id: u8, scope: StageScope) -> &'static str {
    let scope_name = scope.to_string();
    pystamps_stages::native_stage_details(stage_id, &scope_name)
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
            source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty execution command"),
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
    command.output().map_err(|source| CoreError::StartExecution {
        program: program.clone(),
        source,
    })
}

fn temp_runtime_config_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!("pystamps-core-runtime-{}-{nanos}.yaml", std::process::id()))
}

fn plan_single_scope(
    stage_id: u8,
    scope: StageScope,
    target_dir: &Path,
    target_name: &str,
    dry_run: bool,
) -> StageResult {
    let Some(expected) = expected_stage_artifact(stage_id, scope) else {
        return StageResult {
            stage: stage_id,
            scope,
            target: target_name.to_string(),
            status: StageStatus::Skipped,
            details: "No expected artifact mapping".to_string(),
            duration_sec: None,
        };
    };

    if expected_bundle(stage_id, scope)
        .iter()
        .all(|filename| target_dir.join(filename).exists())
    {
        return StageResult {
            stage: stage_id,
            scope,
            target: target_name.to_string(),
            status: StageStatus::SkippedExisting,
            details: format!("{expected} present"),
            duration_sec: None,
        };
    }

    let status = if dry_run {
        StageStatus::Planned
    } else {
        StageStatus::PendingExecution
    };
    let verb = if dry_run { "Would produce" } else { "Will produce" };
    StageResult {
        stage: stage_id,
        scope,
        target: target_name.to_string(),
        status,
        details: format!("{verb} {expected}"),
        duration_sec: None,
    }
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
            "ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "hgt2.mat", "la2.mat", "rc2.mat",
            "psver.mat",
        ],
        (5, StageScope::Merged) => &[
            "ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "hgt2.mat", "la2.mat", "rc2.mat",
            "psver.mat", "ifgstd2.mat",
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let stages: Vec<(u8, StageScope)> = coverage.iter().map(|row| (row.stage, row.scope)).collect();

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
    fn full_native_chain_verification_fails_until_stage_ports_exist() {
        let err = verify_full_native_processing_chain(1, 8).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("stage 2 patch"));
        assert!(message.contains("stage 8 merged"));
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
            },
        )
        .unwrap();

        assert_eq!(execution.exit_code, Some(0));
        assert_eq!(execution.results.len(), 1);
        assert_eq!(execution.results[0].status, StageStatus::Completed);
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

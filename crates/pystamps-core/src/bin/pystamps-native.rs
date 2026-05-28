use pystamps_core::native_stage1::run_stage1_native;
use pystamps_core::native_stage2::{run_stage2_native, run_stage2_native_with_threads};
use pystamps_core::native_stage3::run_stage3_native;
use pystamps_core::native_stage4::run_stage4_native;
use pystamps_core::native_stage5::{run_stage5_merge_native, run_stage5_patch_native};
use pystamps_core::native_stage6::run_stage6_native;
use pystamps_core::native_stage7::run_stage7_native;
use pystamps_core::native_stage8::run_stage8_native;
use pystamps_core::{
    enrich_stage_result_telemetry, plan_pipeline, processing_chain_coverage, RunRequest,
    StageResult, StageScope, StageStatus,
};
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RunTimeConfig {
    backend: String,
    stage2_kernel_backend: String,
    io_workers: u16,
    cpu_workers: u16,
    stage2_native_threads: u16,
    native_only: bool,
}

impl Default for RunTimeConfig {
    fn default() -> Self {
        Self {
            backend: "auto".to_string(),
            stage2_kernel_backend: "auto".to_string(),
            io_workers: 8,
            cpu_workers: 0,
            stage2_native_threads: 0,
            native_only: false,
        }
    }
}

impl RunTimeConfig {
    fn stage2_thread_count(&self) -> usize {
        self.stage2_thread_count_with_available(default_processing_threads())
    }

    fn stage2_thread_count_with_available(&self, available: usize) -> usize {
        if self.stage2_native_threads > 0 {
            self.stage2_native_threads as usize
        } else if self.cpu_workers > 0 {
            self.cpu_workers as usize
        } else {
            available.max(1)
        }
    }
}

fn default_processing_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            usage();
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let Some((command, rest)) = args.split_first() else {
        return Err("missing subcommand".to_string());
    };
    match command.as_str() {
        "run" => {
            let (request, runtime) = parse_run_args(rest)?;
            execute_run_pipeline(&request, &runtime)
        }
        "coverage" => {
            let (start_step, end_step) = parse_coverage_args(rest)?;
            let coverage =
                processing_chain_coverage(start_step, end_step).map_err(|err| err.to_string())?;
            let json = serde_json::to_string_pretty(&coverage).map_err(|err| err.to_string())?;
            println!("{json}");
            Ok(())
        }
        "stage" => run_stage(rest),
        "stage1" => {
            let patch = parse_patch_arg(rest)?;
            let details = run_stage1_native(patch).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage2" => {
            let patch = parse_patch_arg(rest)?;
            let details = run_stage2_native(patch).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage3" => {
            let patch = parse_patch_arg(rest)?;
            let details = run_stage3_native(patch).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage4" => {
            let patch = parse_patch_arg(rest)?;
            let details = run_stage4_native(patch).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage5" => {
            let patch = parse_patch_arg(rest)?;
            let details = run_stage5_patch_native(patch).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage5-merge" => {
            let dataset = parse_dataset_arg(rest)?;
            let details = run_stage5_merge_native(dataset).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage6" => {
            let dataset = parse_dataset_arg(rest)?;
            let details = run_stage6_native(dataset).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage7" => {
            let dataset = parse_dataset_arg(rest)?;
            let details = run_stage7_native(dataset).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        "stage8" => {
            let dataset = parse_dataset_arg(rest)?;
            let details = run_stage8_native(dataset).map_err(|err| err.to_string())?;
            println!("{details}");
            Ok(())
        }
        _ => Err(format!("unknown subcommand '{command}'")),
    }
}

fn execute_run_pipeline(request: &RunRequest, runtime: &RunTimeConfig) -> Result<(), String> {
    let mut results = plan_pipeline(request).map_err(|err| err.to_string())?;
    if request.dry_run {
        let json = serde_json::to_string_pretty(&results).map_err(|err| err.to_string())?;
        println!("{json}");
        return Ok(());
    }

    for result in &mut results {
        if result.status != StageStatus::PendingExecution {
            continue;
        }

        let start = Instant::now();
        match run_native_stage(request.dataset_root.as_path(), result, runtime) {
            Ok(executed) => {
                result.status = StageStatus::Completed;
                result.details = executed;
                result.duration_sec = Some(start.elapsed().as_secs_f64());
            }
            Err(err) => {
                result.status = StageStatus::Failed;
                result.details = err;
                result.duration_sec = Some(start.elapsed().as_secs_f64());
            }
        }
    }

    for result in &mut results {
        enrich_stage_result_telemetry(request.dataset_root.as_path(), result);
    }

    let json = serde_json::to_string_pretty(&results).map_err(|err| err.to_string())?;
    println!("{json}");
    Ok(())
}

fn run_native_stage(
    dataset_root: &Path,
    result: &StageResult,
    runtime: &RunTimeConfig,
) -> Result<String, String> {
    let target = match result.scope {
        StageScope::Patch => dataset_root.join(&result.target),
        StageScope::Merged => dataset_root.to_path_buf(),
    };
    match (result.stage, result.scope) {
        (1, StageScope::Patch) => run_stage1_native(target).map_err(|err| err.to_string()),
        (2, StageScope::Patch) => {
            run_stage2_native_with_threads(target, runtime.stage2_thread_count())
                .map_err(|err| err.to_string())
        }
        (3, StageScope::Patch) => run_stage3_native(target).map_err(|err| err.to_string()),
        (4, StageScope::Patch) => run_stage4_native(target).map_err(|err| err.to_string()),
        (5, StageScope::Patch) => run_stage5_patch_native(target).map_err(|err| err.to_string()),
        (5, StageScope::Merged) => run_stage5_merge_native(target).map_err(|err| err.to_string()),
        (6, StageScope::Merged) => run_stage6_native(target).map_err(|err| err.to_string()),
        (7, StageScope::Merged) => run_stage7_native(target).map_err(|err| err.to_string()),
        (8, StageScope::Merged) => run_stage8_native(target).map_err(|err| err.to_string()),
        _ => Err(format!(
            "unsupported stage {} for scope {}",
            result.stage, result.scope
        )),
    }
}

fn parse_run_args(args: &[String]) -> Result<(RunRequest, RunTimeConfig), String> {
    let mut request = RunRequest {
        dataset_root: PathBuf::new(),
        start_step: 1,
        end_step: 8,
        dry_run: false,
    };
    let mut runtime = RunTimeConfig::default();
    let mut ix = 0;
    while ix < args.len() {
        match args[ix].as_str() {
            "--dataset" => {
                let value = args
                    .get(ix + 1)
                    .ok_or_else(|| "expected --dataset PATH".to_string())?;
                request.dataset_root = PathBuf::from(value);
                ix += 2;
            }
            "--start-step" => {
                request.start_step = parse_step_value(args, ix, "--start-step")?;
                ix += 2;
            }
            "--end-step" => {
                request.end_step = parse_step_value(args, ix, "--end-step")?;
                ix += 2;
            }
            "--dry-run" => {
                request.dry_run = true;
                ix += 1;
            }
            "--native-only" => {
                runtime.native_only = true;
                ix += 1;
            }
            "--backend" => {
                runtime.backend = parse_backend_arg(args, ix, "--backend")?;
                ix += 2;
            }
            "--stage2-kernel-backend" => {
                runtime.stage2_kernel_backend =
                    parse_kernel_backend_arg(args, ix, "--stage2-kernel-backend")?;
                ix += 2;
            }
            "--io-workers" => {
                runtime.io_workers = parse_u16_value(args, ix, "--io-workers")?;
                ix += 2;
            }
            "--cpu-workers" => {
                runtime.cpu_workers = parse_u16_value(args, ix, "--cpu-workers")?;
                ix += 2;
            }
            "--stage2-native-threads" => {
                runtime.stage2_native_threads =
                    parse_u16_value(args, ix, "--stage2-native-threads")?;
                ix += 2;
            }
            other => return Err(format!("unexpected run argument '{other}'")),
        }
    }

    if request.dataset_root.as_os_str().is_empty() {
        return Err("expected --dataset PATH".to_string());
    }

    let _ = validate_runtime_config(&runtime)?;
    Ok((request, runtime))
}

fn validate_runtime_config(config: &RunTimeConfig) -> Result<(), String> {
    let backend = config.backend.as_str();
    if !matches!(backend, "auto" | "threads" | "processes" | "gpu" | "native") {
        return Err(format!("unsupported runtime backend '{backend}'"));
    }
    let kernel_backend = config.stage2_kernel_backend.as_str();
    if !matches!(kernel_backend, "auto" | "python" | "native") {
        return Err(format!(
            "unsupported stage-2 kernel backend '{kernel_backend}'"
        ));
    }
    if config.stage2_native_threads > 0 && matches!(kernel_backend, "python") {
        return Err(
            "stage2-native-threads requires stage2 kernel backend auto or native".to_string(),
        );
    }
    if config.native_only {
        if backend != "native" {
            return Err(format!(
                "native-only mode requires --backend native; got '{backend}'"
            ));
        }
        if kernel_backend != "native" {
            return Err(format!(
                "native-only mode requires --stage2-kernel-backend native; got '{kernel_backend}'"
            ));
        }
    }
    Ok(())
}

fn parse_backend_arg(args: &[String], ix: usize, name: &str) -> Result<String, String> {
    let value = args
        .get(ix + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    let value = value.to_lowercase();
    match value.as_str() {
        "auto" => Ok("auto".to_string()),
        "threads" | "thread" | "io" => Ok("threads".to_string()),
        "processes" | "process" | "cpu" => Ok("processes".to_string()),
        "gpu" => Ok("gpu".to_string()),
        "native" => Ok("native".to_string()),
        unsupported => Err(format!("unsupported runtime backend '{unsupported}'")),
    }
}

fn parse_kernel_backend_arg(args: &[String], ix: usize, name: &str) -> Result<String, String> {
    let value = args
        .get(ix + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    let value = value.to_lowercase();
    match value.as_str() {
        "auto" => Ok("auto".to_string()),
        "native" => Ok("native".to_string()),
        "python" | "cpu" => Ok("python".to_string()),
        unsupported => Err(format!(
            "unsupported stage-2 kernel backend '{unsupported}'"
        )),
    }
}

fn parse_u16_value(args: &[String], ix: usize, name: &str) -> Result<u16, String> {
    let value = args
        .get(ix + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    value
        .parse::<u16>()
        .map_err(|err| format!("invalid {name} value '{value}': {err}"))
}

fn run_stage(args: &[String]) -> Result<(), String> {
    let Some((stage, rest)) = args.split_first() else {
        return Err("expected stage number".to_string());
    };
    let stage = stage
        .parse::<u8>()
        .map_err(|err| format!("invalid stage number '{stage}': {err}"))?;
    if !matches!(stage, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8) {
        return Err(format!("stage {stage} is not native-executable yet"));
    }

    let details = match stage {
        1 => run_stage1_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        2 => run_stage2_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        3 => run_stage3_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        4 => run_stage4_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        5 if rest.len() == 2 && rest[0] == "--dataset" => {
            run_stage5_merge_native(parse_dataset_arg(rest)?).map_err(|err| err.to_string())?
        }
        5 => run_stage5_patch_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        6 => run_stage6_native(parse_dataset_arg(rest)?).map_err(|err| err.to_string())?,
        7 => run_stage7_native(parse_dataset_arg(rest)?).map_err(|err| err.to_string())?,
        8 => run_stage8_native(parse_dataset_arg(rest)?).map_err(|err| err.to_string())?,
        _ => unreachable!("stage was validated above"),
    };
    println!("{details}");
    Ok(())
}

fn parse_coverage_args(args: &[String]) -> Result<(u8, u8), String> {
    let mut start_step = 1;
    let mut end_step = 8;
    let mut ix = 0;
    while ix < args.len() {
        match args[ix].as_str() {
            "--start-step" => {
                start_step = parse_step_value(args, ix, "--start-step")?;
                ix += 2;
            }
            "--end-step" => {
                end_step = parse_step_value(args, ix, "--end-step")?;
                ix += 2;
            }
            other => return Err(format!("unexpected coverage argument '{other}'")),
        }
    }
    Ok((start_step, end_step))
}

fn parse_step_value(args: &[String], ix: usize, name: &str) -> Result<u8, String> {
    let value = args
        .get(ix + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    value
        .parse::<u8>()
        .map_err(|err| format!("invalid {name} value '{value}': {err}"))
}

fn parse_patch_arg(args: &[String]) -> Result<PathBuf, String> {
    if args.len() == 2 && args[0] == "--patch" {
        Ok(PathBuf::from(&args[1]))
    } else {
        Err("expected --patch PATH".to_string())
    }
}

fn parse_dataset_arg(args: &[String]) -> Result<PathBuf, String> {
    if args.len() == 2 && args[0] == "--dataset" {
        Ok(PathBuf::from(&args[1]))
    } else {
        Err("expected --dataset PATH".to_string())
    }
}

fn usage() {
    eprintln!(
        "Usage:
  pystamps-native run --dataset PATH [--start-step N] [--end-step N] [--dry-run] [--native-only]
    [--backend auto|threads|processes|gpu|native]
    [--stage2-kernel-backend auto|python|native]
    [--io-workers N] [--cpu-workers N] [--stage2-native-threads N]
  pystamps-native coverage [--start-step N] [--end-step N]
  pystamps-native stage 1 --patch PATH
  pystamps-native stage 2 --patch PATH
  pystamps-native stage 3 --patch PATH
  pystamps-native stage 4 --patch PATH
  pystamps-native stage 5 --patch PATH
  pystamps-native stage 5 --dataset PATH
  pystamps-native stage 6 --dataset PATH
  pystamps-native stage 7 --dataset PATH
  pystamps-native stage1 --patch PATH
  pystamps-native stage2 --patch PATH
  pystamps-native stage3 --patch PATH
  pystamps-native stage4 --patch PATH
  pystamps-native stage5 --patch PATH
  pystamps-native stage5-merge --dataset PATH
  pystamps-native stage6 --dataset PATH
  pystamps-native stage7 --dataset PATH
  pystamps-native stage8 --dataset PATH"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(cpu_workers: u16, stage2_native_threads: u16) -> RunTimeConfig {
        RunTimeConfig {
            cpu_workers,
            stage2_native_threads,
            ..RunTimeConfig::default()
        }
    }

    #[test]
    fn stage2_auto_threads_use_available_parallelism() {
        assert_eq!(runtime(0, 0).stage2_thread_count_with_available(16), 16);
    }

    #[test]
    fn stage2_auto_threads_honor_cpu_worker_cap() {
        assert_eq!(runtime(4, 0).stage2_thread_count_with_available(16), 4);
    }

    #[test]
    fn stage2_explicit_native_threads_override_cpu_workers() {
        assert_eq!(runtime(4, 6).stage2_thread_count_with_available(16), 6);
    }
}

use pystamps_core::native_stage1::run_stage1_native;
use pystamps_core::native_stage2::run_stage2_native;
use pystamps_core::native_stage3::run_stage3_native;
use pystamps_core::native_stage5::{run_stage5_merge_native, run_stage5_patch_native};
use pystamps_core::native_stage7::run_stage7_native;
use pystamps_core::native_stage8::run_stage8_native;
use pystamps_core::processing_chain_coverage;
use std::path::PathBuf;
use std::process::ExitCode;

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
        "coverage" => {
            let (start_step, end_step) = parse_coverage_args(rest)?;
            let coverage = processing_chain_coverage(start_step, end_step).map_err(|err| err.to_string())?;
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

fn run_stage(args: &[String]) -> Result<(), String> {
    let Some((stage, rest)) = args.split_first() else {
        return Err("expected stage number".to_string());
    };
    let stage = stage
        .parse::<u8>()
        .map_err(|err| format!("invalid stage number '{stage}': {err}"))?;
    if !matches!(stage, 1 | 2 | 3 | 5 | 7 | 8) {
        return Err(format!("stage {stage} is not native-executable yet"));
    }

    let details = match stage {
        1 => run_stage1_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        2 => run_stage2_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        3 => run_stage3_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
        5 if rest.len() == 2 && rest[0] == "--dataset" => {
            run_stage5_merge_native(parse_dataset_arg(rest)?).map_err(|err| err.to_string())?
        }
        5 => run_stage5_patch_native(parse_patch_arg(rest)?).map_err(|err| err.to_string())?,
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
  pystamps-native coverage [--start-step N] [--end-step N]
  pystamps-native stage 1 --patch PATH
  pystamps-native stage 2 --patch PATH
  pystamps-native stage 3 --patch PATH
  pystamps-native stage 5 --patch PATH
  pystamps-native stage 5 --dataset PATH
  pystamps-native stage 7 --dataset PATH
  pystamps-native stage1 --patch PATH
  pystamps-native stage2 --patch PATH
  pystamps-native stage3 --patch PATH
  pystamps-native stage5 --patch PATH
  pystamps-native stage5-merge --dataset PATH
  pystamps-native stage7 --dataset PATH
  pystamps-native stage8 --dataset PATH"
    );
}

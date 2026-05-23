use std::process::Command;

#[test]
fn no_arguments_prints_usage_and_exits_nonzero() {
    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage:"));
    assert!(stderr.contains("missing subcommand"));
}

#[test]
fn unknown_subcommand_prints_clear_error_and_exits_nonzero() {
    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("bogus")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown subcommand 'bogus'"));
    assert!(stderr.contains("Usage:"));
}

#[test]
fn coverage_subcommand_reports_stage_matrix() {
    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("coverage")
        .arg("--start-step")
        .arg("1")
        .arg("--end-step")
        .arg("1")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"stage\": 1"));
    assert!(stdout.contains("\"rust_driver\": true"));
    assert!(stdout.contains("\"native_stage\": false"));
}

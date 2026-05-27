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
    assert!(stdout.contains("\"native_stage\": true"));
}

#[test]
fn run_dry_run_plans_stages_for_existing_patch_tree() {
    let root = std::env::temp_dir().join(format!(
        "pystamps-core-native-run-dry-run-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("PATCH_1")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("run")
        .arg("--dataset")
        .arg(&root)
        .arg("--start-step")
        .arg("1")
        .arg("--end-step")
        .arg("2")
        .arg("--dry-run")
        .output()
        .unwrap();

    let _ = std::fs::remove_dir_all(&root);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"stage\": 1"));
    assert!(stdout.contains("\"status\": \"planned\""));
    assert!(stdout.contains("PATCH_1"));
}

#[test]
fn run_rejects_unsupported_runtime_backend() {
    let root = std::env::temp_dir().join(format!(
        "pystamps-core-native-run-error-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir(&root);

    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("run")
        .arg("--dataset")
        .arg(&root)
        .arg("--backend")
        .arg("bogus")
        .output()
        .unwrap();

    let _ = std::fs::remove_dir_all(&root);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsupported runtime backend"));
}

#[test]
fn run_native_only_rejects_non_native_runtime_backend() {
    let root = std::env::temp_dir().join(format!(
        "pystamps-core-native-only-backend-error-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("PATCH_1")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("run")
        .arg("--dataset")
        .arg(&root)
        .arg("--native-only")
        .arg("--backend")
        .arg("auto")
        .arg("--stage2-kernel-backend")
        .arg("native")
        .arg("--dry-run")
        .output()
        .unwrap();

    let _ = std::fs::remove_dir_all(&root);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("native-only mode requires --backend native"));
}

#[test]
fn run_native_only_rejects_python_stage2_kernel_backend() {
    let root = std::env::temp_dir().join(format!(
        "pystamps-core-native-only-stage2-error-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("PATCH_1")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pystamps-native"))
        .arg("run")
        .arg("--dataset")
        .arg(&root)
        .arg("--native-only")
        .arg("--backend")
        .arg("native")
        .arg("--stage2-kernel-backend")
        .arg("python")
        .arg("--dry-run")
        .output()
        .unwrap();

    let _ = std::fs::remove_dir_all(&root);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("native-only mode requires --stage2-kernel-backend native"));
}

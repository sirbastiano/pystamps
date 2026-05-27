use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParityError {
    #[error("invalid fixture name '{0}'")]
    InvalidFixtureName(String),
    #[error("fixture source does not exist: {0}")]
    MissingFixtureSource(PathBuf),
    #[error("fixture source is not a directory: {0}")]
    FixtureSourceNotDirectory(PathBuf),
    #[error("unable to remove existing fixture copy {path}: {source}")]
    RemoveFixtureCopy {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to create fixture directory {path}: {source}")]
    CreateFixtureDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to read fixture directory {path}: {source}")]
    ReadFixtureDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to copy fixture file {source_path} to {dest_path}: {source}")]
    CopyFixtureFile {
        source_path: PathBuf,
        dest_path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to read MAT artifact {path}: {source}")]
    ReadMat {
        path: PathBuf,
        source: pystamps_mat::MatError,
    },
    #[error("unable to write parity report {path}: {source}")]
    WriteReport {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to serialize parity report: {0}")]
    SerializeReport(serde_json::Error),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParityTolerance {
    pub rtol: f64,
    pub atol: f64,
    pub wrap_equivalence: bool,
    pub wrap_period: f64,
    pub wrap_keys: Vec<String>,
}

impl Default for ParityTolerance {
    fn default() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-7,
            wrap_equivalence: true,
            wrap_period: 2.0 * std::f64::consts::PI,
            wrap_keys: vec![
                "ph_uw".to_string(),
                "ph".to_string(),
                "dph_noise".to_string(),
                "dph_space_uw".to_string(),
            ],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureRunCopies {
    pub fixture: String,
    pub python_root: PathBuf,
    pub rust_root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactComparisonSpec {
    pub artifact: PathBuf,
    pub variables: Vec<String>,
}

impl ArtifactComparisonSpec {
    pub fn new(
        artifact: impl Into<PathBuf>,
        variables: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            artifact: artifact.into(),
            variables: variables.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ParityComparison {
    pub stage: u8,
    pub scope: String,
    pub fixture: String,
    pub artifact: String,
    pub variable: String,
    pub ok: bool,
    pub rtol: f64,
    pub atol: f64,
    pub message: String,
}

impl ParityComparison {
    pub fn pass(
        stage: u8,
        scope: impl Into<String>,
        fixture: impl Into<String>,
        artifact: impl Into<String>,
        variable: impl Into<String>,
        rtol: f64,
        atol: f64,
    ) -> Self {
        Self {
            stage,
            scope: scope.into(),
            fixture: fixture.into(),
            artifact: artifact.into(),
            variable: variable.into(),
            ok: true,
            rtol,
            atol,
            message: "ok".to_string(),
        }
    }

    pub fn fail(
        stage: u8,
        scope: impl Into<String>,
        fixture: impl Into<String>,
        artifact: impl Into<String>,
        variable: impl Into<String>,
        rtol: f64,
        atol: f64,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            scope: scope.into(),
            fixture: fixture.into(),
            artifact: artifact.into(),
            variable: variable.into(),
            ok: false,
            rtol,
            atol,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct ParityRunSummary {
    pub comparisons: Vec<ParityComparison>,
}

impl ParityRunSummary {
    pub fn all_ok(&self) -> bool {
        self.comparisons.iter().all(|comparison| comparison.ok)
    }

    pub fn failures(&self) -> impl Iterator<Item = &ParityComparison> {
        self.comparisons.iter().filter(|comparison| !comparison.ok)
    }

    pub fn to_json_pretty(&self) -> Result<String, ParityError> {
        serde_json::to_string_pretty(self).map_err(ParityError::SerializeReport)
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), ParityError> {
        let path = path.as_ref();
        let json = self.to_json_pretty()?;
        fs::write(path, json).map_err(|source| ParityError::WriteReport {
            path: path.to_path_buf(),
            source,
        })
    }
}

pub fn create_fixture_run_copies(
    fixture_source: impl AsRef<Path>,
    work_root: impl AsRef<Path>,
    fixture: impl Into<String>,
) -> Result<FixtureRunCopies, ParityError> {
    let fixture = fixture.into();
    validate_fixture_name(&fixture)?;
    let fixture_source = fixture_source.as_ref();
    if !fixture_source.exists() {
        return Err(ParityError::MissingFixtureSource(
            fixture_source.to_path_buf(),
        ));
    }
    if !fixture_source.is_dir() {
        return Err(ParityError::FixtureSourceNotDirectory(
            fixture_source.to_path_buf(),
        ));
    }

    let work_root = work_root.as_ref();
    fs::create_dir_all(work_root).map_err(|source| ParityError::CreateFixtureDirectory {
        path: work_root.to_path_buf(),
        source,
    })?;

    let python_root = work_root.join(format!("{fixture}-python"));
    let rust_root = work_root.join(format!("{fixture}-rust"));
    for dest in [&python_root, &rust_root] {
        if dest.exists() {
            fs::remove_dir_all(dest).map_err(|source| ParityError::RemoveFixtureCopy {
                path: dest.clone(),
                source,
            })?;
        }
        copy_dir_recursive(fixture_source, dest)?;
    }

    Ok(FixtureRunCopies {
        fixture,
        python_root,
        rust_root,
    })
}

pub fn run_fixture_parity<PyRun, RustRun>(
    fixture_source: impl AsRef<Path>,
    work_root: impl AsRef<Path>,
    fixture: impl Into<String>,
    stage: u8,
    scope: impl Into<String>,
    specs: &[ArtifactComparisonSpec],
    tolerance: &ParityTolerance,
    python_run: PyRun,
    rust_run: RustRun,
) -> Result<ParityRunSummary, ParityError>
where
    PyRun: FnOnce(&Path) -> Result<(), ParityError>,
    RustRun: FnOnce(&Path) -> Result<(), ParityError>,
{
    let scope = scope.into();
    let copies = create_fixture_run_copies(fixture_source, work_root, fixture)?;
    python_run(&copies.python_root)?;
    rust_run(&copies.rust_root)?;
    compare_fixture_artifacts(
        stage,
        scope,
        &copies.fixture,
        &copies.python_root,
        &copies.rust_root,
        specs,
        tolerance,
    )
}

pub fn compare_fixture_artifacts(
    stage: u8,
    scope: impl Into<String>,
    fixture: impl Into<String>,
    python_root: impl AsRef<Path>,
    rust_root: impl AsRef<Path>,
    specs: &[ArtifactComparisonSpec],
    tolerance: &ParityTolerance,
) -> Result<ParityRunSummary, ParityError> {
    let scope = scope.into();
    let fixture = fixture.into();
    let python_root = python_root.as_ref();
    let rust_root = rust_root.as_ref();
    let mut summary = ParityRunSummary::default();

    for spec in specs {
        compare_artifact(
            stage,
            &scope,
            &fixture,
            python_root,
            rust_root,
            spec,
            tolerance,
            &mut summary.comparisons,
        )?;
    }

    Ok(summary)
}

fn validate_fixture_name(fixture: &str) -> Result<(), ParityError> {
    let valid = !fixture.is_empty()
        && fixture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    if valid {
        Ok(())
    } else {
        Err(ParityError::InvalidFixtureName(fixture.to_string()))
    }
}

fn copy_dir_recursive(source: &Path, dest: &Path) -> Result<(), ParityError> {
    fs::create_dir_all(dest).map_err(|source| ParityError::CreateFixtureDirectory {
        path: dest.to_path_buf(),
        source,
    })?;
    let entries = fs::read_dir(source).map_err(|source_err| ParityError::ReadFixtureDirectory {
        path: source.to_path_buf(),
        source: source_err,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source_err| ParityError::ReadFixtureDirectory {
            path: source.to_path_buf(),
            source: source_err,
        })?;
        let source_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        let file_type =
            entry
                .file_type()
                .map_err(|source_err| ParityError::ReadFixtureDirectory {
                    path: source.to_path_buf(),
                    source: source_err,
                })?;
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &dest_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &dest_path).map_err(|source| ParityError::CopyFixtureFile {
                source_path,
                dest_path,
                source,
            })?;
        }
    }
    Ok(())
}

fn compare_artifact(
    stage: u8,
    scope: &str,
    fixture: &str,
    python_root: &Path,
    rust_root: &Path,
    spec: &ArtifactComparisonSpec,
    tolerance: &ParityTolerance,
    comparisons: &mut Vec<ParityComparison>,
) -> Result<(), ParityError> {
    let artifact = spec.artifact.to_string_lossy().replace('\\', "/");
    let python_path = python_root.join(&spec.artifact);
    let rust_path = rust_root.join(&spec.artifact);

    if !python_path.exists() {
        push_artifact_missing(
            stage,
            scope,
            fixture,
            &artifact,
            &spec.variables,
            tolerance,
            format!(
                "Missing Python reference artifact {}",
                python_path.display()
            ),
            comparisons,
        );
        return Ok(());
    }
    if !rust_path.exists() {
        push_artifact_missing(
            stage,
            scope,
            fixture,
            &artifact,
            &spec.variables,
            tolerance,
            format!("Missing Rust artifact {}", rust_path.display()),
            comparisons,
        );
        return Ok(());
    }

    let selected_variables = spec
        .variables
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let python_mat = if selected_variables.is_empty() {
        pystamps_mat::MatData::read(&python_path)
    } else {
        pystamps_mat::MatData::read_selected(&python_path, &selected_variables)
    }
    .map_err(|source| ParityError::ReadMat {
        path: python_path.clone(),
        source,
    })?;
    let rust_mat = if selected_variables.is_empty() {
        pystamps_mat::MatData::read(&rust_path)
    } else {
        pystamps_mat::MatData::read_selected(&rust_path, &selected_variables)
    }
    .map_err(|source| ParityError::ReadMat {
        path: rust_path.clone(),
        source,
    })?;

    let variables = if spec.variables.is_empty() {
        python_mat
            .variables()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>()
    } else {
        spec.variables.clone()
    };

    for variable in variables {
        match (python_mat.get(&variable), rust_mat.get(&variable)) {
            (Err(_), _) => comparisons.push(ParityComparison::fail(
                stage,
                scope,
                fixture,
                &artifact,
                &variable,
                tolerance.rtol,
                tolerance.atol,
                format!(
                    "Missing Python variable '{variable}' in {}",
                    python_path.display()
                ),
            )),
            (_, Err(_)) => comparisons.push(ParityComparison::fail(
                stage,
                scope,
                fixture,
                &artifact,
                &variable,
                tolerance.rtol,
                tolerance.atol,
                format!(
                    "Missing Rust variable '{variable}' in {}",
                    rust_path.display()
                ),
            )),
            (Ok(python), Ok(rust)) => comparisons.push(compare_variable(
                stage, scope, fixture, &artifact, &variable, python, rust, tolerance,
            )),
        }
    }

    Ok(())
}

fn push_artifact_missing(
    stage: u8,
    scope: &str,
    fixture: &str,
    artifact: &str,
    variables: &[String],
    tolerance: &ParityTolerance,
    message: String,
    comparisons: &mut Vec<ParityComparison>,
) {
    let variables: Vec<&str> = if variables.is_empty() {
        vec!["<artifact>"]
    } else {
        variables.iter().map(String::as_str).collect()
    };
    for variable in variables {
        comparisons.push(ParityComparison::fail(
            stage,
            scope,
            fixture,
            artifact,
            variable,
            tolerance.rtol,
            tolerance.atol,
            format!("{message}; variable '{variable}'"),
        ));
    }
}

fn compare_variable(
    stage: u8,
    scope: &str,
    fixture: &str,
    artifact: &str,
    variable: &str,
    python: &pystamps_mat::MatArray,
    rust: &pystamps_mat::MatArray,
    tolerance: &ParityTolerance,
) -> ParityComparison {
    if (python.rows, python.cols) != (rust.rows, rust.cols) {
        return ParityComparison::fail(
            stage,
            scope,
            fixture,
            artifact,
            variable,
            tolerance.rtol,
            tolerance.atol,
            format!(
                "Shape mismatch for {} variable '{variable}': Rust {}x{} != Python {}x{}",
                artifact, rust.rows, rust.cols, python.rows, python.cols
            ),
        );
    }
    if python.is_complex() != rust.is_complex() {
        return ParityComparison::fail(
            stage,
            scope,
            fixture,
            artifact,
            variable,
            tolerance.rtol,
            tolerance.atol,
            format!(
                "Type mismatch for {} variable '{variable}': Rust complex={} != Python complex={}",
                artifact,
                rust.is_complex(),
                python.is_complex()
            ),
        );
    }

    let wrap_key = tolerance.wrap_equivalence
        && tolerance
            .wrap_keys
            .iter()
            .any(|key| variable == key || variable.ends_with(&format!(".{key}")));
    let result = if python.is_complex() {
        compare_complex_values(python, rust, tolerance, wrap_key)
    } else {
        compare_real_values(python, rust, tolerance, wrap_key)
    };

    match result {
        Ok(()) => ParityComparison::pass(
            stage,
            scope,
            fixture,
            artifact,
            variable,
            tolerance.rtol,
            tolerance.atol,
        ),
        Err(message) => ParityComparison::fail(
            stage,
            scope,
            fixture,
            artifact,
            variable,
            tolerance.rtol,
            tolerance.atol,
            message,
        ),
    }
}

fn compare_real_values(
    python: &pystamps_mat::MatArray,
    rust: &pystamps_mat::MatArray,
    tolerance: &ParityTolerance,
    wrap_key: bool,
) -> Result<(), String> {
    let python_values = numeric_to_f64(&python.real);
    let rust_values = numeric_to_f64(&rust.real);
    let mut max_abs = 0.0_f64;
    for (&expected, &actual) in python_values.iter().zip(rust_values.iter()) {
        if expected.is_nan() && actual.is_nan() {
            continue;
        }
        if expected.is_nan() || actual.is_nan() {
            let kind = if wrap_key {
                "Wrap mismatch"
            } else {
                "Value mismatch"
            };
            return Err(format!(
                "{kind} for {} variable '{}', max_abs=NaN",
                python.name, python.name
            ));
        }
        let diff = if wrap_key {
            wrapped_diff(actual - expected, tolerance.wrap_period)
        } else {
            actual - expected
        };
        let abs_diff = diff.abs();
        max_abs = max_abs.max(abs_diff);
        if !within_tolerance(abs_diff, expected.abs(), tolerance) {
            let kind = if wrap_key {
                "Wrap mismatch"
            } else {
                "Value mismatch"
            };
            return Err(format!(
                "{kind} for {} variable '{}', max_abs={max_abs:.6e}",
                python.name, python.name
            ));
        }
    }
    Ok(())
}

fn compare_complex_values(
    python: &pystamps_mat::MatArray,
    rust: &pystamps_mat::MatArray,
    tolerance: &ParityTolerance,
    wrap_key: bool,
) -> Result<(), String> {
    let Some(python_imag) = &python.imag else {
        return Err(format!(
            "Missing Python imaginary payload for variable '{}'",
            python.name
        ));
    };
    let Some(rust_imag) = &rust.imag else {
        return Err(format!(
            "Missing Rust imaginary payload for variable '{}'",
            rust.name
        ));
    };
    let python_real = numeric_to_f64(&python.real);
    let rust_real = numeric_to_f64(&rust.real);
    let python_imag = numeric_to_f64(python_imag);
    let rust_imag = numeric_to_f64(rust_imag);

    let mut max_abs = 0.0_f64;
    for (((&expected_re, &expected_im), &actual_re), &actual_im) in python_real
        .iter()
        .zip(python_imag.iter())
        .zip(rust_real.iter())
        .zip(rust_imag.iter())
    {
        let real_equal_nan = expected_re.is_nan() && actual_re.is_nan();
        let imag_equal_nan = expected_im.is_nan() && actual_im.is_nan();
        if wrap_key && real_equal_nan {
            continue;
        }
        if (expected_re.is_nan() || actual_re.is_nan()) && !real_equal_nan
            || (expected_im.is_nan() || actual_im.is_nan()) && !imag_equal_nan
        {
            let kind = if wrap_key {
                "Wrap mismatch"
            } else {
                "Value mismatch"
            };
            return Err(format!(
                "{kind} for {} variable '{}', max_abs=NaN",
                python.name, python.name
            ));
        }

        let expected_re = if real_equal_nan { 0.0 } else { expected_re };
        let actual_re = if real_equal_nan { 0.0 } else { actual_re };
        let expected_im = if imag_equal_nan { 0.0 } else { expected_im };
        let actual_im = if imag_equal_nan { 0.0 } else { actual_im };
        let (abs_diff, scale) = if wrap_key {
            (
                complex_phase_diff(actual_re, actual_im, expected_re, expected_im).abs(),
                0.0_f64,
            )
        } else {
            let diff_re = actual_re - expected_re;
            let diff_im = actual_im - expected_im;
            let expected_abs = expected_re.hypot(expected_im);
            (diff_re.hypot(diff_im), expected_abs)
        };
        max_abs = max_abs.max(abs_diff);
        if !within_tolerance(abs_diff, scale, tolerance) {
            let kind = if wrap_key {
                "Wrap mismatch"
            } else {
                "Value mismatch"
            };
            return Err(format!(
                "{kind} for {} variable '{}', max_abs={max_abs:.6e}",
                python.name, python.name
            ));
        }
    }
    Ok(())
}

fn numeric_to_f64(data: &pystamps_mat::NumericData) -> Vec<f64> {
    match data {
        pystamps_mat::NumericData::F64(values) => values.clone(),
        pystamps_mat::NumericData::F32(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::I8(values) => values.iter().map(|&value| value as f64).collect(),
        pystamps_mat::NumericData::U8(values) => values.iter().map(|&value| value as f64).collect(),
        pystamps_mat::NumericData::I16(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::U16(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::I32(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::U32(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::I64(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
        pystamps_mat::NumericData::U64(values) => {
            values.iter().map(|&value| value as f64).collect()
        }
    }
}

fn wrapped_diff(diff: f64, period: f64) -> f64 {
    (diff + period / 2.0).rem_euclid(period) - period / 2.0
}

fn complex_phase_diff(actual_re: f64, actual_im: f64, expected_re: f64, expected_im: f64) -> f64 {
    let product_re = actual_re * expected_re + actual_im * expected_im;
    let product_im = actual_im * expected_re - actual_re * expected_im;
    product_im.atan2(product_re)
}

fn within_tolerance(abs_diff: f64, expected_abs: f64, tolerance: &ParityTolerance) -> bool {
    abs_diff <= tolerance.atol + tolerance.rtol * expected_abs
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_mat::MatFile;

    #[test]
    fn summary_reports_failures() {
        let summary = ParityRunSummary {
            comparisons: vec![
                ParityComparison::pass(1, "patch", "synthetic", "ps1.mat", "ij", 0.0, 0.0),
                ParityComparison::fail(
                    1,
                    "patch",
                    "synthetic",
                    "ph1.mat",
                    "ph",
                    1e-6,
                    1e-9,
                    "shape mismatch",
                ),
            ],
        };

        assert!(!summary.all_ok());
        assert_eq!(summary.failures().count(), 1);
    }

    #[test]
    fn comparison_serializes_prd_fields() {
        let comparison = ParityComparison::pass(1, "patch", "fixture", "ps1.mat", "ij", 0.0, 0.0);
        let json = serde_json::to_string(&comparison).unwrap();

        assert!(json.contains("\"stage\":1"));
        assert!(json.contains("\"scope\":\"patch\""));
        assert!(json.contains("\"ok\":true"));
    }

    #[test]
    fn fixture_runner_creates_identical_python_and_rust_copies() {
        let root = temp_root("fixture-runner");
        let source = root.join("source");
        fs::create_dir_all(source.join("PATCH_1")).unwrap();
        fs::write(source.join("PATCH_1").join("seed.txt"), b"same-inputs").unwrap();

        let copies = create_fixture_run_copies(&source, root.join("work"), "synthetic").unwrap();

        assert_eq!(
            fs::read(copies.python_root.join("PATCH_1").join("seed.txt")).unwrap(),
            b"same-inputs"
        );
        assert_eq!(
            fs::read(copies.rust_root.join("PATCH_1").join("seed.txt")).unwrap(),
            b"same-inputs"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn identical_scalar_and_complex_matrix_artifacts_pass_and_serialize() {
        let root = temp_root("parity-pass");
        let source = root.join("source");
        fs::create_dir_all(source.join("PATCH_1")).unwrap();
        let specs = vec![ArtifactComparisonSpec::new(
            "PATCH_1/artifact.mat",
            ["scalar", "ph"],
        )];

        let summary = run_fixture_parity(
            &source,
            root.join("work"),
            "synthetic",
            1,
            "patch",
            &specs,
            &ParityTolerance::default(),
            |python_root| {
                write_example_artifact(
                    python_root.join("PATCH_1/artifact.mat"),
                    2,
                    2,
                    vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
                )
            },
            |rust_root| {
                write_example_artifact(
                    rust_root.join("PATCH_1/artifact.mat"),
                    2,
                    2,
                    vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
                )
            },
        )
        .unwrap();

        assert!(summary.all_ok());
        assert_eq!(summary.comparisons.len(), 2);
        let json = summary.to_json_pretty().unwrap();
        assert!(json.contains("\"stage\": 1"));
        assert!(json.contains("\"scope\": \"patch\""));
        assert!(json.contains("\"fixture\": \"synthetic\""));
        assert!(json.contains("\"artifact\": \"PATCH_1/artifact.mat\""));
        assert!(json.contains("\"variable\": \"ph\""));
        assert!(json.contains("\"ok\": true"));
        assert!(json.contains("\"rtol\": 0.00001"));
        assert!(json.contains("\"atol\": 1e-7"));
        assert!(json.contains("\"message\": \"ok\""));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_artifact_reports_path_and_variable_details() {
        let root = temp_root("parity-missing");
        let python = root.join("python");
        let rust = root.join("rust");
        fs::create_dir_all(python.join("PATCH_1")).unwrap();
        fs::create_dir_all(rust.join("PATCH_1")).unwrap();
        write_example_artifact(python.join("PATCH_1/artifact.mat"), 1, 1, vec![(1.0, 0.0)])
            .unwrap();

        let summary = compare_fixture_artifacts(
            1,
            "patch",
            "synthetic",
            &python,
            &rust,
            &[ArtifactComparisonSpec::new("PATCH_1/artifact.mat", ["ph"])],
            &ParityTolerance::default(),
        )
        .unwrap();

        assert!(!summary.all_ok());
        let failure = summary.failures().next().unwrap();
        assert_eq!(failure.artifact, "PATCH_1/artifact.mat");
        assert_eq!(failure.variable, "ph");
        assert!(failure.message.contains("Missing Rust artifact"));
        assert!(failure.message.contains("PATCH_1/artifact.mat"));
        assert!(failure.message.contains("variable 'ph'"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mismatched_shape_reports_false_with_artifact_and_variable() {
        let root = temp_root("parity-shape");
        let python = root.join("python");
        let rust = root.join("rust");
        fs::create_dir_all(python.join("PATCH_1")).unwrap();
        fs::create_dir_all(rust.join("PATCH_1")).unwrap();
        write_example_artifact(
            python.join("PATCH_1/artifact.mat"),
            2,
            2,
            vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        )
        .unwrap();
        write_example_artifact(
            rust.join("PATCH_1/artifact.mat"),
            1,
            4,
            vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        )
        .unwrap();

        let summary = compare_fixture_artifacts(
            1,
            "patch",
            "synthetic",
            &python,
            &rust,
            &[ArtifactComparisonSpec::new("PATCH_1/artifact.mat", ["ph"])],
            &ParityTolerance::default(),
        )
        .unwrap();

        assert!(!summary.all_ok());
        let failure = summary.failures().next().unwrap();
        assert_eq!(failure.artifact, "PATCH_1/artifact.mat");
        assert_eq!(failure.variable, "ph");
        assert!(failure.message.contains("Shape mismatch"));
        assert!(failure.message.contains("Rust 1x4"));
        assert!(failure.message.contains("Python 2x2"));
        fs::remove_dir_all(root).unwrap();
    }

    fn write_example_artifact(
        path: impl AsRef<Path>,
        rows: usize,
        cols: usize,
        ph: Vec<(f32, f32)>,
    ) -> Result<(), ParityError> {
        let path = path.as_ref();
        let mut mat = MatFile::new(path);
        mat.add_f64_scalar("scalar", 7.5).unwrap();
        mat.add_complex_f32_matrix("ph", rows, cols, ph).unwrap();
        mat.write().map_err(|source| ParityError::ReadMat {
            path: path.to_path_buf(),
            source,
        })
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "pystamps-parity-{name}-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn unique_suffix() -> u128 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}

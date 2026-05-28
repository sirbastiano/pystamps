use askama::Template;
use axum::extract::{Form, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use pystamps_core::{
    execute_pipeline_cli_bridge, plan_pipeline, processing_chain_coverage, CliBridgeOptions,
    RunRequest, RuntimeOptions, StageCoverage, StageResult,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use uuid::Uuid;

const REPORT_DIR_NAME: &str = "_native_gate_reports";
const RUN_REPORT_FILE: &str = "native-run-report.json";
const TIMING_REPORT_FILE: &str = "native-run-timings.json";
const VERIFY_REPORT_FILE: &str = "native-verify-report.json";
const MAX_RECENT_RUNS: usize = 50;

#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    runs_root: PathBuf,
}

#[derive(Clone, Debug)]
struct Job {
    id: String,
    request: RunForm,
    state: JobState,
    results: Vec<StageResult>,
    stdout: String,
    stderr: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
}

impl JobState {
    fn label(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct RunForm {
    dataset: String,
    start_step: u8,
    end_step: u8,
    backend: String,
    io_workers: u16,
    cpu_workers: u16,
    dry_run: Option<String>,
}

impl RunForm {
    fn is_dry_run(&self) -> bool {
        self.dry_run.is_some()
    }
}

#[derive(Clone, Debug)]
struct NativeRunSummary {
    run_id: String,
    run_root: String,
    generated_at: String,
    status_label: String,
    status_class: String,
    total_duration: String,
    verifier_label: String,
    verifier_class: String,
    peak_memory: String,
}

#[derive(Clone, Debug)]
struct NativeRunDetail {
    summary: NativeRunSummary,
    dataset: String,
    golden: String,
    stage_range: String,
    backend: String,
    command: String,
    stage_rows: Vec<StageTimingRow>,
    failure_rows: Vec<VerifierFailureRow>,
}

#[derive(Clone, Debug)]
struct StageTimingRow {
    stage: String,
    scope: String,
    target: String,
    status: String,
    status_class: String,
    duration: String,
    input_artifacts: String,
    output_artifacts: String,
    rows_processed: String,
    peak_memory: String,
    details: String,
}

#[derive(Clone, Debug)]
struct VerifierFailureRow {
    artifact_path: String,
    key: String,
    observed_shape: String,
    expected_shape: String,
    max_abs: String,
    tolerance: String,
    message: String,
}

#[derive(Template)]
#[template(
    source = r#"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pySTAMPS Execution</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #16202a; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 18px; font-size: 30px; letter-spacing: 0; }
    h2 { margin: 26px 0 10px; font-size: 18px; letter-spacing: 0; }
    form { display: grid; grid-template-columns: 2fr repeat(4, minmax(108px, 150px)); gap: 12px; align-items: end; background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    label { display: grid; gap: 6px; font-size: 12px; color: #526171; font-weight: 650; }
    input, select, button { min-height: 38px; border: 1px solid #bdc7d3; border-radius: 6px; padding: 7px 10px; font: inherit; background: #fff; box-sizing: border-box; }
    input[type="checkbox"] { min-height: auto; width: 18px; height: 18px; }
    button { cursor: pointer; background: #1f6feb; color: white; border-color: #1f6feb; font-weight: 700; }
    .checkbox { display: flex; align-items: center; gap: 8px; min-height: 38px; }
    .panel { background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    .jobs { margin-top: 22px; display: grid; gap: 14px; }
    .job { background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    .job-head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 12px; }
    .muted { color: #667587; font-size: 13px; }
    .path { overflow-wrap: anywhere; }
    .badge { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 750; border: 1px solid transparent; }
    .badge.ok { background: #e6f4ea; color: #137333; border-color: #b7dfc0; }
    .badge.failed { background: #fde8e8; color: #b42318; border-color: #f5b8b8; }
    .badge.unknown { background: #eef2f6; color: #475467; border-color: #d5dde7; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-top: 1px solid #e3e8ef; padding: 8px 6px; vertical-align: top; }
    th { color: #526171; font-size: 12px; }
    a { color: #1f6feb; text-decoration: none; font-weight: 650; }
    a:hover { text-decoration: underline; }
    pre { overflow: auto; max-height: 240px; background: #111827; color: #f9fafb; border-radius: 6px; padding: 12px; }
    @media (max-width: 920px) { form { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 560px) { main { padding: 18px; } form { grid-template-columns: 1fr; } .job-head { display: block; } }
  </style>
</head>
<body>
<main>
  <h1>pySTAMPS Execution</h1>

  <section class="panel">
    <div class="job-head">
      <h2>Recent native runs</h2>
      <div class="muted path">{{ runs_root }}</div>
    </div>
    {% if runs.is_empty() %}
      <div class="muted">No native run reports found.</div>
    {% else %}
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Total duration</th>
            <th>Verifier</th>
            <th>Peak memory</th>
            <th>Generated</th>
          </tr>
        </thead>
        <tbody>
        {% for run in runs %}
          <tr>
            <td><a href="/runs/{{ run.run_id }}">{{ run.run_id }}</a></td>
            <td><span class="badge {{ run.status_class }}">{{ run.status_label }}</span></td>
            <td>{{ run.total_duration }}</td>
            <td><span class="badge {{ run.verifier_class }}">{{ run.verifier_label }}</span></td>
            <td>{{ run.peak_memory }}</td>
            <td>{{ run.generated_at }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    {% endif %}
  </section>

  <form method="post" action="/runs">
    <label>Dataset path<input name="dataset" required placeholder="/path/to/dataset"></label>
    <label>Start step<input name="start_step" type="number" min="1" max="8" value="1"></label>
    <label>End step<input name="end_step" type="number" min="1" max="8" value="8"></label>
    <label>Backend
      <select name="backend">
        <option value="native">native</option>
        <option value="auto">auto</option>
        <option value="threads">threads</option>
        <option value="processes">processes</option>
        <option value="gpu">gpu</option>
      </select>
    </label>
    <label>IO workers<input name="io_workers" type="number" min="1" value="8"></label>
    <label>CPU workers<input name="cpu_workers" type="number" min="0" value="0"></label>
    <label><span>Mode</span><span class="checkbox"><input name="dry_run" type="checkbox" checked> Dry run</span></label>
    <button type="submit">Run</button>
  </form>

  <section class="jobs">
    {% for job in jobs %}
      <article class="job">
        <div class="job-head">
          <div><strong>{{ job.id }}</strong> <span class="muted">{{ job.state.label() }}</span></div>
          <div class="muted">{{ job.request.dataset }} · stages {{ job.request.start_step }}-{{ job.request.end_step }} · {{ job.request.backend }}</div>
        </div>
        {% if !job.results.is_empty() %}
        <table>
          <thead><tr><th>Stage</th><th>Scope</th><th>Target</th><th>Status</th><th>Details</th></tr></thead>
          <tbody>
          {% for result in job.results %}
            <tr>
              <td>{{ result.stage }}</td>
              <td>{{ result.scope }}</td>
              <td>{{ result.target }}</td>
              <td>{{ result.status }}</td>
              <td>{{ result.details }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
        {% endif %}
        {% if !job.stderr.is_empty() %}
          <pre>{{ job.stderr }}</pre>
        {% endif %}
      </article>
    {% endfor %}
  </section>
</main>
</body>
</html>
"#,
    ext = "html"
)]
struct IndexTemplate {
    jobs: Vec<Job>,
    runs: Vec<NativeRunSummary>,
    runs_root: String,
}

#[derive(Template)]
#[template(
    source = r#"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ run.summary.run_id }} · pySTAMPS Run</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #16202a; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 26px 0 10px; font-size: 18px; letter-spacing: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border: 1px solid #d7dee8; }
    th, td { text-align: left; border-top: 1px solid #e3e8ef; padding: 8px 6px; vertical-align: top; }
    th { color: #526171; font-size: 12px; }
    .panel { background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
    .metric { background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 12px; }
    .metric span { display: block; color: #667587; font-size: 12px; font-weight: 700; margin-bottom: 4px; }
    .metric .badge { display: inline-flex; }
    .path { overflow-wrap: anywhere; }
    .muted { color: #667587; font-size: 13px; }
    .badge { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 750; border: 1px solid transparent; }
    .badge.ok { background: #e6f4ea; color: #137333; border-color: #b7dfc0; }
    .badge.failed { background: #fde8e8; color: #b42318; border-color: #f5b8b8; }
    .badge.unknown { background: #eef2f6; color: #475467; border-color: #d5dde7; }
    a { color: #1f6feb; text-decoration: none; font-weight: 650; }
    a:hover { text-decoration: underline; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    @media (max-width: 880px) { .grid { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 560px) { main { padding: 18px; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <p><a href="/">Back to runs</a></p>
  <h1>{{ run.summary.run_id }}</h1>
  <div class="muted path">{{ run.summary.run_root }}</div>

  <section class="grid">
    <div class="metric"><span>Status</span><span class="badge {{ run.summary.status_class }}">{{ run.summary.status_label }}</span></div>
    <div class="metric"><span>Total duration</span>{{ run.summary.total_duration }}</div>
    <div class="metric"><span>Verifier</span><span class="badge {{ run.summary.verifier_class }}">{{ run.summary.verifier_label }}</span></div>
    <div class="metric"><span>Peak memory</span>{{ run.summary.peak_memory }}</div>
    <div class="metric"><span>Generated</span>{{ run.summary.generated_at }}</div>
    <div class="metric"><span>Stages</span>{{ run.stage_range }}</div>
    <div class="metric"><span>Backend</span>{{ run.backend }}</div>
    <div class="metric"><span>Dataset</span><span class="path">{{ run.dataset }}</span></div>
    <div class="metric"><span>Golden</span><span class="path">{{ run.golden }}</span></div>
  </section>

  <h2>Command</h2>
  <section class="panel"><code>{{ run.command }}</code></section>

  <h2>Stage timings</h2>
  <table>
    <thead>
      <tr>
        <th>Stage</th>
        <th>Scope</th>
        <th>Target</th>
        <th>Status</th>
        <th>Duration</th>
        <th>Inputs</th>
        <th>Outputs</th>
        <th>Rows</th>
        <th>Peak memory</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
    {% for stage in run.stage_rows %}
      <tr>
        <td>{{ stage.stage }}</td>
        <td>{{ stage.scope }}</td>
        <td>{{ stage.target }}</td>
        <td><span class="badge {{ stage.status_class }}">{{ stage.status }}</span></td>
        <td>{{ stage.duration }}</td>
        <td>{{ stage.input_artifacts }}</td>
        <td>{{ stage.output_artifacts }}</td>
        <td>{{ stage.rows_processed }}</td>
        <td>{{ stage.peak_memory }}</td>
        <td>{{ stage.details }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Verifier failures</h2>
  {% if run.failure_rows.is_empty() %}
    <section class="panel muted">No verifier failures reported.</section>
  {% else %}
    <table>
      <thead>
        <tr>
          <th>Artifact path</th>
          <th>Key</th>
          <th>Observed shape</th>
          <th>Expected shape</th>
          <th>max_abs</th>
          <th>Tolerance</th>
          <th>Message</th>
        </tr>
      </thead>
      <tbody>
      {% for failure in run.failure_rows %}
        <tr>
          <td>{{ failure.artifact_path }}</td>
          <td>{{ failure.key }}</td>
          <td>{{ failure.observed_shape }}</td>
          <td>{{ failure.expected_shape }}</td>
          <td>{{ failure.max_abs }}</td>
          <td>{{ failure.tolerance }}</td>
          <td>{{ failure.message }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  {% endif %}
</main>
</body>
</html>
"#,
    ext = "html"
)]
struct RunDetailTemplate {
    run: NativeRunDetail,
}

#[derive(Template)]
#[template(
    source = r#"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ job.id }} · pySTAMPS Job</title>
</head>
<body>
<main>
  <p><a href="/">Back to runs</a></p>
  <h1>{{ job.id }}</h1>
  <p>{{ job.state.label() }} · {{ job.request.dataset }}</p>
  {% if !job.results.is_empty() %}
  <table>
    <thead><tr><th>Stage</th><th>Scope</th><th>Target</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>
    {% for result in job.results %}
      <tr>
        <td>{{ result.stage }}</td>
        <td>{{ result.scope }}</td>
        <td>{{ result.target }}</td>
        <td>{{ result.status }}</td>
        <td>{{ result.details }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
  {% if !job.stderr.is_empty() %}
    <pre>{{ job.stderr }}</pre>
  {% endif %}
</main>
</body>
</html>
"#,
    ext = "html"
)]
struct JobTemplate {
    job: Job,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState {
        jobs: Arc::new(RwLock::new(HashMap::new())),
        runs_root: default_runs_root(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/runs", post(start_run))
        .route("/runs/{id}", get(show_run))
        .route("/api/native-coverage", get(native_coverage))
        .with_state(state);

    let addr: SocketAddr = "127.0.0.1:8787".parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("pystamps-web listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index(State(state): State<AppState>) -> Result<Html<String>, AppError> {
    let mut jobs: Vec<Job> = state.jobs.read().await.values().cloned().collect();
    jobs.sort_by(|left, right| right.id.cmp(&left.id));
    let runs = load_recent_native_runs(&state.runs_root)?;
    render_index(IndexTemplate {
        jobs,
        runs,
        runs_root: state.runs_root.display().to_string(),
    })
}

async fn show_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Html<String>, AppError> {
    if let Some(run_root) = find_run_root_by_id(&state.runs_root, &id)? {
        let Some(run) = load_native_run_detail(&run_root)? else {
            return Err(AppError::not_found(format!("unknown native run id: {id}")));
        };
        return render_run_detail(RunDetailTemplate { run });
    }

    let jobs = state.jobs.read().await;
    let Some(job) = jobs.get(&id) else {
        return Err(AppError::not_found(format!("unknown run id: {id}")));
    };
    render_job(JobTemplate { job: job.clone() })
}

async fn start_run(
    State(state): State<AppState>,
    Form(form): Form<RunForm>,
) -> Result<Response, AppError> {
    validate_form(&form)?;
    let id = Uuid::new_v4().to_string();
    let job = Job {
        id: id.clone(),
        request: form.clone(),
        state: JobState::Queued,
        results: Vec::new(),
        stdout: String::new(),
        stderr: String::new(),
    };
    state.jobs.write().await.insert(id.clone(), job);

    let state_for_task = state.clone();
    let id_for_task = id.clone();
    tokio::spawn(async move {
        run_job(state_for_task, id_for_task).await;
    });

    Ok((
        StatusCode::SEE_OTHER,
        [("Location", format!("/runs/{id}"))],
        "",
    )
        .into_response())
}

async fn native_coverage() -> Result<Json<Vec<StageCoverage>>, AppError> {
    processing_chain_coverage(1, 8)
        .map(Json)
        .map_err(|err| AppError::internal(err.to_string()))
}

async fn run_job(state: AppState, id: String) {
    let request = {
        let mut jobs = state.jobs.write().await;
        let Some(job) = jobs.get_mut(&id) else {
            return;
        };
        job.state = JobState::Running;
        job.request.clone()
    };

    if request.is_dry_run() {
        let planned = plan_pipeline(&RunRequest {
            dataset_root: PathBuf::from(&request.dataset),
            start_step: request.start_step,
            end_step: request.end_step,
            dry_run: true,
        });
        let mut jobs = state.jobs.write().await;
        let Some(job) = jobs.get_mut(&id) else {
            return;
        };
        match planned {
            Ok(results) => {
                job.results = results;
                job.state = JobState::Completed;
            }
            Err(err) => {
                job.stderr = err.to_string();
                job.state = JobState::Failed;
            }
        }
        return;
    }

    let core_request = RunRequest {
        dataset_root: PathBuf::from(&request.dataset),
        start_step: request.start_step,
        end_step: request.end_step,
        dry_run: false,
    };
    let options = CliBridgeOptions {
        runtime: RuntimeOptions {
            backend: request.backend.clone(),
            stage2_kernel_backend: "native".to_string(),
            io_workers: request.io_workers,
            cpu_workers: request.cpu_workers,
        },
        ..CliBridgeOptions::default()
    };
    let execution =
        tokio::task::spawn_blocking(move || execute_pipeline_cli_bridge(&core_request, &options))
            .await;

    let mut jobs = state.jobs.write().await;
    let Some(job) = jobs.get_mut(&id) else {
        return;
    };
    match execution {
        Ok(Ok(execution)) => {
            job.stdout = execution.stdout;
            job.stderr = execution.stderr;
            job.results = execution.results;
            job.state = if execution.exit_code == Some(0) {
                JobState::Completed
            } else {
                JobState::Failed
            };
        }
        Ok(Err(err)) => {
            job.stderr = err.to_string();
            job.state = JobState::Failed;
        }
        Err(err) => {
            job.stderr = format!("execution task failed: {err}");
            job.state = JobState::Failed;
        }
    }
}

fn validate_form(form: &RunForm) -> Result<(), AppError> {
    if form.start_step == 0
        || form.end_step == 0
        || form.start_step > form.end_step
        || form.end_step > 8
    {
        return Err(AppError::bad_request("stage range must be within 1..8"));
    }
    if form.dataset.trim().is_empty() {
        return Err(AppError::bad_request("dataset path is required"));
    }
    match form.backend.as_str() {
        "auto" | "threads" | "processes" | "gpu" | "native" => Ok(()),
        _ => Err(AppError::bad_request("unsupported backend")),
    }
}

fn default_runs_root() -> PathBuf {
    std::env::var_os("PYSTAMPS_WEB_RUNS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("inputs_and_outputs/validation_runs"))
}

fn load_recent_native_runs(runs_root: &Path) -> Result<Vec<NativeRunSummary>, AppError> {
    let mut runs = Vec::new();
    for run_root in candidate_run_roots(runs_root)? {
        if let Some(detail) = load_native_run_detail(&run_root)? {
            runs.push(detail.summary);
        }
    }
    runs.sort_by(|left, right| {
        right
            .generated_at
            .cmp(&left.generated_at)
            .then_with(|| right.run_id.cmp(&left.run_id))
    });
    runs.truncate(MAX_RECENT_RUNS);
    Ok(runs)
}

fn find_run_root_by_id(runs_root: &Path, run_id: &str) -> Result<Option<PathBuf>, AppError> {
    if run_id.contains('/') || run_id.contains('\\') || run_id == "." || run_id == ".." {
        return Ok(None);
    }
    for run_root in candidate_run_roots(runs_root)? {
        if run_root.file_name().and_then(|value| value.to_str()) == Some(run_id) {
            return Ok(Some(run_root));
        }
    }
    Ok(None)
}

fn candidate_run_roots(runs_root: &Path) -> Result<Vec<PathBuf>, AppError> {
    let Ok(entries) = fs::read_dir(runs_root) else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| AppError::internal(err.to_string()))?;
        let path = entry.path();
        if !path.is_dir() || !has_native_report(&path) {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((modified, path));
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    candidates.truncate(MAX_RECENT_RUNS);
    Ok(candidates.into_iter().map(|(_, path)| path).collect())
}

fn has_native_report(run_root: &Path) -> bool {
    let report_dir = run_root.join(REPORT_DIR_NAME);
    report_dir.join(RUN_REPORT_FILE).is_file()
        || report_dir.join(TIMING_REPORT_FILE).is_file()
        || report_dir.join(VERIFY_REPORT_FILE).is_file()
}

fn load_native_run_detail(run_root: &Path) -> Result<Option<NativeRunDetail>, AppError> {
    let report_dir = run_root.join(REPORT_DIR_NAME);
    let run_report = read_json_optional(&report_dir.join(RUN_REPORT_FILE))?;
    let timing_report = read_json_optional(&report_dir.join(TIMING_REPORT_FILE))?;
    let verify_report = read_json_optional(&report_dir.join(VERIFY_REPORT_FILE))?;

    if run_report.is_none() && timing_report.is_none() && verify_report.is_none() {
        return Ok(None);
    }

    let run_json = run_report.as_ref();
    let timing_json = timing_report.as_ref();
    let verify_json = verify_report.as_ref();
    let verifier_payload = verify_json
        .and_then(|payload| payload.get("verifier"))
        .or(verify_json);
    let run_id = run_root
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown")
        .to_string();
    let generated_at =
        string_value(verify_json.and_then(|payload| payload.get("generated_at_utc")))
            .or_else(|| string_value(run_json.and_then(|payload| payload.get("generated_at_utc"))))
            .or_else(|| {
                string_value(timing_json.and_then(|payload| payload.get("generated_at_utc")))
            })
            .unwrap_or_else(|| "unknown".to_string());
    let run_ok = bool_value(run_json.and_then(|payload| payload.get("ok")));
    let verifier_ok = bool_value(verify_json.and_then(|payload| payload.get("ok")))
        .or_else(|| bool_value(verifier_payload.and_then(|payload| payload.get("ok"))));
    let overall_ok = overall_status(run_ok, verifier_ok);
    let summary = NativeRunSummary {
        run_id,
        run_root: run_root.display().to_string(),
        generated_at,
        status_label: status_label(overall_ok).to_string(),
        status_class: status_class(overall_ok).to_string(),
        total_duration: format_optional_seconds(
            number_value(run_json.and_then(|payload| payload.get("elapsed_sec"))).or_else(|| {
                number_value(timing_json.and_then(|payload| payload.get("elapsed_sec")))
            }),
        ),
        verifier_label: verifier_label(verifier_ok, verify_json.is_some()).to_string(),
        verifier_class: verifier_class(verifier_ok, verify_json.is_some()).to_string(),
        peak_memory: format_optional_bytes(peak_memory_bytes(run_json, timing_json)),
    };
    let dataset = path_string(run_json.and_then(|payload| payload.pointer("/setup/dataset")));
    let golden = path_string(verify_json.and_then(|payload| payload.get("golden_root")));
    let start_stage =
        integer_string(run_json.and_then(|payload| payload.pointer("/setup/start_step")));
    let end_stage = integer_string(run_json.and_then(|payload| payload.pointer("/setup/end_step")));
    let stage_range = if start_stage == "-" && end_stage == "-" {
        "-".to_string()
    } else {
        format!("{start_stage}-{end_stage}")
    };
    let backend = command_arg(
        run_json.and_then(|payload| payload.get("command")),
        "--backend",
    )
    .unwrap_or_else(|| "-".to_string());
    let command = command_string(
        run_json
            .and_then(|payload| payload.get("command"))
            .or_else(|| verify_json.and_then(|payload| payload.get("command"))),
    );

    Ok(Some(NativeRunDetail {
        summary,
        dataset,
        golden,
        stage_range,
        backend,
        command,
        stage_rows: stage_rows(run_json, timing_json),
        failure_rows: verifier_failure_rows(verifier_payload),
    }))
}

fn read_json_optional(path: &Path) -> Result<Option<Value>, AppError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)
        .map_err(|err| AppError::internal(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|err| AppError::internal(format!("failed to parse {}: {err}", path.display())))
}

fn stage_rows(run_json: Option<&Value>, timing_json: Option<&Value>) -> Vec<StageTimingRow> {
    let source_rows = timing_json
        .and_then(|payload| payload.get("stages"))
        .and_then(Value::as_array)
        .or_else(|| {
            run_json
                .and_then(|payload| payload.get("results"))
                .and_then(Value::as_array)
        });
    let Some(rows) = source_rows else {
        return Vec::new();
    };
    rows.iter()
        .filter_map(|row| {
            if !row.is_object() {
                return None;
            }
            let status = string_value(row.get("status")).unwrap_or_else(|| "unknown".to_string());
            Some(StageTimingRow {
                stage: integer_string(row.get("stage")),
                scope: path_string(row.get("scope")),
                target: path_string(row.get("target")),
                status_class: stage_status_class(&status).to_string(),
                status,
                duration: format_optional_seconds(number_value(row.get("duration_sec"))),
                input_artifacts: count_or_list(
                    row.get("input_artifact_count"),
                    row.get("input_artifacts"),
                ),
                output_artifacts: count_or_list(
                    row.get("output_artifact_count"),
                    row.get("output_artifacts"),
                ),
                rows_processed: integer_string(row.get("rows_processed")),
                peak_memory: format_optional_bytes(integer_value(row.get("memory_peak_bytes"))),
                details: path_string(row.get("details")),
            })
        })
        .collect()
}

fn verifier_failure_rows(verifier_payload: Option<&Value>) -> Vec<VerifierFailureRow> {
    let Some(failures) = verifier_payload
        .and_then(|payload| payload.get("failed"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    failures
        .iter()
        .filter_map(|failure| {
            if !failure.is_object() {
                return None;
            }
            let message = path_string(failure.get("message"));
            let key = string_value(failure.get("failing_key"))
                .or_else(|| extract_key_from_message(&message))
                .unwrap_or_else(|| "-".to_string());
            Some(VerifierFailureRow {
                artifact_path: path_string(failure.get("path")),
                key,
                observed_shape: shape_string(failure.get("shape_run")),
                expected_shape: shape_string(failure.get("shape_oracle")),
                max_abs: format_optional_number(number_value(failure.get("max_abs"))),
                tolerance: string_value(failure.get("tolerance_rule_id"))
                    .or_else(|| string_value(failure.get("comparison_mode")))
                    .unwrap_or_else(|| "-".to_string()),
                message,
            })
        })
        .collect()
}

fn overall_status(run_ok: Option<bool>, verifier_ok: Option<bool>) -> Option<bool> {
    if run_ok == Some(false) || verifier_ok == Some(false) {
        Some(false)
    } else if run_ok == Some(true) || verifier_ok == Some(true) {
        Some(true)
    } else {
        None
    }
}

fn status_label(ok: Option<bool>) -> &'static str {
    match ok {
        Some(true) => "passed",
        Some(false) => "failed",
        None => "unknown",
    }
}

fn status_class(ok: Option<bool>) -> &'static str {
    match ok {
        Some(true) => "ok",
        Some(false) => "failed",
        None => "unknown",
    }
}

fn verifier_label(ok: Option<bool>, report_present: bool) -> &'static str {
    match (ok, report_present) {
        (Some(true), _) => "ok",
        (Some(false), _) => "failed",
        (None, true) => "unknown",
        (None, false) => "not run",
    }
}

fn verifier_class(ok: Option<bool>, report_present: bool) -> &'static str {
    match (ok, report_present) {
        (Some(true), _) => "ok",
        (Some(false), _) => "failed",
        (None, true) => "unknown",
        (None, false) => "unknown",
    }
}

fn stage_status_class(status: &str) -> &'static str {
    match status {
        "completed" | "passed" | "ok" => "ok",
        "failed" => "failed",
        _ => "unknown",
    }
}

fn peak_memory_bytes(run_json: Option<&Value>, timing_json: Option<&Value>) -> Option<u64> {
    let direct = run_json
        .and_then(|payload| integer_value(payload.get("peak_rss_bytes")))
        .or_else(|| run_json.and_then(|payload| integer_value(payload.get("peakRssBytes"))));
    if direct.is_some() {
        return direct;
    }
    timing_json
        .and_then(|payload| payload.get("stages"))
        .and_then(Value::as_array)
        .or_else(|| {
            run_json
                .and_then(|payload| payload.get("results"))
                .and_then(Value::as_array)
        })
        .and_then(|rows| {
            rows.iter()
                .filter_map(|row| integer_value(row.get("memory_peak_bytes")))
                .max()
        })
}

fn command_arg(command: Option<&Value>, arg_name: &str) -> Option<String> {
    let args = command?.as_array()?;
    for pair in args.windows(2) {
        if pair.first().and_then(|value| value.as_str()) == Some(arg_name) {
            return pair
                .get(1)
                .and_then(|value| value.as_str())
                .map(str::to_string);
        }
    }
    None
}

fn command_string(command: Option<&Value>) -> String {
    let Some(args) = command.and_then(Value::as_array) else {
        return "-".to_string();
    };
    args.iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(" ")
}

fn count_or_list(count: Option<&Value>, list: Option<&Value>) -> String {
    if let Some(items) = list.and_then(Value::as_array) {
        if !items.is_empty() {
            return items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
        }
    }
    integer_string(count)
}

fn string_value(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_string)
}

fn bool_value(value: Option<&Value>) -> Option<bool> {
    value.and_then(Value::as_bool)
}

fn number_value(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn integer_value(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64).or_else(|| {
        value
            .and_then(Value::as_i64)
            .and_then(|number| u64::try_from(number).ok())
    })
}

fn path_string(value: Option<&Value>) -> String {
    string_value(value).unwrap_or_else(|| "-".to_string())
}

fn integer_string(value: Option<&Value>) -> String {
    integer_value(value)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(|seconds| format!("{seconds:.3}s"))
        .unwrap_or_else(|| "-".to_string())
}

fn format_optional_number(value: Option<f64>) -> String {
    let Some(number) = value else {
        return "-".to_string();
    };
    if number == 0.0 {
        return "0".to_string();
    }
    let abs = number.abs();
    if !(1.0e-4..1.0e6).contains(&abs) {
        return format!("{number:.6e}");
    }
    trim_float(format!("{number:.6}"))
}

fn trim_float(mut value: String) -> String {
    while value.contains('.') && value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
    value
}

fn format_optional_bytes(value: Option<u64>) -> String {
    let Some(bytes) = value else {
        return "-".to_string();
    };
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} B")
    }
}

fn shape_string(value: Option<&Value>) -> String {
    let Some(shape) = value.and_then(Value::as_array) else {
        return "-".to_string();
    };
    if shape.is_empty() {
        return "()".to_string();
    }
    let dims = shape
        .iter()
        .filter_map(Value::as_u64)
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>();
    if dims.is_empty() {
        "-".to_string()
    } else {
        format!("({})", dims.join(", "))
    }
}

fn extract_key_from_message(message: &str) -> Option<String> {
    let marker = "key '";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn render_index(template: IndexTemplate) -> Result<Html<String>, AppError> {
    template
        .render()
        .map(Html)
        .map_err(|err| AppError::internal(err.to_string()))
}

fn render_run_detail(template: RunDetailTemplate) -> Result<Html<String>, AppError> {
    template
        .render()
        .map(Html)
        .map_err(|err| AppError::internal(err.to_string()))
}

fn render_job(template: JobTemplate) -> Result<Html<String>, AppError> {
    template
        .render()
        .map(Html)
        .map_err(|err| AppError::internal(err.to_string()))
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, self.message).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_run_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("pystamps-web-{name}-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join(REPORT_DIR_NAME)).unwrap();
        root
    }

    fn write_report(root: &Path, file_name: &str, payload: Value) {
        fs::write(
            root.join(REPORT_DIR_NAME).join(file_name),
            serde_json::to_string_pretty(&payload).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn failed_verifier_forces_failed_summary_class() {
        let root = temp_run_root("failed-verifier");
        write_report(
            &root,
            RUN_REPORT_FILE,
            json!({
                "generated_at_utc": "2026-05-28T00:00:00+00:00",
                "ok": true,
                "status": "passed",
                "elapsed_sec": 12.5,
                "setup": {"dataset": "/dataset", "start_step": 1, "end_step": 8},
                "command": ["pystamps-native", "run", "--backend", "native"],
                "results": [{
                    "stage": 1,
                    "scope": "patch",
                    "target": "PATCH_1",
                    "status": "completed",
                    "duration_sec": 1.25,
                    "input_artifact_count": 3,
                    "output_artifact_count": 4,
                    "rows_processed": 10,
                    "memory_peak_bytes": 1073741824
                }]
            }),
        );
        write_report(
            &root,
            VERIFY_REPORT_FILE,
            json!({
                "generated_at_utc": "2026-05-28T00:00:01+00:00",
                "ok": false,
                "status": "failed",
                "golden_root": "/golden",
                "verifier": {"ok": false, "checked": 1, "failed": []}
            }),
        );

        let detail = load_native_run_detail(&root).unwrap().unwrap();

        assert_eq!(detail.summary.status_label, "failed");
        assert_eq!(detail.summary.status_class, "failed");
        assert_eq!(detail.summary.verifier_label, "failed");
        assert_eq!(detail.summary.verifier_class, "failed");
        assert_eq!(detail.summary.peak_memory, "1.00 GiB");
        assert_eq!(detail.stage_rows[0].output_artifacts, "4");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verifier_failures_expose_shape_key_and_tolerance() {
        let root = temp_run_root("shape-mismatch");
        write_report(
            &root,
            RUN_REPORT_FILE,
            json!({
                "generated_at_utc": "2026-05-28T00:00:00+00:00",
                "ok": true,
                "elapsed_sec": 1.0,
                "setup": {"dataset": "/dataset", "start_step": 1, "end_step": 8},
                "command": ["pystamps-native", "run", "--backend", "native"],
                "results": []
            }),
        );
        write_report(
            &root,
            VERIFY_REPORT_FILE,
            json!({
                "generated_at_utc": "2026-05-28T00:00:01+00:00",
                "ok": false,
                "verifier": {
                    "ok": false,
                    "checked": 1,
                    "failed": [{
                        "path": "PATCH_1/select1.mat",
                        "message": "Shape mismatch for key 'C_ps2' using tolerance rule 'patch_select1.C_ps2.numeric_f32': (10, 1) != (11, 1)",
                        "failing_key": "C_ps2",
                        "shape_run": [10, 1],
                        "shape_oracle": [11, 1],
                        "max_abs": 0.0,
                        "tolerance_rule_id": "patch_select1.C_ps2.numeric_f32"
                    }]
                }
            }),
        );

        let detail = load_native_run_detail(&root).unwrap().unwrap();
        let failure = &detail.failure_rows[0];

        assert_eq!(failure.artifact_path, "PATCH_1/select1.mat");
        assert_eq!(failure.key, "C_ps2");
        assert_eq!(failure.observed_shape, "(10, 1)");
        assert_eq!(failure.expected_shape, "(11, 1)");
        assert_eq!(failure.max_abs, "0");
        assert_eq!(failure.tolerance, "patch_select1.C_ps2.numeric_f32");

        fs::remove_dir_all(root).unwrap();
    }
}

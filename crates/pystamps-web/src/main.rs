use askama::Template;
use axum::extract::{Form, Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use pystamps_core::{
    execute_pipeline_cli_bridge, plan_pipeline, processing_chain_coverage, CliBridgeOptions, RunRequest,
    RuntimeOptions, StageCoverage, StageResult,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Job>>>,
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
    form { display: grid; grid-template-columns: 2fr repeat(4, minmax(108px, 150px)); gap: 12px; align-items: end; background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    label { display: grid; gap: 6px; font-size: 12px; color: #526171; font-weight: 650; }
    input, select, button { min-height: 38px; border: 1px solid #bdc7d3; border-radius: 6px; padding: 7px 10px; font: inherit; background: #fff; box-sizing: border-box; }
    input[type="checkbox"] { min-height: auto; width: 18px; height: 18px; }
    button { cursor: pointer; background: #1f6feb; color: white; border-color: #1f6feb; font-weight: 700; }
    .checkbox { display: flex; align-items: center; gap: 8px; min-height: 38px; }
    .jobs { margin-top: 22px; display: grid; gap: 14px; }
    .job { background: #fff; border: 1px solid #d7dee8; border-radius: 8px; padding: 16px; }
    .job-head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 12px; }
    .muted { color: #667587; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-top: 1px solid #e3e8ef; padding: 8px 6px; vertical-align: top; }
    th { color: #526171; font-size: 12px; }
    pre { overflow: auto; max-height: 240px; background: #111827; color: #f9fafb; border-radius: 6px; padding: 12px; }
    @media (max-width: 920px) { form { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 560px) { main { padding: 18px; } form { grid-template-columns: 1fr; } .job-head { display: block; } }
  </style>
</head>
<body>
<main>
  <h1>pySTAMPS Execution</h1>
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState {
        jobs: Arc::new(RwLock::new(HashMap::new())),
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
    render(IndexTemplate { jobs })
}

async fn show_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Html<String>, AppError> {
    let jobs = state.jobs.read().await;
    let Some(job) = jobs.get(&id) else {
        return Err(AppError::not_found(format!("unknown run id: {id}")));
    };
    render(IndexTemplate {
        jobs: vec![job.clone()],
    })
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
        tokio::task::spawn_blocking(move || execute_pipeline_cli_bridge(&core_request, &options)).await;

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
    if form.start_step == 0 || form.end_step == 0 || form.start_step > form.end_step || form.end_step > 8 {
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

fn render(template: IndexTemplate) -> Result<Html<String>, AppError> {
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

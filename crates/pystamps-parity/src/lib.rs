use serde::{Deserialize, Serialize};

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

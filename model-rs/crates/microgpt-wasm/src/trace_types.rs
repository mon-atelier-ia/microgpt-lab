//! Serializable DTO types for training trace data.
//! These match the TypeScript `TrainingTrace` interface.

use std::collections::HashMap;

use serde::Serialize;

/// Optimizer configuration snapshot.
#[derive(Serialize)]
pub struct TraceOptimizer {
    pub name: String,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub base_learning_rate: f64,
    pub schedule: String,
}

/// A selectable parameter option for the trace UI.
#[derive(Serialize)]
pub struct TraceParamOption {
    pub id: String,
    pub label: String,
    pub matrix: String,
    pub row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_char_display: Option<String>,
}

/// Gradient and post-update values for a single parameter row.
#[derive(Serialize)]
pub struct TraceStepParam {
    pub grad: Vec<f64>,
    pub after: Vec<f64>,
}

/// One training step's trace data.
#[derive(Serialize)]
pub struct TraceStep {
    pub step: usize,
    pub word: String,
    pub loss: Option<f64>,
    pub learning_rate: f64,
    pub params: HashMap<String, TraceStepParam>,
}

/// Result of a single (non-traced) training step.
#[derive(Serialize)]
pub struct StepResult {
    pub step: usize,
    pub loss: f64,
    pub word: String,
    pub lr: f64,
}

/// Training metadata: optimizer config, parameter options, and initial state.
#[derive(Serialize)]
pub struct TrainingMeta {
    pub optimizer: TraceOptimizer,
    pub parameter_options: Vec<TraceParamOption>,
    pub initial_step: TraceStep,
}

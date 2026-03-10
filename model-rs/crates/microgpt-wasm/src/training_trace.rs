//! Training-trace types and helpers shared between `train_step_traced()`
//! (lib.rs) and `training_meta()`.

use std::collections::HashMap;

use microgpt_rs::model::Model;
use microgpt_rs::value::Value;
use serde::Serialize;

// ── Serializable result types (match TypeScript TrainingTrace) ────────────

#[derive(Serialize)]
pub struct TraceOptimizer {
    pub name: String,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub base_learning_rate: f64,
    pub schedule: String,
}

#[derive(Serialize)]
pub struct TraceParamOption {
    pub id: String,
    pub label: String,
    pub matrix: String,
    pub row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_char_display: Option<String>,
}

#[derive(Serialize)]
pub struct TraceStepParam {
    pub grad: Vec<f64>,
    pub after: Vec<f64>,
}

#[derive(Serialize)]
pub struct TraceStep {
    pub step: usize,
    pub word: String,
    pub loss: Option<f64>,
    pub learning_rate: f64,
    pub params: HashMap<String, TraceStepParam>,
}

// ── Parameter tracking (pub(crate) — shared with train_step_traced) ─────

/// Which matrix a tracked parameter belongs to.
#[derive(Clone, Copy)]
pub enum MatrixKind {
    Wte,
    Wpe,
    LmHead,
    AttnWq,
}

pub struct TrackedParam {
    pub id: String,
    pub kind: MatrixKind,
    pub row_index: usize,
}

pub(crate) fn get_matrix_row<'a>(
    model: &'a Model,
    kind: &MatrixKind,
    row_index: usize,
) -> &'a [Value] {
    match kind {
        MatrixKind::Wte => &model.sd.wte[row_index],
        MatrixKind::Wpe => &model.sd.wpe[row_index],
        MatrixKind::LmHead => &model.sd.lm_head[row_index],
        MatrixKind::AttnWq => &model.sd.layers[0].attn_wq[row_index],
    }
}

pub(crate) fn row_grads(row: &[Value]) -> Vec<f64> {
    row.iter().map(Value::grad).collect()
}

pub(crate) fn row_data(row: &[Value]) -> Vec<f64> {
    row.iter().map(Value::data).collect()
}

// ── Build parameter options ──────────────────────────────────────────────

pub(crate) fn build_param_options(tokens: &[String]) -> (Vec<TraceParamOption>, Vec<TrackedParam>) {
    let first_char = tokens.first().map(|s| s.as_str()).unwrap_or("?");

    let options = vec![
        TraceParamOption {
            id: "token_first".into(),
            label: format!("Token '{}' embedding", first_char),
            matrix: "wte".into(),
            row_index: 0,
            token_char_display: Some(first_char.to_string()),
        },
        TraceParamOption {
            id: "lm_head_first".into(),
            label: format!("Token '{}' LM Head", first_char),
            matrix: "lm_head".into(),
            row_index: 0,
            token_char_display: Some(first_char.to_string()),
        },
        TraceParamOption {
            id: "position_0".into(),
            label: "POS 0 position embedding".into(),
            matrix: "wpe".into(),
            row_index: 0,
            token_char_display: None,
        },
        TraceParamOption {
            id: "attn_wq_row_0".into(),
            label: "W_Q row 0".into(),
            matrix: "attn_wq".into(),
            row_index: 0,
            token_char_display: None,
        },
    ];

    let tracked: Vec<TrackedParam> = options
        .iter()
        .map(|o| TrackedParam {
            id: o.id.clone(),
            kind: match o.matrix.as_str() {
                "wte" => MatrixKind::Wte,
                "wpe" => MatrixKind::Wpe,
                "lm_head" => MatrixKind::LmHead,
                "attn_wq" => MatrixKind::AttnWq,
                _ => unreachable!(),
            },
            row_index: o.row_index,
        })
        .collect();

    (options, tracked)
}

// ── Capture one step's param snapshots ───────────────────────────────────

pub(crate) fn capture_step(
    model: &Model,
    tracked: &[TrackedParam],
    step: usize,
    word: &str,
    loss: Option<f64>,
    lr: f64,
    has_grads: bool,
) -> TraceStep {
    let mut params = HashMap::with_capacity(tracked.len());
    for tp in tracked {
        let row = get_matrix_row(model, &tp.kind, tp.row_index);
        params.insert(
            tp.id.clone(),
            TraceStepParam {
                grad: if has_grads { row_grads(row) } else { vec![0.0; row.len()] },
                after: row_data(row),
            },
        );
    }
    TraceStep {
        step,
        word: word.to_string(),
        loss,
        learning_rate: lr,
        params,
    }
}


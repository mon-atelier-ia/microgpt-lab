//! Compute per-position embedding gradients via Rust autodiff.
//!
//! Replaces the O(positions × n_embd × 2) JS finite-difference perturbation
//! with a single forward + backward pass.

use microgpt_rs::forward::{forward_probs, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::value::Value;
use serde::Serialize;

// ── Serializable result types ─────────────────────────────────────────────

#[derive(Serialize)]
pub struct PositionGradient {
    /// Token ID at this sequence position.
    pub token_id: usize,
    /// Sequence position index.
    pub position: usize,
    /// Per-dimension embedding gradient: ∂L/∂(wte+wpe) at this position.
    /// Shape: [n_embd].
    pub embedding_grad: Vec<f64>,
}

#[derive(Serialize)]
pub struct BackpropTraceResult {
    /// Cross-entropy loss at selected_pos: -log(p[target]).
    pub loss: f64,
    /// Per-position embedding gradients for positions [0..=selected_pos].
    pub position_grads: Vec<PositionGradient>,
}

// ── Core computation ──────────────────────────────────────────────────────

/// Forward pass up to `selected_pos`, compute cross-entropy loss at that
/// position, then backward to get per-position embedding gradients.
///
/// Uses `wpe[pos].grad()` as the per-position gradient (always unambiguous,
/// unlike `wte[tid].grad()` which accumulates across repeated tokens).
///
/// **Side effect**: temporarily populates `.grad` on model parameters via
/// `RefCell` interior mutability, then zeros them before returning.
pub fn compute_backprop_trace(
    model: &Model,
    token_ids: &[usize],
    selected_pos: usize,
) -> Result<BackpropTraceResult, String> {
    let cfg = model.config;
    let n = token_ids.len().saturating_sub(1).min(cfg.block_size);

    if selected_pos >= n {
        return Err(format!(
            "selected_pos {} out of range (seq_len-1 = {})",
            selected_pos, n
        ));
    }

    let target_token_id = token_ids[selected_pos + 1];
    if target_token_id >= model.vocab_size {
        return Err(format!(
            "target token_id {} out of range (vocab_size = {})",
            target_token_id, model.vocab_size
        ));
    }

    // Forward pass up to selected_pos (inclusive).
    let (mut keys, mut vals) = new_kv_cache(&cfg);
    let mut selected_probs: Option<Vec<Value>> = None;

    for pos_id in 0..=selected_pos {
        let tid = token_ids[pos_id];
        let probs = forward_probs(tid, pos_id, &mut keys, &mut vals, &model.sd, &cfg);
        if pos_id == selected_pos {
            selected_probs = Some(probs);
        }
    }

    let probs = selected_probs.unwrap();
    let target_prob = &probs[target_token_id];
    let loss = target_prob.log().neg();
    let loss_val = loss.data();

    // Backward: populate .grad() on all reachable nodes.
    loss.backward();

    // Extract per-position gradients from wpe (always per-position).
    let position_grads: Vec<PositionGradient> = (0..=selected_pos)
        .map(|pos| {
            let tid = token_ids[pos];
            let wpe_row = &model.sd.wpe[pos];
            let embedding_grad: Vec<f64> = wpe_row.iter().map(Value::grad).collect();
            PositionGradient {
                token_id: tid,
                position: pos,
                embedding_grad,
            }
        })
        .collect();

    // Zero all parameter gradients to avoid accumulation across calls.
    for p in model.params() {
        p.zero_grad();
    }

    Ok(BackpropTraceResult {
        loss: loss_val,
        position_grads,
    })
}

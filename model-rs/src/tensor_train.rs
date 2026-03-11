//! Tensor-level training step: forward, backward, Adam update.
//!
//! Mirrors `train.rs` exactly but uses tensor engine (~100× faster).

use crate::config::TrainConfig;
use crate::tensor::Tensor;
use crate::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use crate::tensor_model::TensorModel;

/// Run one training step on tensor engine.
/// Returns average cross-entropy loss.
pub fn tensor_train_step(
    model: &mut TensorModel,
    tokens: &[usize],
    step: usize,
    tc: &TrainConfig,
) -> f64 {
    let cfg = model.config;
    let n = cfg.block_size.min(tokens.len().saturating_sub(1));
    if n == 0 {
        return 0.0;
    }

    let (mut keys, mut vals) = new_tensor_kv_cache(cfg.n_layer);
    let mut losses: Vec<Tensor> = Vec::with_capacity(n);

    for pos_id in 0..n {
        let token_id = tokens[pos_id];
        let target_id = tokens[pos_id + 1];
        let probs = tensor_forward_probs(
            token_id, pos_id, &mut keys, &mut vals, &model.sd, cfg.n_head, cfg.n_embd,
        );
        losses.push(probs.nll_loss(target_id));
    }

    // Average loss over all positions (matching scalar engine).
    let loss = Tensor::sum_scalars(&losses).scale(1.0 / n as f64);

    loss.backward();
    let loss_val = loss.data()[0];

    // Adam update.
    tensor_adam_step(model, step, tc);

    loss_val
}

/// Adam optimizer — delegates to `TensorModel::adam_step`.
fn tensor_adam_step(model: &mut TensorModel, step: usize, tc: &TrainConfig) {
    model.adam_step(step, tc);
}

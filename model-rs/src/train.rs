//! Single training step: forward, backward, Adam update.

use crate::config::TrainConfig;
use crate::forward::{forward_probs, new_kv_cache};
use crate::model::Model;

/// Run one training step: forward all positions → loss → backward → Adam.
/// Returns the average cross-entropy loss for this sequence.
pub fn train_step(model: &mut Model, tokens: &[usize], step: usize, tc: &TrainConfig) -> f64 {
    let cfg = model.config;
    let n = cfg.block_size.min(tokens.len().saturating_sub(1));
    if n == 0 {
        return 0.0;
    }

    let (mut keys, mut vals) = new_kv_cache(&cfg);
    let mut losses = Vec::with_capacity(n);

    for pos_id in 0..n {
        let token_id = tokens[pos_id];
        let target_id = tokens[pos_id + 1];
        let probs = forward_probs(token_id, pos_id, &mut keys, &mut vals, &model.sd, &cfg);
        losses.push(probs[target_id].log().neg());
    }

    let loss = losses
        .iter()
        .skip(1)
        .fold(losses[0].clone(), |a, b| a.add(b))
        .mul_f64(1.0 / n as f64);

    loss.backward();
    let loss_val = loss.data();

    model.adam_step(step, tc);

    loss_val
}

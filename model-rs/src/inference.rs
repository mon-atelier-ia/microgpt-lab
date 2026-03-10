//! Autoregressive text generation with temperature sampling.

use crate::config::ModelConfig;
use crate::data::Vocab;
use crate::forward::{forward, new_kv_cache};
use crate::model::StateDict;
use crate::ops::softmax;
use crate::rng::Rng;
use crate::value::Value;

/// Generate `n_samples` names autoregressively with temperature scaling.
///
/// If `prefix` is non-empty, its characters are forced as the first tokens
/// (they must exist in the vocabulary). Sampling starts after the prefix.
pub fn generate(
    sd: &StateDict,
    vocab: &Vocab,
    rng: &mut Rng,
    cfg: &ModelConfig,
    n_samples: usize,
    temperature: f64,
    prefix: &str,
) -> Vec<String> {
    let bos = vocab.bos();

    // Pre-tokenize prefix characters (skip unknown chars).
    let prefix_ids: Vec<usize> = prefix
        .chars()
        .filter_map(|c| vocab.stoi.get(&c.to_string()).copied())
        .collect();

    let mut results = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let (mut keys, mut vals) = new_kv_cache(cfg);
        let mut token_id = bos;
        let mut name = String::new();

        for pos_id in 0..cfg.block_size {
            let logits = forward(token_id, pos_id, &mut keys, &mut vals, sd, cfg);

            if pos_id < prefix_ids.len() {
                // Force prefix token — no sampling.
                token_id = prefix_ids[pos_id];
            } else {
                let scaled: Vec<Value> = logits
                    .iter()
                    .map(|l| l.mul_f64(1.0 / temperature))
                    .collect();
                let probs = softmax(&scaled);
                let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
                token_id = rng.categorical(&weights);
            }

            if token_id == bos {
                break;
            }
            name.push_str(&vocab.tokens[token_id]);
        }

        results.push(name);
    }

    results
}

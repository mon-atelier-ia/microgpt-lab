use crate::config::ModelConfig;
use crate::model::StateDict;
use crate::ops::{linear, rmsnorm, softmax};
use crate::value::Value;

/// KV cache: `[layer][position][dim]`.
pub type KvCache = Vec<Vec<Vec<Value>>>;

/// Create empty KV caches for all layers.
pub fn new_kv_cache(cfg: &ModelConfig) -> (KvCache, KvCache) {
    (vec![vec![]; cfg.n_layer], vec![vec![]; cfg.n_layer])
}

/// Forward one token through the GPT, returns logits.
pub fn forward(
    token_id: usize,
    pos_id: usize,
    keys: &mut KvCache,
    vals: &mut KvCache,
    sd: &StateDict,
    cfg: &ModelConfig,
) -> Vec<Value> {
    let n_head = cfg.n_head;
    let head_dim = cfg.head_dim();

    let tok_emb = &sd.wte[token_id];
    let pos_emb = &sd.wpe[pos_id];
    let mut x: Vec<Value> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
    x = rmsnorm(&x);

    for li in 0..sd.layers.len() {
        let lw = &sd.layers[li];

        // Multi-head self-attention
        let x_res = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &lw.attn_wq);
        let k = linear(&x, &lw.attn_wk);
        let v = linear(&x, &lw.attn_wv);
        keys[li].push(k);
        vals[li].push(v);

        let seq_len = keys[li].len();
        let scale = (head_dim as f64).sqrt();
        let mut x_attn: Vec<Value> = Vec::with_capacity(n_head * head_dim);

        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];

            let attn_logits: Vec<Value> = (0..seq_len)
                .map(|t| {
                    let dot = q_h
                        .iter()
                        .zip(&keys[li][t][hs..hs + head_dim])
                        .map(|(qi, ki)| qi.mul(ki))
                        .reduce(|a, b| a.add(&b))
                        .expect("head_dim is always > 0");
                    dot.mul_f64(1.0 / scale)
                })
                .collect();

            let attn_w = softmax(&attn_logits);

            for j in 0..head_dim {
                let out = (0..seq_len)
                    .map(|t| attn_w[t].mul(&vals[li][t][hs + j]))
                    .reduce(|a, b| a.add(&b))
                    .expect("seq_len is always > 0");
                x_attn.push(out);
            }
        }

        x = linear(&x_attn, &lw.attn_wo);
        x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();

        // MLP
        let x_res = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &lw.mlp_fc1);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &lw.mlp_fc2);
        x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();
    }

    linear(&x, &sd.lm_head)
}

/// Forward one token and return softmax probabilities.
pub fn forward_probs(
    token_id: usize,
    pos_id: usize,
    keys: &mut KvCache,
    vals: &mut KvCache,
    sd: &StateDict,
    cfg: &ModelConfig,
) -> Vec<Value> {
    let logits = forward(token_id, pos_id, keys, vals, sd, cfg);
    softmax(&logits)
}

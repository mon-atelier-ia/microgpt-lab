use microgpt_rs::config::ModelConfig;
use microgpt_rs::forward::new_kv_cache;
use microgpt_rs::model::StateDict;
use microgpt_rs::ops::{linear, rmsnorm, softmax};
use microgpt_rs::value::Value;
use serde::Serialize;

fn vals_to_f64(vs: &[Value]) -> Vec<f64> {
    vs.iter().map(|v| v.data()).collect()
}

/// All intermediate values from one forward pass, for animation.
#[derive(Serialize)]
pub struct ForwardTrace {
    /// Embedded + normed x vector at each position `[seq_len][n_embd]`.
    pub x_vectors: Vec<Vec<f64>>,
    /// Q vector for selected head at query_pos `[head_dim]`.
    pub q_vector: Vec<f64>,
    /// K rows for selected head, all positions `[seq_len][head_dim]`.
    pub k_rows: Vec<Vec<f64>>,
    /// V rows for selected head, all positions `[seq_len][head_dim]`.
    pub v_rows: Vec<Vec<f64>>,
    /// Attention weights for selected head at query_pos `[seq_len]`.
    pub attn_weights: Vec<f64>,
    /// Attention output for selected head `[head_dim]`.
    pub attn_output: Vec<f64>,
    /// Per-head attention output `[n_head][head_dim]`.
    pub head_outputs: Vec<Vec<f64>>,
    /// Concatenated multi-head attention output (before Wo) `[n_embd]`.
    pub x_attn_vector: Vec<f64>,
    /// After Wo projection + residual `[n_embd]`.
    pub mha_output: Vec<f64>,
    /// After attention block (= mha_output, same as residual add) `[n_embd]`.
    pub attn_block_result: Vec<f64>,
    /// After MLP block + residual `[n_embd]`.
    pub block_output: Vec<f64>,
    /// Raw logits `[vocab_size]`.
    pub logits: Vec<f64>,
    /// Softmax probabilities `[vocab_size]`.
    pub probs: Vec<f64>,
}

/// Forward pass that captures all intermediates at `query_pos` for `head_idx`.
///
/// Single-pass: processes positions 0..=query_pos, capturing intermediates
/// inline at query_pos instead of recomputing them.
pub fn forward_with_trace(
    token_ids: &[usize],
    query_pos: usize,
    head_idx: usize,
    sd: &StateDict,
    cfg: &ModelConfig,
) -> ForwardTrace {
    let n_head = cfg.n_head;
    let head_dim = cfg.head_dim();
    let (mut keys, mut vals) = new_kv_cache(cfg);

    let mut x_vectors: Vec<Vec<f64>> = Vec::with_capacity(query_pos + 1);

    // Trace intermediates — populated when pos == query_pos.
    let mut trace_q_vector = vec![];
    let mut trace_attn_weights = vec![];
    let mut trace_attn_output = vec![];
    let mut trace_head_outputs: Vec<Vec<f64>> = vec![];
    let mut trace_x_attn_vector = vec![];
    let mut trace_mha_output = vec![];
    let mut trace_block_output = vec![];
    let mut trace_logits = vec![];
    let mut trace_probs = vec![];

    for pos in 0..=query_pos {
        let tok_emb = &sd.wte[token_ids[pos]];
        let pos_emb = &sd.wpe[pos];
        let x: Vec<Value> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
        let x = rmsnorm(&x);
        x_vectors.push(vals_to_f64(&x));

        let is_query = pos == query_pos;

        // Transformer layers (n_layer = 1 typically).
        let mut x = x;
        for li in 0..sd.layers.len() {
            let lw = &sd.layers[li];
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
                            .expect("head_dim > 0");
                        dot.mul_f64(1.0 / scale)
                    })
                    .collect();

                let attn_w = softmax(&attn_logits);

                let mut head_out: Vec<Value> = Vec::with_capacity(head_dim);
                for j in 0..head_dim {
                    let out = (0..seq_len)
                        .map(|t| attn_w[t].mul(&vals[li][t][hs + j]))
                        .reduce(|a, b| a.add(&b))
                        .expect("seq_len > 0");
                    head_out.push(out);
                }

                // Capture per-head intermediates at query_pos.
                if is_query {
                    if h == head_idx {
                        trace_q_vector = vals_to_f64(&q[hs..hs + head_dim]);
                        trace_attn_weights = vals_to_f64(&attn_w);
                        trace_attn_output = vals_to_f64(&head_out);
                    }
                    trace_head_outputs.push(vals_to_f64(&head_out));
                }

                x_attn.extend(head_out);
            }

            if is_query {
                trace_x_attn_vector = vals_to_f64(&x_attn);
            }

            // Wo projection + residual.
            x = linear(&x_attn, &lw.attn_wo);
            x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();

            if is_query {
                trace_mha_output = vals_to_f64(&x);
            }

            // MLP block.
            let x_res = x.clone();
            x = rmsnorm(&x);
            x = linear(&x, &lw.mlp_fc1);
            x = x.iter().map(|xi| xi.relu()).collect();
            x = linear(&x, &lw.mlp_fc2);
            x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();
        }

        // Capture final intermediates at query_pos.
        if is_query {
            trace_block_output = vals_to_f64(&x);
            let logit_vals = linear(&x, &sd.lm_head);
            trace_logits = vals_to_f64(&logit_vals);
            let prob_vals = softmax(&logit_vals);
            trace_probs = vals_to_f64(&prob_vals);
        }
    }

    // Extract K/V rows for selected head from cache.
    let li = 0;
    let hs = head_idx * head_dim;
    let seq_len = query_pos + 1;
    let k_rows: Vec<Vec<f64>> = (0..seq_len)
        .map(|t| vals_to_f64(&keys[li][t][hs..hs + head_dim]))
        .collect();
    let v_rows: Vec<Vec<f64>> = (0..seq_len)
        .map(|t| vals_to_f64(&vals[li][t][hs..hs + head_dim]))
        .collect();

    ForwardTrace {
        x_vectors,
        q_vector: trace_q_vector,
        k_rows,
        v_rows,
        attn_weights: trace_attn_weights,
        attn_output: trace_attn_output,
        head_outputs: trace_head_outputs,
        x_attn_vector: trace_x_attn_vector,
        mha_output: trace_mha_output.clone(),
        attn_block_result: trace_mha_output,
        block_output: trace_block_output,
        logits: trace_logits,
        probs: trace_probs,
    }
}

//! Tensor-level forward pass through the GPT.
//!
//! Single implementation used by training, inference, and trace capture.
//! Mirrors `forward.rs` algorithm exactly but operates on `Tensor` nodes
//! instead of scalar `Value` nodes (~25 nodes vs ~6000).

use crate::tensor::Tensor;

/// KV cache: `[layer][position]`, each entry is `[1, n_embd]`.
pub type TensorKvCache = Vec<Vec<Tensor>>;

/// Create empty KV caches for `n_layer` layers.
pub fn new_tensor_kv_cache(n_layer: usize) -> (TensorKvCache, TensorKvCache) {
    (vec![vec![]; n_layer], vec![vec![]; n_layer])
}

/// Weight matrices as tensors.
pub struct TensorStateDict {
    /// Token embeddings `[vocab_size, n_embd]`.
    pub wte: Tensor,
    /// Position embeddings `[block_size, n_embd]`.
    pub wpe: Tensor,
    /// Output projection `[vocab_size, n_embd]`.
    pub lm_head: Tensor,
    /// Per-layer weights.
    pub layers: Vec<TensorLayerWeights>,
}

/// One transformer block's weights.
pub struct TensorLayerWeights {
    /// `[n_embd, n_embd]`
    pub attn_wq: Tensor,
    /// `[n_embd, n_embd]`
    pub attn_wk: Tensor,
    /// `[n_embd, n_embd]`
    pub attn_wv: Tensor,
    /// `[n_embd, n_embd]`
    pub attn_wo: Tensor,
    /// `[4*n_embd, n_embd]`
    pub mlp_fc1: Tensor,
    /// `[n_embd, 4*n_embd]`
    pub mlp_fc2: Tensor,
}

/// Forward one token, returns logits `[1, vocab_size]`.
///
/// Mirrors `forward()` in `forward.rs`:
/// embed → rmsnorm → (attention + residual) → (MLP + residual) → logits.
pub fn tensor_forward(
    token_id: usize,
    pos_id: usize,
    keys: &mut TensorKvCache,
    vals: &mut TensorKvCache,
    sd: &TensorStateDict,
    n_head: usize,
    n_embd: usize,
) -> Tensor {
    let head_dim = n_embd / n_head;

    // Embedding: wte[token_id] + wpe[pos_id] → [1, n_embd]
    let tok_emb = sd.wte.select_row(token_id);
    let pos_emb = sd.wpe.select_row(pos_id);
    let mut x = tok_emb.add(&pos_emb);
    x = x.rmsnorm();

    for (li, lw) in sd.layers.iter().enumerate() {
        let x_res = x.clone();
        x = x.rmsnorm();

        // Q, K, V: x[1, e] @ W^T[e, e] = [1, e]
        let q = x.matmul_rhs_t(&lw.attn_wq);
        let k = x.matmul_rhs_t(&lw.attn_wk);
        let v = x.matmul_rhs_t(&lw.attn_wv);

        keys[li].push(k);
        vals[li].push(v);

        let seq_len = keys[li].len();
        let scale = (head_dim as f64).sqrt();

        // Per-head attention
        let mut head_outs: Vec<Tensor> = Vec::with_capacity(n_head);
        for h in 0..n_head {
            let hs = h * head_dim;
            let he = hs + head_dim;

            // Q_h: select columns [hs..he] from current Q → [1, head_dim]
            // Actually Q is computed above, not in cache. Use the q tensor.
            let q_h = q.select_cols(hs, he); // [1, head_dim]

            // Build K matrix for this head: stack k_cache[0..seq_len][hs..he]
            let k_head_rows: Vec<Tensor> = (0..seq_len)
                .map(|t| keys[li][t].select_cols(hs, he))
                .collect();
            let k_mat = Tensor::stack_rows(&k_head_rows); // [seq_len, head_dim]

            // attn_logits = Q_h @ K^T / sqrt(d) → [1, seq_len]
            let attn_logits = q_h.matmul_rhs_t(&k_mat).scale(1.0 / scale);
            let attn_weights = attn_logits.softmax(); // [1, seq_len]

            // Build V matrix for this head: [seq_len, head_dim]
            let v_head_rows: Vec<Tensor> = (0..seq_len)
                .map(|t| vals[li][t].select_cols(hs, he))
                .collect();
            let v_mat = Tensor::stack_rows(&v_head_rows);

            // head_out = attn_weights @ V → [1, head_dim]
            let head_out = attn_weights.matmul(&v_mat);
            head_outs.push(head_out);
        }

        // Concatenate heads → [1, n_embd]
        let x_attn = Tensor::cat_cols(&head_outs);

        // Wo projection + residual
        x = x_attn.matmul_rhs_t(&lw.attn_wo);
        x = x.add(&x_res);

        // MLP
        let x_res = x.clone();
        x = x.rmsnorm();
        x = x.matmul_rhs_t(&lw.mlp_fc1); // [1, 4*e]
        x = x.relu();
        x = x.matmul_rhs_t(&lw.mlp_fc2); // [1, e]
        x = x.add(&x_res);
    }

    // Output logits: [1, vocab_size]
    x.matmul_rhs_t(&sd.lm_head)
}

/// Forward + softmax → probabilities `[1, vocab_size]`.
pub fn tensor_forward_probs(
    token_id: usize,
    pos_id: usize,
    keys: &mut TensorKvCache,
    vals: &mut TensorKvCache,
    sd: &TensorStateDict,
    n_head: usize,
    n_embd: usize,
) -> Tensor {
    tensor_forward(token_id, pos_id, keys, vals, sd, n_head, n_embd).softmax()
}

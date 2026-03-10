//! Test full transformer layer (attention + MLP) backward.
//! Use actual model to get realistic weights.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_model::TensorModel;

/// Manually compute tensor forward for one position and check gradients.
/// This reproduces tensor_forward.rs line by line.
#[test]
fn layer_by_layer_grad_check() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let token_id = tokens[0];
    let target_id = tokens[1];
    let n_embd = mc.n_embd;
    let n_head = mc.n_head;
    let head_dim = n_embd / n_head;

    // Create identical scalar and tensor models
    let mut rng_s = Rng::new(42);
    let scalar = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let mut rng_t = Rng::new(42);
    let tensor = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    // === SCALAR FORWARD (manually, step by step) ===
    use microgpt_rs::value::Value;
    use microgpt_rs::ops::{linear, rmsnorm, softmax};

    let tok_emb = &scalar.sd.wte[token_id];
    let pos_emb = &scalar.sd.wpe[0];
    let mut sx: Vec<Value> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
    sx = rmsnorm(&sx);

    let lw = &scalar.sd.layers[0];
    let sx_res = sx.clone();
    sx = rmsnorm(&sx);
    let sq = linear(&sx, &lw.attn_wq);
    let sk = linear(&sx, &lw.attn_wk);
    let sv = linear(&sx, &lw.attn_wv);

    // Single position attention: softmax([dot/scale]) = [1.0], output = V
    // For each head: attn_w = [1.0], head_out = V[hs..he]
    let scale = (head_dim as f64).sqrt();
    let mut sx_attn: Vec<Value> = Vec::with_capacity(n_embd);
    for h in 0..n_head {
        let hs = h * head_dim;
        let sq_h = &sq[hs..hs + head_dim];
        let sk_h = &sk[hs..hs + head_dim];
        // dot product
        let dot = sq_h.iter().zip(sk_h)
            .map(|(qi, ki)| qi.mul(ki))
            .reduce(|a, b| a.add(&b)).unwrap()
            .mul_f64(1.0 / scale);
        let attn_w = softmax(&[dot]); // [1.0]
        for j in 0..head_dim {
            sx_attn.push(attn_w[0].mul(&sv[hs + j]));
        }
    }

    sx = linear(&sx_attn, &lw.attn_wo);
    sx = sx.iter().zip(&sx_res).map(|(a, b)| a.add(b)).collect();

    let sx_res = sx.clone();
    sx = rmsnorm(&sx);
    sx = linear(&sx, &lw.mlp_fc1);
    sx = sx.iter().map(|xi| xi.relu()).collect();
    sx = linear(&sx, &lw.mlp_fc2);
    sx = sx.iter().zip(&sx_res).map(|(a, b)| a.add(b)).collect();

    let s_logits = linear(&sx, &scalar.sd.lm_head);
    let s_probs = softmax(&s_logits);
    let s_loss = s_probs[target_id].log().neg();
    s_loss.backward();

    // === TENSOR FORWARD ===
    use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let t_probs = tensor_forward_probs(token_id, 0, &mut tk, &mut tv, &tensor.sd, n_head, n_embd);
    let t_loss = t_probs.nll_loss(target_id);
    t_loss.backward();

    // Compare wpe grads
    let s_wpe_grads: Vec<f64> = scalar.sd.wpe[0].iter().map(|v| v.grad()).collect();
    let t_wpe_grads = tensor.sd.wpe.row_grad(0);

    // Also compare Wo grads (deep in the network)
    let s_wo_grads: Vec<f64> = scalar.sd.layers[0].attn_wo[0].iter().map(|v| v.grad()).collect();
    let t_wo_grads = tensor.sd.layers[0].attn_wo.row_grad(0);
    let max_wo: f64 = s_wo_grads.iter().zip(&t_wo_grads).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max attn_wo[0] grad diff: {max_wo:.2e}");

    // Compare Wv grads
    let s_wv_grads: Vec<f64> = scalar.sd.layers[0].attn_wv[0].iter().map(|v| v.grad()).collect();
    let t_wv_grads = tensor.sd.layers[0].attn_wv.row_grad(0);
    let max_wv: f64 = s_wv_grads.iter().zip(&t_wv_grads).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max attn_wv[0] grad diff: {max_wv:.2e}");

    // Compare Wk grads
    let s_wk_grads: Vec<f64> = scalar.sd.layers[0].attn_wk[0].iter().map(|v| v.grad()).collect();
    let t_wk_grads = tensor.sd.layers[0].attn_wk.row_grad(0);
    let max_wk: f64 = s_wk_grads.iter().zip(&t_wk_grads).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max attn_wk[0] grad diff: {max_wk:.2e}");

    let max_wpe: f64 = s_wpe_grads.iter().zip(&t_wpe_grads).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max wpe[0] grad diff: {max_wpe:.2e}");
    println!("Scalar wpe[0] grad[0..4]: {:?}", &s_wpe_grads[..4]);
    println!("Tensor wpe[0] grad[0..4]: {:?}", &t_wpe_grads[..4]);

    assert!(max_wo < 1e-12, "Wo grads differ: {max_wo}");
    assert!(max_wpe < 1e-12, "wpe grads differ: {max_wpe}");
}

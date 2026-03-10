//! Compare softmax + nll_loss backward between scalar and tensor.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::forward::{forward, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::ops::softmax;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn logit_grads_match() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let target = tokens[1];

    // Scalar: get logits, softmax, nll, backward
    let mut rng_s = Rng::new(42);
    let scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let (mut sk, mut sv) = new_kv_cache(&mc);
    let s_logits = forward(tokens[0], 0, &mut sk, &mut sv, &scalar_model.sd, &mc);
    let s_probs = softmax(&s_logits);
    let s_loss = s_probs[target].log().neg();
    s_loss.backward();

    // Get grad on lm_head (the last weight matrix used to produce logits)
    let s_lm_grad: Vec<f64> = scalar_model.sd.lm_head[0].iter().map(|v| v.grad()).collect();

    // Tensor: same
    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let t_logits = tensor_forward(tokens[0], 0, &mut tk, &mut tv, &tensor_model.sd, mc.n_head, mc.n_embd);
    let t_probs = t_logits.softmax();
    let t_loss = t_probs.nll_loss(target);
    t_loss.backward();

    let t_lm_grad = tensor_model.sd.lm_head.row_grad(0);

    println!("Scalar lm_head[0] grad[0..4]: {:?}", &s_lm_grad[..4]);
    println!("Tensor lm_head[0] grad[0..4]: {:?}", &t_lm_grad[..4]);
    let max_lm: f64 = s_lm_grad.iter().zip(&t_lm_grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max lm_head[0] grad diff: {max_lm}");

    // Get grad on block_output (x before lm_head projection)
    // In scalar: the input x to linear(&x, &sd.lm_head) has grads
    // This is harder to extract... let's check attn_wq instead
    let s_wq_grad: Vec<f64> = scalar_model.sd.layers[0].attn_wq[0].iter().map(|v| v.grad()).collect();
    let t_wq_grad = tensor_model.sd.layers[0].attn_wq.row_grad(0);
    let max_wq: f64 = s_wq_grad.iter().zip(&t_wq_grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max attn_wq[0] grad diff: {max_wq}");

    // Check mlp_fc1
    let s_fc1_grad: Vec<f64> = scalar_model.sd.layers[0].mlp_fc1[0].iter().map(|v| v.grad()).collect();
    let t_fc1_grad = tensor_model.sd.layers[0].mlp_fc1.row_grad(0);
    let max_fc1: f64 = s_fc1_grad.iter().zip(&t_fc1_grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max mlp_fc1[0] grad diff: {max_fc1}");

    // Check wte
    let s_wte_grad: Vec<f64> = scalar_model.sd.wte[0].iter().map(|v| v.grad()).collect();
    let t_wte_grad = tensor_model.sd.wte.row_grad(0);
    let max_wte: f64 = s_wte_grad.iter().zip(&t_wte_grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max wte[0] grad diff: {max_wte}");

    // Check wpe
    let s_wpe_grad: Vec<f64> = scalar_model.sd.wpe[0].iter().map(|v| v.grad()).collect();
    let t_wpe_grad = tensor_model.sd.wpe.row_grad(0);
    let max_wpe: f64 = s_wpe_grad.iter().zip(&t_wpe_grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max wpe[0] grad diff: {max_wpe}");

    assert!(max_lm < 1e-12, "lm_head grads differ: {max_lm}");
    assert!(max_wq < 1e-12, "attn_wq grads differ: {max_wq}");
    assert!(max_fc1 < 1e-12, "mlp_fc1 grads differ: {max_fc1}");
    assert!(max_wte < 1e-12, "wte grads differ: {max_wte}");
    assert!(max_wpe < 1e-12, "wpe grads differ: {max_wpe}");
}

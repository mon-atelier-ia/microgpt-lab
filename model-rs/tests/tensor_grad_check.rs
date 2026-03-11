//! Compare gradients from scalar vs tensor backward.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::forward::{forward_probs, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor::Tensor;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn gradients_match_after_backward() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let n = mc.block_size.min(tokens.len().saturating_sub(1));

    // Scalar: forward + loss + backward
    let mut rng_s = Rng::new(42);
    let scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let (mut sk, mut sv) = new_kv_cache(&mc);
    let mut s_losses = Vec::new();
    for pos in 0..n {
        let probs = forward_probs(tokens[pos], pos, &mut sk, &mut sv, &scalar_model.sd, &mc);
        s_losses.push(probs[tokens[pos + 1]].log().neg());
    }
    let s_loss = s_losses
        .iter()
        .skip(1)
        .fold(s_losses[0].clone(), |a, b| a.add(b))
        .mul_f64(1.0 / n as f64);
    s_loss.backward();

    // Tensor: forward + loss + backward
    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let mut t_losses = Vec::new();
    for pos in 0..n {
        let probs = tensor_forward_probs(
            tokens[pos],
            pos,
            &mut tk,
            &mut tv,
            &tensor_model.sd,
            mc.n_head,
            mc.n_embd,
        );
        t_losses.push(probs.nll_loss(tokens[pos + 1]));
    }
    let t_loss = Tensor::sum_scalars(&t_losses).scale(1.0 / n as f64);
    t_loss.backward();

    // Compare wte[0] gradients (first row of token embedding)
    let s_wte0_grads: Vec<f64> = scalar_model.sd.wte[0].iter().map(|v| v.grad()).collect();
    let t_wte0_grads = tensor_model.sd.wte.row_grad(0);

    println!("Scalar wte[0] grads: {:?}", &s_wte0_grads[..4]);
    println!("Tensor wte[0] grads: {:?}", &t_wte0_grads[..4]);

    let mut max_diff = 0.0_f64;
    for (i, (s, t)) in s_wte0_grads.iter().zip(&t_wte0_grads).enumerate() {
        let diff = (s - t).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-10 {
            println!("  wte[0][{i}]: scalar={s} tensor={t} diff={diff}");
        }
    }
    println!("Max wte[0] grad diff: {max_diff}");

    // Compare lm_head[0] gradients
    let s_lm0_grads: Vec<f64> = scalar_model.sd.lm_head[0]
        .iter()
        .map(|v| v.grad())
        .collect();
    let t_lm0_grads = tensor_model.sd.lm_head.row_grad(0);
    println!("Scalar lm_head[0] grads: {:?}", &s_lm0_grads[..4]);
    println!("Tensor lm_head[0] grads: {:?}", &t_lm0_grads[..4]);

    // Compare attn_wq[0] gradients
    let s_wq0_grads: Vec<f64> = scalar_model.sd.layers[0].attn_wq[0]
        .iter()
        .map(|v| v.grad())
        .collect();
    let t_wq0_grads = tensor_model.sd.layers[0].attn_wq.row_grad(0);
    println!("Scalar attn_wq[0] grads: {:?}", &s_wq0_grads[..4]);
    println!("Tensor attn_wq[0] grads: {:?}", &t_wq0_grads[..4]);

    // Compare wpe[0] gradients
    let s_wpe0_grads: Vec<f64> = scalar_model.sd.wpe[0].iter().map(|v| v.grad()).collect();
    let t_wpe0_grads = tensor_model.sd.wpe.row_grad(0);
    println!("Scalar wpe[0] grads: {:?}", &s_wpe0_grads[..4]);
    println!("Tensor wpe[0] grads: {:?}", &t_wpe0_grads[..4]);
    let max_wpe_diff: f64 = s_wpe0_grads
        .iter()
        .zip(&t_wpe0_grads)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("Max wpe[0] grad diff: {max_wpe_diff}");

    // Compare wpe[1] gradients
    let s_wpe1_grads: Vec<f64> = scalar_model.sd.wpe[1].iter().map(|v| v.grad()).collect();
    let t_wpe1_grads = tensor_model.sd.wpe.row_grad(1);
    println!("Scalar wpe[1] grads: {:?}", &s_wpe1_grads[..4]);
    println!("Tensor wpe[1] grads: {:?}", &t_wpe1_grads[..4]);

    assert!(
        max_diff < 1e-10,
        "wte[0] gradients differ: max_diff={max_diff}"
    );
    assert!(
        max_wpe_diff < 1e-10,
        "wpe[0] gradients differ: max_diff={max_wpe_diff}"
    );
}

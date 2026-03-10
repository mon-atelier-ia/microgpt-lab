//! Check tensor and scalar produce same loss before backward/Adam.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::forward::{forward_probs, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor::Tensor;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn loss_before_backward_matches() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let n = mc.block_size.min(tokens.len().saturating_sub(1));

    // Scalar
    let mut rng_s = Rng::new(42);
    let scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let (mut sk, mut sv) = new_kv_cache(&mc);
    let mut s_losses = Vec::new();
    for pos in 0..n {
        let probs = forward_probs(tokens[pos], pos, &mut sk, &mut sv, &scalar_model.sd, &mc);
        let target = tokens[pos + 1];
        s_losses.push(probs[target].log().neg());
    }
    let s_loss = s_losses.iter().skip(1)
        .fold(s_losses[0].clone(), |a, b| a.add(b))
        .mul_f64(1.0 / n as f64);
    let s_val = s_loss.data();

    // Tensor
    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let mut t_losses = Vec::new();
    for pos in 0..n {
        let probs = tensor_forward_probs(tokens[pos], pos, &mut tk, &mut tv, &tensor_model.sd, mc.n_head, mc.n_embd);
        let target = tokens[pos + 1];
        t_losses.push(probs.nll_loss(target));
    }
    let t_loss = Tensor::sum_scalars(&t_losses).scale(1.0 / n as f64);
    let t_val = t_loss.data()[0];

    println!("Scalar loss: {s_val}");
    println!("Tensor loss: {t_val}");
    println!("n positions: {n}");
    println!("Scalar per-pos losses:");
    for (i, l) in s_losses.iter().enumerate() {
        println!("  pos {i}: {}", l.data());
    }
    println!("Tensor per-pos losses:");
    for (i, l) in t_losses.iter().enumerate() {
        println!("  pos {i}: {}", l.data()[0]);
    }

    let diff = (s_val - t_val).abs();
    assert!(diff < 1e-12, "Loss differs: scalar={s_val} tensor={t_val} diff={diff}");
}

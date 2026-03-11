//! Check single-position gradient match.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::forward::{forward_probs, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;

use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn single_position_grads_match() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);

    // Scalar: single position forward + backward
    let mut rng_s = Rng::new(42);
    let scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let (mut sk, mut sv) = new_kv_cache(&mc);
    let s_probs = forward_probs(tokens[0], 0, &mut sk, &mut sv, &scalar_model.sd, &mc);
    let s_loss = s_probs[tokens[1]].log().neg();
    s_loss.backward();

    // Tensor: single position forward + backward
    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let t_probs = tensor_forward_probs(
        tokens[0],
        0,
        &mut tk,
        &mut tv,
        &tensor_model.sd,
        mc.n_head,
        mc.n_embd,
    );
    let t_loss = t_probs.nll_loss(tokens[1]);
    t_loss.backward();

    // Compare ALL parameter gradients
    let s_params = scalar_model.sd.params();
    let t_params = tensor_model.params();

    let mut s_flat: Vec<f64> = Vec::new();
    for p in &s_params {
        s_flat.push(p.grad());
    }

    let mut t_flat: Vec<f64> = Vec::new();
    for p in t_params {
        t_flat.extend_from_slice(&p.grad());
    }

    assert_eq!(
        s_flat.len(),
        t_flat.len(),
        "Param count mismatch: {} vs {}",
        s_flat.len(),
        t_flat.len()
    );

    let mut max_diff = 0.0_f64;
    let mut max_idx = 0;
    for (i, (s, t)) in s_flat.iter().zip(&t_flat).enumerate() {
        let diff = (s - t).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max grad diff: {max_diff} at index {max_idx}");
    println!("  scalar: {}", s_flat[max_idx]);
    println!("  tensor: {}", t_flat[max_idx]);

    assert!(
        max_diff < 1e-12,
        "Gradients diverge: max_diff={max_diff} at idx={max_idx}"
    );
}

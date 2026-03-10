//! Numerical gradient check for tensor engine.
//! Perturb each wpe element by epsilon, measure loss change, compare with backward grad.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;

fn compute_loss(model: &TensorModel, tokens: &[usize], mc: &ModelConfig) -> f64 {
    let (mut keys, mut vals) = new_tensor_kv_cache(mc.n_layer);
    let probs = tensor_forward_probs(tokens[0], 0, &mut keys, &mut vals, &model.sd, mc.n_head, mc.n_embd);
    let target = tokens[1];
    let p = probs.data()[target];
    -(p.ln())
}

#[test]
fn wpe_numerical_gradient_check() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);

    let mut rng = Rng::new(42);
    let model = TensorModel::new(vocab.size(), &mut rng, mc, &tc);

    // Analytical gradient
    let (mut keys, mut vals) = new_tensor_kv_cache(mc.n_layer);
    let probs = tensor_forward_probs(tokens[0], 0, &mut keys, &mut vals, &model.sd, mc.n_head, mc.n_embd);
    let loss = probs.nll_loss(tokens[1]);
    loss.backward();
    let analytical_grad = model.sd.wpe.row_grad(0);

    // Numerical gradient (central differences)
    let eps = 1e-5;
    let n_embd = mc.n_embd;
    let wpe_data = model.sd.wpe.data();

    for dim in 0..n_embd {
        // +eps
        let mut data_plus = wpe_data.clone();
        data_plus[dim] += eps;
        model.sd.wpe.set_data(&data_plus);
        let loss_plus = compute_loss(&model, &tokens, &mc);

        // -eps
        let mut data_minus = wpe_data.clone();
        data_minus[dim] -= eps;
        model.sd.wpe.set_data(&data_minus);
        let loss_minus = compute_loss(&model, &tokens, &mc);

        // Restore
        model.sd.wpe.set_data(&wpe_data);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = analytical_grad[dim];
        let diff = (numerical - analytical).abs();
        let rel = if analytical.abs() > 1e-8 { diff / analytical.abs() } else { diff };

        println!("wpe[0][{dim}]: analytical={analytical:.8e} numerical={numerical:.8e} diff={diff:.2e} rel={rel:.2e}");
    }
}

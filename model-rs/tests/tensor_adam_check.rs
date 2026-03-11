//! Compare weights after one Adam step.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_model::TensorModel;
use microgpt_rs::tensor_train::tensor_train_step;
use microgpt_rs::train::train_step;

#[test]
fn weights_match_after_one_step() {
    let mc = ModelConfig::default();
    let tc = TrainConfig {
        n_steps: 10,
        ..TrainConfig::default()
    };
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);

    let mut rng_s = Rng::new(42);
    let mut scalar = Model::new(vocab.size(), &mut rng_s, mc, &tc);

    let mut rng_t = Rng::new(42);
    let mut tensor = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    train_step(&mut scalar, &tokens, 0, &tc);
    tensor_train_step(&mut tensor, &tokens, 0, &tc);

    // Compare wte[0]
    let s_wte0: Vec<f64> = scalar.sd.wte[0].iter().map(|v| v.data()).collect();
    let t_wte0 = tensor.sd.wte.row(0);
    println!("After step 0:");
    println!("  Scalar wte[0][0..4]: {:?}", &s_wte0[..4]);
    println!("  Tensor wte[0][0..4]: {:?}", &t_wte0[..4]);
    let max_diff: f64 = s_wte0
        .iter()
        .zip(&t_wte0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("  Max wte[0] diff: {max_diff}");

    // Compare lm_head[0]
    let s_lm0: Vec<f64> = scalar.sd.lm_head[0].iter().map(|v| v.data()).collect();
    let t_lm0 = tensor.sd.lm_head.row(0);
    let max_lm: f64 = s_lm0
        .iter()
        .zip(&t_lm0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("  Max lm_head[0] diff: {max_lm}");

    // Compare wpe[0]
    let s_wpe0: Vec<f64> = scalar.sd.wpe[0].iter().map(|v| v.data()).collect();
    let t_wpe0 = tensor.sd.wpe.row(0);
    let max_wpe: f64 = s_wpe0
        .iter()
        .zip(&t_wpe0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("  Max wpe[0] diff: {max_wpe}");

    assert!(
        max_diff < 1e-14,
        "wte weights diverge after Adam: {max_diff}"
    );
}

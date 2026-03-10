//! Fidelity test: tensor engine must produce identical results to scalar engine.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_model::TensorModel;
use microgpt_rs::train::train_step;
use microgpt_rs::tensor_train::tensor_train_step;

fn small_dataset() -> Vec<&'static str> {
    vec!["emma", "olivia", "liam", "noah"]
}

#[test]
fn tensor_and_scalar_produce_same_loss() {
    let mc = ModelConfig::default();
    let tc = TrainConfig {
        n_steps: 10,
        ..TrainConfig::default()
    };
    let docs = small_dataset();
    let vocab = build_vocab(&docs);

    // Scalar engine
    let mut rng_s = Rng::new(42);
    let mut scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);

    // Tensor engine (same seed → same weights)
    let mut rng_t = Rng::new(42);
    let mut tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    // Train both for 10 steps, compare losses
    for step in 0..10 {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);

        let scalar_loss = train_step(&mut scalar_model, &tokens, step, &tc);
        let tensor_loss = tensor_train_step(&mut tensor_model, &tokens, step, &tc);

        let diff = (scalar_loss - tensor_loss).abs();
        assert!(
            diff < 1e-10,
            "Step {step}: scalar={scalar_loss} tensor={tensor_loss} diff={diff}"
        );
    }
}

#[test]
fn tensor_training_reduces_loss() {
    let mc = ModelConfig::default();
    let tc = TrainConfig {
        n_steps: 50,
        ..TrainConfig::default()
    };
    let docs = small_dataset();
    let vocab = build_vocab(&docs);

    let mut rng = Rng::new(42);
    let mut model = TensorModel::new(vocab.size(), &mut rng, mc, &tc);

    let tokens = tokenize("emma", &vocab, mc.block_size);
    let loss_first = tensor_train_step(&mut model, &tokens, 0, &tc);

    for step in 1..50 {
        let doc = docs[step % docs.len()];
        let toks = tokenize(doc, &vocab, mc.block_size);
        tensor_train_step(&mut model, &toks, step, &tc);
    }

    let loss_last = tensor_train_step(&mut model, &tokens, 50, &tc);
    assert!(
        loss_last < loss_first,
        "Tensor loss should decrease: {loss_first} -> {loss_last}"
    );
}

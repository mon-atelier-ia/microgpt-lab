use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::inference::generate;
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::train::train_step;

fn small_dataset() -> Vec<&'static str> {
    vec!["emma", "olivia", "liam", "noah"]
}

#[test]
fn vocab_uses_bos_only() {
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    // BOS is the last token (id = unique chars count)
    assert_eq!(vocab.bos(), vocab.size() - 1);
    // No EOS — BOS serves as both start and end
    assert!(!vocab.tokens.iter().any(|t| t == "<EOS>"));
}

#[test]
fn tokenize_wraps_with_bos() {
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    let mc = ModelConfig::default();
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let bos = vocab.bos();
    assert_eq!(tokens[0], bos, "First token should be BOS");
    assert_eq!(*tokens.last().unwrap(), bos, "Last token should be BOS");
}

#[test]
fn model_has_separate_lm_head() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    let model = Model::new(vocab.size(), &mut rng, mc, &tc);
    // With separate lm_head, param count > old weight-tied count (3632)
    assert!(
        model.param_count() > 3632,
        "Separate lm_head should add parameters: got {}",
        model.param_count()
    );
}

#[test]
fn training_reduces_loss() {
    let mc = ModelConfig::default();
    let tc = TrainConfig {
        n_steps: 50,
        ..TrainConfig::default()
    };
    let mut rng = Rng::new(42);
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    let mut model = Model::new(vocab.size(), &mut rng, mc, &tc);

    let tokens = tokenize("emma", &vocab, mc.block_size);
    let loss_first = train_step(&mut model, &tokens, 0, &tc);

    for step in 1..50 {
        let doc = docs[step % docs.len()];
        let toks = tokenize(doc, &vocab, mc.block_size);
        train_step(&mut model, &toks, step, &tc);
    }

    let loss_last = train_step(&mut model, &tokens, 50, &tc);
    assert!(
        loss_last < loss_first,
        "Loss should decrease: {loss_first} -> {loss_last}"
    );
}

#[test]
fn generate_produces_output() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    let model = Model::new(vocab.size(), &mut rng, mc, &tc);
    let samples = generate(&model.sd, &vocab, &mut rng, &mc, 3, tc.temperature, "");
    assert_eq!(samples.len(), 3);
}

#[test]
fn generate_with_prefix() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);
    let docs = small_dataset();
    let vocab = build_vocab(&docs);
    let model = Model::new(vocab.size(), &mut rng, mc, &tc);
    let samples = generate(&model.sd, &vocab, &mut rng, &mc, 3, tc.temperature, "em");
    assert_eq!(samples.len(), 3);
    for s in &samples {
        assert!(s.starts_with("em"), "Should start with prefix: got '{s}'");
    }
}

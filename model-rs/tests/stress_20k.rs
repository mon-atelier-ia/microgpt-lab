//! Stress test: 20000 steps with Prénoms FR (50) default config.
//! Reproduce reported OOM/panic.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_model::TensorModel;
use microgpt_rs::tensor_train::tensor_train_step;

const PRENOMS: &[&str] = &[
    "alain",
    "alexandre",
    "andre",
    "anne",
    "bernard",
    "brigitte",
    "catherine",
    "cecile",
    "charles",
    "christiane",
    "christine",
    "claude",
    "daniel",
    "denis",
    "dominique",
    "elisabeth",
    "eric",
    "francois",
    "francoise",
    "gerard",
    "henri",
    "isabelle",
    "jacques",
    "jean",
    "laurent",
    "louis",
    "marcel",
    "marguerite",
    "marie",
    "martine",
    "michel",
    "monique",
    "nathalie",
    "nicolas",
    "patrick",
    "paul",
    "philippe",
    "pierre",
    "raymond",
    "rene",
    "robert",
    "roger",
    "simone",
    "sophie",
    "stephane",
    "sylvie",
    "thierry",
    "thomas",
    "xavier",
    "yves",
];

#[test]
fn stress_20k_prenoms_default_config() {
    let docs: Vec<&str> = PRENOMS.to_vec();
    let vocab = build_vocab(&docs);
    let mc = ModelConfig {
        n_embd: 16,
        n_head: 4,
        n_layer: 1,
        block_size: 16,
    };
    let tc = TrainConfig {
        n_steps: 20000,
        ..TrainConfig::default()
    };
    let mut rng = Rng::new(42);
    let mut model = TensorModel::new(vocab.size(), &mut rng, mc, &tc);

    let mut last_loss = f64::MAX;
    for step in 0..20000 {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        last_loss = tensor_train_step(&mut model, &tokens, step, &tc);

        if step % 5000 == 4999 {
            println!("Step {}: loss={last_loss:.4}", step + 1);
        }

        // Panic early if NaN/Inf
        assert!(
            last_loss.is_finite(),
            "Loss became non-finite at step {step}: {last_loss}"
        );
    }

    println!("20000 steps complete. Final loss: {last_loss:.4}");
}

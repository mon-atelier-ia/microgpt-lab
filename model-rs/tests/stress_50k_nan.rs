//! Push to 50000 steps and check for NaN loss.

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
fn stress_50k_no_nan() {
    let docs: Vec<&str> = PRENOMS.to_vec();
    let vocab = build_vocab(&docs);
    let mc = ModelConfig {
        n_embd: 16,
        n_head: 4,
        n_layer: 1,
        block_size: 16,
    };
    let tc = TrainConfig {
        n_steps: 50000,
        ..TrainConfig::default()
    };
    let mut rng = Rng::new(42);
    let mut model = TensorModel::new(vocab.size(), &mut rng, mc, &tc);

    for step in 0..50000 {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        let loss = tensor_train_step(&mut model, &tokens, step, &tc);

        if !loss.is_finite() {
            panic!("NaN/Inf loss at step {step}: {loss}");
        }
        if step % 10000 == 9999 {
            println!("Step {}: loss={loss:.4}", step + 1);
        }
    }
    println!("50000 steps complete, no NaN.");
}

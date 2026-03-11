//! Check tensor forward produces same logits as scalar forward.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::forward::{forward, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn single_token_logits_match() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);

    let mut rng_s = Rng::new(42);
    let scalar_model = Model::new(vocab.size(), &mut rng_s, mc, &tc);

    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    let tokens = tokenize("emma", &vocab, mc.block_size);

    // Scalar forward
    let (mut sk, mut sv) = new_kv_cache(&mc);
    let s_logits = forward(tokens[0], 0, &mut sk, &mut sv, &scalar_model.sd, &mc);
    let s_vals: Vec<f64> = s_logits.iter().map(|v| v.data()).collect();

    // Tensor forward
    let (mut tk, mut tv) = new_tensor_kv_cache(mc.n_layer);
    let t_logits = tensor_forward(
        tokens[0],
        0,
        &mut tk,
        &mut tv,
        &tensor_model.sd,
        mc.n_head,
        mc.n_embd,
    );
    let t_vals = t_logits.data();

    println!("Scalar logits: {:?}", &s_vals[..5]);
    println!("Tensor logits: {:?}", &t_vals[..5]);

    for (i, (s, t)) in s_vals.iter().zip(&t_vals).enumerate() {
        let diff = (s - t).abs();
        assert!(diff < 1e-12, "Logit {i}: scalar={s} tensor={t} diff={diff}");
    }
}

//! Simulate the exact Ch6 WASM flow: reset → meta → train_step_traced × N

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor::Tensor;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;

#[test]
fn ch6_flow_no_panic() {
    let docs = ["emma", "olivia"];
    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_ref()).collect();
    let vocab = build_vocab(&doc_refs);
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();

    // Constructor flow
    let mut rng = Rng::new(42);
    let _model = Model::new(vocab.size(), &mut rng, mc, &tc);
    let mut rng_t = Rng::new(42);
    let _tensor_model_init = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    // Simulate train(100)
    let mut tensor_model = {
        let mut rng_t2 = Rng::new(42);
        TensorModel::new(vocab.size(), &mut rng_t2, mc, &tc)
    };
    for step in 0..5 {
        let doc = &docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        microgpt_rs::tensor_train::tensor_train_step(&mut tensor_model, &tokens, step, &tc);
    }

    // reset_training flow
    let mut rng2 = Rng::new(42);
    let model2 = Model::new(vocab.size(), &mut rng2, mc, &tc);
    let mut rng_t3 = Rng::new(42);
    let mut tensor_model2 = TensorModel::new(vocab.size(), &mut rng_t3, mc, &tc);

    // training_meta: capture_step reads from scalar model
    let _wte0 = model2.sd.wte[0]
        .iter()
        .map(|v| v.data())
        .collect::<Vec<f64>>();

    // train_step_traced flow × 5 steps
    let mut step_count = 0;
    let mut rng_shuffle = Rng::new(42);
    // skip model creation rng advances
    let _: Vec<usize> = {
        let mut o: Vec<usize> = (0..docs.len()).collect();
        rng_shuffle.shuffle(&mut o);
        o
    };
    for _ in 0..5 {
        let epoch_idx = step_count % docs.len();
        let doc = &docs[epoch_idx];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        let n = mc.block_size.min(tokens.len().saturating_sub(1));
        if n == 0 {
            step_count += 1;
            continue;
        }

        let (mut keys, mut vals) = new_tensor_kv_cache(mc.n_layer);
        let mut losses: Vec<Tensor> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let probs = tensor_forward_probs(
                tokens[pos_id],
                pos_id,
                &mut keys,
                &mut vals,
                &tensor_model2.sd,
                mc.n_head,
                mc.n_embd,
            );
            losses.push(probs.nll_loss(tokens[pos_id + 1]));
        }
        let loss = Tensor::sum_scalars(&losses).scale(1.0 / n as f64);
        loss.backward();
        let loss_val = loss.data()[0];

        // Capture grads
        let wte_grad = tensor_model2.sd.wte.row_grad(0);
        let _wpe_grad = tensor_model2.sd.wpe.row_grad(0);

        // Adam
        tensor_model2.adam_step(step_count, &tc);

        // Sync weights
        let s_params = model2.params();
        let mut offset = 0;
        for t in tensor_model2.params() {
            t.data_ref(|data| {
                for (i, &val) in data.iter().enumerate() {
                    s_params[offset + i].set_data(val);
                }
            });
            offset += t.shape().len();
        }

        // Read post-update values
        let wte_after = tensor_model2.sd.wte.row(0);

        step_count += 1;
        println!(
            "Step {step_count}: loss={loss_val:.4}, wte_grad[0]={:.4e}, wte_after[0]={:.4e}",
            wte_grad[0], wte_after[0]
        );
    }
    assert!(step_count == 5);
}

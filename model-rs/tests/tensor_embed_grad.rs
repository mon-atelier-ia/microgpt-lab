use microgpt_rs::ops::{linear as scalar_linear, rmsnorm as scalar_rmsnorm};
use microgpt_rs::tensor::{Shape, Tensor};
use microgpt_rs::value::Value;

/// Minimal test: embed + rmsnorm + linear + loss, compare scalar vs tensor gradients.
#[test]
fn embed_rmsnorm_linear_grads() {
    let n_embd = 4;
    let vocab_size = 3;
    let token_id = 1;

    // Weight data (deterministic)
    let wte_data: Vec<f64> = (0..vocab_size * n_embd)
        .map(|i| (i as f64 * 0.1) - 0.5)
        .collect();
    let wpe_data: Vec<f64> = (0..n_embd).map(|i| (i as f64 * 0.05) - 0.1).collect();
    let w_data: Vec<f64> = (0..vocab_size * n_embd)
        .map(|i| (i as f64 * 0.07) - 0.3)
        .collect();

    // === SCALAR ===
    let s_wte: Vec<Vec<Value>> = wte_data
        .chunks(n_embd)
        .map(|row| row.iter().map(|&v| Value::new(v)).collect())
        .collect();
    let s_wpe: Vec<Value> = wpe_data.iter().map(|&v| Value::new(v)).collect();
    let s_w: Vec<Vec<Value>> = w_data
        .chunks(n_embd)
        .map(|row| row.iter().map(|&v| Value::new(v)).collect())
        .collect();

    // x = wte[token_id] + wpe
    let s_x: Vec<Value> = s_wte[token_id]
        .iter()
        .zip(&s_wpe)
        .map(|(t, p)| t.add(p))
        .collect();
    let s_xn = scalar_rmsnorm(&s_x);
    let s_logits = scalar_linear(&s_xn, &s_w);
    let s_probs = microgpt_rs::ops::softmax(&s_logits);
    let s_loss = s_probs[0].log().neg(); // target=0
    s_loss.backward();

    let s_wpe_grads: Vec<f64> = s_wpe.iter().map(|v| v.grad()).collect();
    let s_wte_grads: Vec<f64> = s_wte[token_id].iter().map(|v| v.grad()).collect();

    // === TENSOR ===
    let t_wte = Tensor::new(wte_data, Shape(vocab_size, n_embd));
    let t_wpe = Tensor::new(wpe_data.clone(), Shape(1, n_embd)); // [1, n_embd] for single pos
    let t_w = Tensor::new(w_data, Shape(vocab_size, n_embd));

    let t_tok = t_wte.select_row(token_id);
    // wpe is [1, n_embd], same shape as t_tok [1, n_embd]
    let t_x = t_tok.add(&t_wpe);
    let t_xn = t_x.rmsnorm();
    let t_logits = t_xn.matmul_rhs_t(&t_w);
    let t_probs = t_logits.softmax();
    let t_loss = t_probs.nll_loss(0);
    t_loss.backward();

    let t_wpe_grads = t_wpe.grad();
    let t_wte_grads = t_wte.row_grad(token_id);

    println!("Scalar wpe grads: {s_wpe_grads:?}");
    println!("Tensor wpe grads: {t_wpe_grads:?}");
    println!("Scalar wte[{token_id}] grads: {s_wte_grads:?}");
    println!("Tensor wte[{token_id}] grads: {t_wte_grads:?}");

    for (i, (s, t)) in s_wpe_grads.iter().zip(&t_wpe_grads).enumerate() {
        let diff = (s - t).abs();
        assert!(
            diff < 1e-12,
            "wpe grad[{i}]: scalar={s} tensor={t} diff={diff}"
        );
    }
}

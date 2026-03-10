/// Test attention layer backward by comparing gradients on simple inputs.
use microgpt_rs::tensor::{Shape, Tensor};
use microgpt_rs::value::Value;
use microgpt_rs::ops::linear as scalar_linear;

#[test]
fn attention_backward_simple() {
    let n_embd = 4;
    let _n_head = 2;
    let _head_dim = 2;

    // Simple weight matrices [4x4] for Q, K, V, O
    let wq_data = vec![0.1, 0.2, 0.3, 0.4,
                       0.5, 0.6, 0.7, 0.8,
                       0.1, -0.1, 0.2, -0.2,
                       0.3, -0.3, 0.4, -0.4];
    let wo_data = vec![0.05, 0.1, 0.15, 0.2,
                       0.25, 0.3, 0.35, 0.4,
                       0.1, 0.2, 0.3, 0.4,
                       0.5, 0.6, 0.7, 0.8];

    let x_data = vec![0.5, -0.3, 0.8, -0.1];

    // === SCALAR ===
    let s_x: Vec<Value> = x_data.iter().map(|&v| Value::new(v)).collect();
    let s_wq: Vec<Vec<Value>> = wq_data.chunks(n_embd)
        .map(|r| r.iter().map(|&v| Value::new(v)).collect()).collect();
    let s_wo: Vec<Vec<Value>> = wo_data.chunks(n_embd)
        .map(|r| r.iter().map(|&v| Value::new(v)).collect()).collect();

    // Self-attention with 1 position: Q=K=V=linear(x, W)
    let s_q = scalar_linear(&s_x, &s_wq);
    // For single position, attention is just identity (softmax([1.0]) = [1.0])
    // So output = V = Q (using same weights for simplicity)
    // Actually let's just do Q -> scale -> softmax([dot/scale]) -> weighted V
    // With seq_len=1: attn_weights = [1.0] for each head, output = V
    // So x_attn = V = linear(x, Wq) (reusing Wq for V)

    // MHA output = linear(V, Wo) — simplified
    let s_out = scalar_linear(&s_q, &s_wo);
    let s_loss = s_out.iter().skip(1).fold(s_out[0].clone(), |a, b| a.add(b));
    s_loss.backward();
    let s_x_grads: Vec<f64> = s_x.iter().map(|v| v.grad()).collect();

    // === TENSOR ===
    let t_x = Tensor::new(x_data, Shape(1, n_embd));
    let t_wq = Tensor::new(wq_data, Shape(n_embd, n_embd));
    let t_wo = Tensor::new(wo_data, Shape(n_embd, n_embd));

    let t_q = t_x.matmul_rhs_t(&t_wq);
    let t_out = t_q.matmul_rhs_t(&t_wo);
    let ones = Tensor::new(vec![1.0; n_embd], Shape(1, n_embd));
    let t_loss = t_out.matmul_rhs_t(&ones);
    t_loss.backward();
    let t_x_grads = t_x.grad();

    println!("Linear chain (x -> Wq -> Wo -> sum):");
    println!("  Scalar x grads: {s_x_grads:?}");
    println!("  Tensor x grads: {t_x_grads:?}");

    for (i, (s, t)) in s_x_grads.iter().zip(&t_x_grads).enumerate() {
        let diff = (s - t).abs();
        assert!(diff < 1e-12, "grad[{i}]: scalar={s} tensor={t} diff={diff}");
    }
}

#[test]
fn full_attn_with_select_cols() {
    // Test the per-head attention with select_cols
    let n_embd = 4;
    let _n_head = 2;
    let _head_dim = 2;

    let q_data = vec![0.5, -0.3, 0.8, -0.1];

    // Scalar: per-head dot-product attention with seq_len=1
    let s_q: Vec<Value> = q_data.iter().map(|&v| Value::new(v)).collect();
    // With seq_len=1, softmax of single element = 1.0
    // So head output = V[0] for each head
    // But V = Q for simplicity, so each head output = Q[h*d..(h+1)*d]
    // Just sum all Q elements as loss
    let s_loss = s_q.iter().skip(1).fold(s_q[0].clone(), |a, b| a.add(b));
    s_loss.backward();
    let s_grads: Vec<f64> = s_q.iter().map(|v| v.grad()).collect();

    // Tensor: select_cols per head then sum
    let t_q = Tensor::new(q_data, Shape(1, n_embd));
    let h0 = t_q.select_cols(0, 2);
    let h1 = t_q.select_cols(2, 4);
    let cat = Tensor::cat_cols(&[h0, h1]);
    let ones = Tensor::new(vec![1.0; n_embd], Shape(1, n_embd));
    let t_loss = cat.matmul_rhs_t(&ones);
    t_loss.backward();
    let t_grads = t_q.grad();

    println!("select_cols round-trip:");
    println!("  Scalar grads: {s_grads:?}");
    println!("  Tensor grads: {t_grads:?}");

    for (i, (s, t)) in s_grads.iter().zip(&t_grads).enumerate() {
        let diff = (s - t).abs();
        assert!(diff < 1e-12, "grad[{i}]: s={s} t={t} diff={diff}");
    }
}

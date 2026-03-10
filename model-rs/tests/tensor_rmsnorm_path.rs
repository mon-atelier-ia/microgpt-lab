use microgpt_rs::tensor::{Shape, Tensor};
use microgpt_rs::value::Value;
use microgpt_rs::ops::rmsnorm as scalar_rmsnorm;

#[test]
fn select_row_add_rmsnorm_backward() {
    let n_embd = 4;

    let wte_data = vec![
        0.1, 0.2, 0.3, 0.4,  // row 0
        0.5, -0.3, 0.8, -0.1, // row 1
    ];
    let wpe_data = vec![
        -0.1, 0.15, -0.05, 0.2,  // row 0
    ];

    // === SCALAR ===
    let s_wte: Vec<Vec<Value>> = wte_data.chunks(n_embd)
        .map(|r| r.iter().map(|&v| Value::new(v)).collect()).collect();
    let s_wpe: Vec<Value> = wpe_data.iter().map(|&v| Value::new(v)).collect();

    let s_x: Vec<Value> = s_wte[1].iter().zip(&s_wpe).map(|(t, p)| t.add(p)).collect();
    let s_xn = scalar_rmsnorm(&s_x);
    // Simple loss: sum elements
    let s_loss = s_xn.iter().skip(1).fold(s_xn[0].clone(), |a, b| a.add(b));
    s_loss.backward();
    let s_wpe_grads: Vec<f64> = s_wpe.iter().map(|v| v.grad()).collect();
    let s_wte_grads: Vec<f64> = s_wte[1].iter().map(|v| v.grad()).collect();

    // === TENSOR ===
    let t_wte = Tensor::new(wte_data, Shape(2, n_embd));
    let t_wpe = Tensor::new(wpe_data, Shape(1, n_embd));

    let tok = t_wte.select_row(1);
    let x = tok.add(&t_wpe);
    let xn = x.rmsnorm();
    let ones = Tensor::new(vec![1.0; n_embd], Shape(1, n_embd));
    let t_loss = xn.matmul_rhs_t(&ones);
    t_loss.backward();
    let t_wpe_grads = t_wpe.grad();
    let t_wte_grads = t_wte.row_grad(1);

    println!("Scalar wpe grads: {s_wpe_grads:?}");
    println!("Tensor wpe grads: {t_wpe_grads:?}");
    println!("Scalar wte[1] grads: {s_wte_grads:?}");
    println!("Tensor wte[1] grads: {t_wte_grads:?}");

    for (i, (s, t)) in s_wpe_grads.iter().zip(&t_wpe_grads).enumerate() {
        let d = (s - t).abs();
        assert!(d < 1e-12, "wpe grad[{i}] differs: s={s} t={t} d={d}");
    }
}

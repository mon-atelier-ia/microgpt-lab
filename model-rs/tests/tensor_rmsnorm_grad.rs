use microgpt_rs::ops::rmsnorm as scalar_rmsnorm;
use microgpt_rs::tensor::{Shape, Tensor};
use microgpt_rs::value::Value;

#[test]
fn rmsnorm_backward_matches_scalar() {
    let x_data = vec![0.5, -0.3, 0.8, -0.1];

    // Scalar rmsnorm + backward
    let s_x: Vec<Value> = x_data.iter().map(|&v| Value::new(v)).collect();
    let s_out = scalar_rmsnorm(&s_x);
    // Sum outputs to create a scalar loss
    let s_loss = s_out.iter().skip(1).fold(s_out[0].clone(), |a, b| a.add(b));
    s_loss.backward();
    let s_grads: Vec<f64> = s_x.iter().map(|v| v.grad()).collect();

    // Tensor rmsnorm + backward
    let t_x = Tensor::new(x_data.clone(), Shape(1, 4));
    let t_out = t_x.rmsnorm();
    // Same loss: sum all elements
    let ones = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape(1, 4));
    let t_loss = t_out.matmul_rhs_t(&ones); // dot with ones = sum
    t_loss.backward();
    let t_grads = t_x.grad();

    println!("Scalar grads: {s_grads:?}");
    println!("Tensor grads: {t_grads:?}");

    for (i, (s, t)) in s_grads.iter().zip(&t_grads).enumerate() {
        let diff = (s - t).abs();
        assert!(
            diff < 1e-12,
            "rmsnorm grad[{i}]: scalar={s} tensor={t} diff={diff}"
        );
    }
}

#[test]
fn rmsnorm_then_linear_backward() {
    let x_data = vec![0.5, -0.3, 0.8, -0.1];
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    // w is [2, 4]

    // Scalar: rmsnorm(x) then linear
    let s_x: Vec<Value> = x_data.iter().map(|&v| Value::new(v)).collect();
    let s_w: Vec<Vec<Value>> = w_data
        .chunks(4)
        .map(|row| row.iter().map(|&v| Value::new(v)).collect())
        .collect();
    let s_normed = scalar_rmsnorm(&s_x);
    let s_out = microgpt_rs::ops::linear(&s_normed, &s_w);
    let s_loss = s_out[0].add(&s_out[1]);
    s_loss.backward();
    let s_x_grads: Vec<f64> = s_x.iter().map(|v| v.grad()).collect();

    // Tensor: same
    let t_x = Tensor::new(x_data, Shape(1, 4));
    let t_w = Tensor::new(w_data, Shape(2, 4));
    let t_normed = t_x.rmsnorm();
    let t_out = t_normed.matmul_rhs_t(&t_w); // [1,4] @ [4,2] = [1,2]
    let ones = Tensor::new(vec![1.0, 1.0], Shape(1, 2));
    let t_loss = t_out.matmul_rhs_t(&ones);
    t_loss.backward();
    let t_x_grads = t_x.grad();

    println!("rmsnorm+linear scalar x grads: {s_x_grads:?}");
    println!("rmsnorm+linear tensor x grads: {t_x_grads:?}");

    for (i, (s, t)) in s_x_grads.iter().zip(&t_x_grads).enumerate() {
        let diff = (s - t).abs();
        assert!(diff < 1e-12, "grad[{i}]: scalar={s} tensor={t} diff={diff}");
    }
}

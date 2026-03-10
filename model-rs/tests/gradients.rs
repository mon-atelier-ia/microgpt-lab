use microgpt_rs::value::Value;

/// Numerical gradient via central finite differences: (f(x+h) - f(x-h)) / 2h.
fn numerical_grad(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-7;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Assert analytical gradient matches numerical gradient within tolerance.
fn check_grad(analytical: f64, numerical: f64, label: &str) {
    let tol = 1e-5;
    assert!(
        (analytical - numerical).abs() < tol,
        "{label}: analytical={analytical:.8} vs numerical={numerical:.8}"
    );
}

#[test]
fn sub_gradients() {
    let a = Value::new(3.0);
    let b = Value::new(2.0);
    let c = a.sub(&b); // c = a - b = 1.0
    c.backward();

    assert!((c.data() - 1.0).abs() < 1e-12);
    check_grad(a.grad(), numerical_grad(|x| x - 2.0, 3.0), "∂(a-b)/∂a");
    check_grad(b.grad(), numerical_grad(|x| 3.0 - x, 2.0), "∂(a-b)/∂b");
}

#[test]
fn div_gradients() {
    let a = Value::new(6.0);
    let b = Value::new(3.0);
    let c = a.div(&b); // c = a / b = 2.0
    c.backward();

    assert!((c.data() - 2.0).abs() < 1e-12);
    check_grad(a.grad(), numerical_grad(|x| x / 3.0, 6.0), "∂(a/b)/∂a");
    check_grad(b.grad(), numerical_grad(|x| 6.0 / x, 3.0), "∂(a/b)/∂b");
}

#[test]
fn sub_f64_gradient() {
    let a = Value::new(5.0);
    let c = a.sub_f64(2.0); // c = a - 2 = 3.0
    c.backward();

    assert!((c.data() - 3.0).abs() < 1e-12);
    check_grad(a.grad(), numerical_grad(|x| x - 2.0, 5.0), "∂(a-s)/∂a");
}

#[test]
fn div_chain_rule() {
    // Compose: f(a,b) = (a*a) / (b+1)  — tests gradient flow through div.
    let a = Value::new(4.0);
    let b = Value::new(2.0);
    let a2 = a.mul(&a);
    let bp1 = b.add_f64(1.0);
    let c = a2.div(&bp1); // 16 / 3 ≈ 5.333
    c.backward();

    check_grad(
        a.grad(),
        numerical_grad(|x| (x * x) / (2.0 + 1.0), 4.0),
        "∂(a²/(b+1))/∂a",
    );
    check_grad(
        b.grad(),
        numerical_grad(|x| (4.0 * 4.0) / (x + 1.0), 2.0),
        "∂(a²/(b+1))/∂b",
    );
}

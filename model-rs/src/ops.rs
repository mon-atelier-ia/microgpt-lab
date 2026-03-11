use crate::value::Value;

/// y = x @ W^T   x:[inp]  W:[out][inp]  → y:[out]
pub fn linear(x: &[Value], w: &[Vec<Value>]) -> Vec<Value> {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi.mul(xi))
                .reduce(|a, b| a.add(&b))
                .expect("row is always non-empty")
        })
        .collect()
}

/// Numerically stable softmax.
pub fn softmax(logits: &[Value]) -> Vec<Value> {
    let max_val = logits
        .iter()
        .map(|v| v.data())
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value> = logits.iter().map(|v| v.sub_f64(max_val).exp()).collect();
    let total = exps.iter().skip(1).fold(exps[0].clone(), |a, b| a.add(b));
    exps.iter().map(|e| e.div(&total)).collect()
}

/// RMS layer normalization.
pub fn rmsnorm(x: &[Value]) -> Vec<Value> {
    let n = x.len() as f64;
    let ms = x
        .iter()
        .map(|xi| xi.mul(xi))
        .reduce(|a, b| a.add(&b))
        .expect("x is always non-empty")
        .mul_f64(1.0 / n);
    let scale = ms.add_f64(1e-5).pow_f64(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

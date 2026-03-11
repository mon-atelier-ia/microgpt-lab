//! Unit test for select_row backward.

use microgpt_rs::tensor::{Shape, Tensor};

#[test]
fn select_row_gradient_flows() {
    // Matrix [3, 2] with known values
    let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(3, 2));

    // Select row 1 → [3.0, 4.0]
    let r = m.select_row(1);
    assert_eq!(r.data(), vec![3.0, 4.0]);

    // Sum elements of selected row
    let s = Tensor::new(vec![1.0, 1.0], Shape(1, 2));
    let dot = r.matmul_rhs_t(&s); // [1,2] @ [1,2]^T = [1,1] = 3+4 = 7
    assert!((dot.data()[0] - 7.0).abs() < 1e-10);

    dot.backward();

    // Grad should be [0,0, 1,1, 0,0] — only row 1 gets gradient
    let mg = m.grad();
    assert_eq!(
        mg,
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "select_row grad: {mg:?}"
    );
}

#[test]
fn two_select_rows_accumulate() {
    let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(3, 2));

    let r0 = m.select_row(0); // [1, 2]
    let r1 = m.select_row(1); // [3, 4]
    let sum = r0.add(&r1); // [4, 6]

    // Dot with [1, 1] → 10
    let ones = Tensor::new(vec![1.0, 1.0], Shape(1, 2));
    let loss = sum.matmul_rhs_t(&ones);
    loss.backward();

    let mg = m.grad();
    // Row 0 and row 1 should each get grad [1, 1], row 2 = [0, 0]
    assert_eq!(
        mg,
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "two rows grad: {mg:?}"
    );
}

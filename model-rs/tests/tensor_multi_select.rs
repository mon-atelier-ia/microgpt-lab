use microgpt_rs::tensor::{Shape, Tensor};

#[test]
fn multiple_select_cols_accumulate_grads() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape(1, 4));

    // Select two halves
    let h0 = x.select_cols(0, 2); // [1, 2]
    let h1 = x.select_cols(2, 4); // [1, 2]

    // Scale each half differently
    let out0 = h0.scale(2.0);
    let out1 = h1.scale(3.0);

    // Cat and sum
    let cat = Tensor::cat_cols(&[out0, out1]);
    let ones = Tensor::new(vec![1.0; 4], Shape(1, 4));
    let loss = cat.matmul_rhs_t(&ones);
    loss.backward();

    // Expected grad: x[0,1] get 2.0 (from scale 2.0), x[2,3] get 3.0
    let g = x.grad();
    assert_eq!(g, vec![2.0, 2.0, 3.0, 3.0], "grads: {g:?}");
}

//! Tensor autograd engine — reverse-mode autodiff at the matrix level.
//!
//! Drop-in replacement for `value.rs` scalar autograd: same topo-sort
//! backward algorithm, but each node is a 2D tensor (`Vec<f64>` + shape)
//! instead of a single `f64`. ~25 nodes per forward pass vs ~6000.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

// ── Shape helpers ────────────────────────────────────────────────────────

/// 2D shape `[rows, cols]`. Scalar = `[1,1]`, vector = `[1,n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape(pub usize, pub usize);

impl Shape {
    /// Total number of elements.
    pub fn len(self) -> usize {
        self.0 * self.1
    }

    /// Returns true if the shape has zero elements.
    pub fn is_empty(self) -> bool {
        self.0 == 0 || self.1 == 0
    }
}

// ── GradFn — how to propagate gradients through each op ──────────────────

/// Describes how to compute ∂loss/∂input given ∂loss/∂output for one op.
#[derive(Clone)]
enum GradFn {
    /// C = A @ B. grad_A = grad_C @ B^T, grad_B = A^T @ grad_C.
    MatMul {
        lhs_data: Vec<f64>,
        lhs_shape: Shape,
        rhs_data: Vec<f64>,
        rhs_shape: Shape,
    },
    /// C = A @ B^T. grad_A = grad_C @ B, grad_B = grad_C^T @ A.
    MatMulRhsT {
        lhs_data: Vec<f64>,
        lhs_shape: Shape,
        rhs_data: Vec<f64>,
        rhs_shape: Shape,
    },
    /// C = A + B (elementwise, same shape).
    Add,
    /// C = relu(A). mask[i] = (A[i] > 0).
    ReLU { mask: Vec<bool> },
    /// C = rmsnorm(A). Stores input x and inverse-rms scale for backward.
    RmsNorm { x: Vec<f64>, inv_rms: f64 },
    /// C = softmax(A). Stores output probs for Jacobian-vector product.
    Softmax { probs: Vec<f64> },
    /// C = -log(A[target]). Scalar loss from probability vector.
    NllLoss { probs: Vec<f64>, target: usize },
    /// C = A * scalar.
    Scale(f64),
    /// C = A[:, start..start+len]. Column slice of a row vector.
    SelectCols { start: usize, _full_cols: usize },
    /// C = A[row_idx, :]. Single row from a matrix.
    SelectRow { row_idx: usize, _total_rows: usize, cols: usize },
    /// C = stack(rows). Gradient splits back to each row.
    StackRows { _n_rows: usize, cols: usize },
    /// C = cat(tensors, dim=1). Gradient splits back to each tensor.
    CatCols { col_sizes: Vec<usize> },
    // Note: Sum and EmbeddingLookup removed — sum uses Add pass-through,
    // embedding lookup handled at the model level.
}

// ── Tensor node ──────────────────────────────────────────────────────────

struct TensorInner {
    data: Vec<f64>,
    shape: Shape,
    grad: Vec<f64>,
    children: Vec<(Tensor, Option<GradFn>)>,
}

/// A node in the tensor computation graph.
///
/// `Clone` is shallow (Rc refcount).
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

impl Tensor {
    /// Create a leaf tensor (parameter or input).
    pub fn new(data: Vec<f64>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.len(), "data/shape mismatch");
        let n = data.len();
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            shape,
            grad: vec![0.0; n],
            children: vec![],
        })))
    }

    fn with_children(data: Vec<f64>, shape: Shape, children: Vec<(Tensor, Option<GradFn>)>) -> Self {
        let n = data.len();
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            shape,
            grad: vec![0.0; n],
            children,
        })))
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Clone the data buffer.
    pub fn data(&self) -> Vec<f64> {
        self.0.borrow().data.clone()
    }

    /// Borrow data slice without cloning. Callback receives `&[f64]`.
    pub fn data_ref<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        f(&self.0.borrow().data)
    }

    /// Clone the gradient buffer.
    pub fn grad(&self) -> Vec<f64> {
        self.0.borrow().grad.clone()
    }

    /// Borrow grad slice without cloning. Callback receives `&[f64]`.
    pub fn grad_ref<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        f(&self.0.borrow().grad)
    }

    /// Borrow data + grad simultaneously without cloning.
    pub fn with_data_grad<R>(&self, f: impl FnOnce(&[f64], &[f64]) -> R) -> R {
        let inner = self.0.borrow();
        f(&inner.data, &inner.grad)
    }

    /// Apply Adam-style update in-place: callback receives (&mut data, &grad).
    /// Zeros grad after callback returns. Avoids cloning either buffer.
    pub fn adam_update(&self, f: impl FnOnce(&mut [f64], &[f64])) {
        let mut inner = self.0.borrow_mut();
        let inner = &mut *inner;
        f(&mut inner.data, &inner.grad);
        inner.grad.fill(0.0);
    }

    /// Return the 2D shape.
    pub fn shape(&self) -> Shape {
        self.0.borrow().shape
    }

    /// Overwrite the data buffer in-place.
    pub fn set_data(&self, data: &[f64]) {
        self.0.borrow_mut().data.copy_from_slice(data);
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad.fill(0.0);
    }

    /// Read a single row from a 2D tensor. Returns `Vec<f64>` of length `cols`.
    pub fn row(&self, i: usize) -> Vec<f64> {
        let inner = self.0.borrow();
        let cols = inner.shape.1;
        inner.data[i * cols..(i + 1) * cols].to_vec()
    }

    /// Read gradient for a single row.
    pub fn row_grad(&self, i: usize) -> Vec<f64> {
        let inner = self.0.borrow();
        let cols = inner.shape.1;
        inner.grad[i * cols..(i + 1) * cols].to_vec()
    }

    // ── Ops ──────────────────────────────────────────────────────────

    /// Matrix multiply: `self @ rhs`.
    /// self: [M, K], rhs: [K, N] → result: [M, N].
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = rhs.0.borrow();
        let (m, k1) = (a.shape.0, a.shape.1);
        let (k2, n) = (b.shape.0, b.shape.1);
        assert_eq!(k1, k2, "matmul: shape mismatch {k1} vs {k2}");
        let k = k1;

        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a.data[i * k + p] * b.data[p * n + j];
                }
                out[i * n + j] = s;
            }
        }

        Tensor::with_children(out, Shape(m, n), vec![
            (self.clone(), Some(GradFn::MatMul {
                lhs_data: a.data.clone(),
                lhs_shape: a.shape,
                rhs_data: b.data.clone(),
                rhs_shape: b.shape,
            })),
            (rhs.clone(), None), // rhs grad computed inside MatMul handler
        ])
    }

    /// Matrix multiply with transposed RHS: `self @ rhs^T`.
    /// self: [M, K], rhs: [N, K] → result: [M, N].
    /// Gradient flows to both self and rhs (rhs is NOT transposed in storage).
    pub fn matmul_rhs_t(&self, rhs: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = rhs.0.borrow();
        let (m, k1) = (a.shape.0, a.shape.1);
        let (n, k2) = (b.shape.0, b.shape.1);
        assert_eq!(k1, k2, "matmul_rhs_t: inner dim mismatch {k1} vs {k2}");
        let k = k1;

        // C[i,j] = sum_p A[i,p] * B[j,p]  (B is accessed row-major, j-th row)
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a.data[i * k + p] * b.data[j * k + p];
                }
                out[i * n + j] = s;
            }
        }

        Tensor::with_children(out, Shape(m, n), vec![
            (self.clone(), Some(GradFn::MatMulRhsT {
                lhs_data: a.data.clone(),
                lhs_shape: a.shape,
                rhs_data: b.data.clone(),
                rhs_shape: b.shape,
            })),
            (rhs.clone(), None),
        ])
    }

    /// Select a single row from a `[R, C]` matrix, returning `[1, C]`.
    /// Gradient flows back to the correct row of the parent tensor.
    pub fn select_row(&self, row_idx: usize) -> Tensor {
        let inner = self.0.borrow();
        let cols = inner.shape.1;
        let start = row_idx * cols;
        let out: Vec<f64> = inner.data[start..start + cols].to_vec();
        Tensor::with_children(out, Shape(1, cols), vec![
            (self.clone(), Some(GradFn::SelectRow { row_idx, _total_rows: inner.shape.0, cols })),
        ])
    }

    /// Select a contiguous column range from a `[1, N]` row vector.
    /// Returns `[1, end - start]`. Gradient flows back to the correct positions.
    pub fn select_cols(&self, start: usize, end: usize) -> Tensor {
        let inner = self.0.borrow();
        assert_eq!(inner.shape.0, 1, "select_cols: expected row vector");
        let out: Vec<f64> = inner.data[start..end].to_vec();
        let len = end - start;
        Tensor::with_children(out, Shape(1, len), vec![
            (self.clone(), Some(GradFn::SelectCols { start, _full_cols: inner.shape.1 })),
        ])
    }

    /// Elementwise add: `self + rhs` (same shape).
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = rhs.0.borrow();
        assert_eq!(a.shape, b.shape, "add: shape mismatch");
        let out: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        Tensor::with_children(out, a.shape, vec![
            (self.clone(), Some(GradFn::Add)),
            (rhs.clone(), None), // both get grad directly
        ])
    }

    /// Elementwise ReLU.
    pub fn relu(&self) -> Tensor {
        let inner = self.0.borrow();
        let mask: Vec<bool> = inner.data.iter().map(|&x| x > 0.0).collect();
        let out: Vec<f64> = inner.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::with_children(out, inner.shape, vec![
            (self.clone(), Some(GradFn::ReLU { mask })),
        ])
    }

    /// RMS layer normalization (no learnable scale/bias).
    pub fn rmsnorm(&self) -> Tensor {
        let inner = self.0.borrow();
        let n = inner.data.len() as f64;
        let ms: f64 = inner.data.iter().map(|x| x * x).sum::<f64>() / n;
        let inv_rms = (ms + 1e-5_f64).powf(-0.5);
        let out: Vec<f64> = inner.data.iter().map(|&x| x * inv_rms).collect();
        Tensor::with_children(out, inner.shape, vec![
            (self.clone(), Some(GradFn::RmsNorm {
                x: inner.data.clone(),
                inv_rms,
            })),
        ])
    }

    /// Numerically stable softmax (over the full flat data — works for 1D vectors).
    pub fn softmax(&self) -> Tensor {
        let inner = self.0.borrow();
        let max_val = inner.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = inner.data.iter().map(|&x| (x - max_val).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / total).collect();
        Tensor::with_children(probs.clone(), inner.shape, vec![
            (self.clone(), Some(GradFn::Softmax { probs })),
        ])
    }

    /// Negative log-likelihood loss: `-log(self[target])`.
    /// Returns a scalar tensor `[1,1]`.
    pub fn nll_loss(&self, target: usize) -> Tensor {
        let inner = self.0.borrow();
        let p = inner.data[target];
        let loss = -(p.ln());
        Tensor::with_children(vec![loss], Shape(1, 1), vec![
            (self.clone(), Some(GradFn::NllLoss {
                probs: inner.data.clone(),
                target,
            })),
        ])
    }

    /// Scalar multiply: `self * s`.
    pub fn scale(&self, s: f64) -> Tensor {
        let inner = self.0.borrow();
        let out: Vec<f64> = inner.data.iter().map(|&x| x * s).collect();
        Tensor::with_children(out, inner.shape, vec![
            (self.clone(), Some(GradFn::Scale(s))),
        ])
    }

    /// Stack multiple `[1, D]` row vectors into a `[N, D]` matrix.
    /// Gradient flows back to each input row.
    pub fn stack_rows(rows: &[Tensor]) -> Tensor {
        assert!(!rows.is_empty());
        let d = rows[0].shape().1;
        let n = rows.len();
        let mut data = Vec::with_capacity(n * d);
        for r in rows {
            assert_eq!(r.shape(), Shape(1, d), "stack_rows: shape mismatch");
            data.extend_from_slice(&r.data());
        }
        let children: Vec<(Tensor, Option<GradFn>)> = rows
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if i == 0 {
                    (t.clone(), Some(GradFn::StackRows { _n_rows: n, cols: d }))
                } else {
                    (t.clone(), None)
                }
            })
            .collect();
        Tensor::with_children(data, Shape(n, d), children)
    }

    /// Concatenate multiple `[1, Di]` row vectors into `[1, sum(Di)]`.
    /// Gradient flows back to each input via scatter.
    pub fn cat_cols(tensors: &[Tensor]) -> Tensor {
        let total_cols: usize = tensors.iter().map(|t| t.shape().1).sum();
        let mut data = Vec::with_capacity(total_cols);
        let mut col_sizes = Vec::with_capacity(tensors.len());
        for t in tensors {
            assert_eq!(t.shape().0, 1, "cat_cols: expected row vectors");
            col_sizes.push(t.shape().1);
            data.extend_from_slice(&t.data());
        }
        let children: Vec<(Tensor, Option<GradFn>)> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if i == 0 {
                    (t.clone(), Some(GradFn::CatCols { col_sizes: col_sizes.clone() }))
                } else {
                    (t.clone(), None)
                }
            })
            .collect();
        Tensor::with_children(data, Shape(1, total_cols), children)
    }

    /// Sum multiple scalar tensors into one scalar.
    pub fn sum_scalars(tensors: &[Tensor]) -> Tensor {
        let total: f64 = tensors.iter().map(|t| t.0.borrow().data[0]).sum();
        let children: Vec<(Tensor, Option<GradFn>)> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if i == 0 {
                    (t.clone(), Some(GradFn::Add))
                } else {
                    (t.clone(), None)
                }
            })
            .collect();
        Tensor::with_children(vec![total], Shape(1, 1), children)
    }

    // ── Backward ─────────────────────────────────────────────────────

    /// Backpropagate gradients from this tensor through the computation graph.
    pub fn backward(&self) {
        let mut topo: Vec<Tensor> = vec![];
        let mut visited: HashSet<*const RefCell<TensorInner>> = HashSet::new();

        fn build(
            v: &Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const RefCell<TensorInner>>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.insert(ptr) {
                let children: Vec<Tensor> = v.0
                    .borrow()
                    .children
                    .iter()
                    .map(|(c, _)| c.clone())
                    .collect();
                for child in &children {
                    build(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        build(self, &mut topo, &mut visited);

        // Seed gradient.
        {
            let mut inner = self.0.borrow_mut();
            for g in &mut inner.grad {
                *g = 1.0;
            }
        }

        // Reverse topological order.
        for v in topo.iter().rev() {
            let v_grad = v.0.borrow().grad.clone();
            let children: Vec<(Tensor, Option<GradFn>)> = v.0.borrow().children.clone();

            match children.as_slice() {
                // MatMul: children[0] = lhs, children[1] = rhs
                [(lhs, Some(GradFn::MatMul { lhs_data, lhs_shape, rhs_data, rhs_shape })), (rhs, None)] => {
                    let Shape(m, k) = *lhs_shape;
                    let Shape(_, n) = *rhs_shape;

                    // grad_lhs = grad_out @ rhs^T : [M,N] @ [N,K] = [M,K]
                    {
                        let mut lhs_inner = lhs.0.borrow_mut();
                        for i in 0..m {
                            for p in 0..k {
                                let mut s = 0.0;
                                for j in 0..n {
                                    s += v_grad[i * n + j] * rhs_data[p * n + j];
                                }
                                lhs_inner.grad[i * k + p] += s;
                            }
                        }
                    }

                    // grad_rhs = lhs^T @ grad_out : [K,M] @ [M,N] = [K,N]
                    {
                        let mut rhs_inner = rhs.0.borrow_mut();
                        for p in 0..k {
                            for j in 0..n {
                                let mut s = 0.0;
                                for i in 0..m {
                                    s += lhs_data[i * k + p] * v_grad[i * n + j];
                                }
                                rhs_inner.grad[p * n + j] += s;
                            }
                        }
                    }
                }

                // MatMulRhsT: C = A @ B^T. A:[M,K], B:[N,K], C:[M,N]
                // grad_A = grad_C @ B : [M,N] @ [N,K] = [M,K]
                // grad_B = grad_C^T @ A : [N,M] @ [M,K] = [N,K]
                [(lhs, Some(GradFn::MatMulRhsT { lhs_data, lhs_shape, rhs_data, rhs_shape })), (rhs, None)] => {
                    let Shape(m, k) = *lhs_shape;
                    let Shape(n, _) = *rhs_shape;

                    // grad_lhs = grad_out @ rhs : [M,N] @ [N,K] = [M,K]
                    {
                        let mut lhs_inner = lhs.0.borrow_mut();
                        for i in 0..m {
                            for p in 0..k {
                                let mut s = 0.0;
                                for j in 0..n {
                                    s += v_grad[i * n + j] * rhs_data[j * k + p];
                                }
                                lhs_inner.grad[i * k + p] += s;
                            }
                        }
                    }

                    // grad_rhs = grad_out^T @ lhs : [N,M] @ [M,K] = [N,K]
                    {
                        let mut rhs_inner = rhs.0.borrow_mut();
                        for j in 0..n {
                            for p in 0..k {
                                let mut s = 0.0;
                                for i in 0..m {
                                    s += v_grad[i * n + j] * lhs_data[i * k + p];
                                }
                                rhs_inner.grad[j * k + p] += s;
                            }
                        }
                    }
                }

                // Add: both children get grad pass-through
                [(lhs, Some(GradFn::Add)), (rhs, None)] => {
                    {
                        let mut li = lhs.0.borrow_mut();
                        for (g, &vg) in li.grad.iter_mut().zip(&v_grad) {
                            *g += vg;
                        }
                    }
                    {
                        let mut ri = rhs.0.borrow_mut();
                        for (g, &vg) in ri.grad.iter_mut().zip(&v_grad) {
                            *g += vg;
                        }
                    }
                }

                // ReLU
                [(child, Some(GradFn::ReLU { mask }))] => {
                    let mut ci = child.0.borrow_mut();
                    for (i, (g, &vg)) in ci.grad.iter_mut().zip(&v_grad).enumerate() {
                        if mask[i] {
                            *g += vg;
                        }
                    }
                }

                // RmsNorm: ∂(x/rms)/∂x = (1/rms) - x * (x·grad) / (n * rms³)
                [(child, Some(GradFn::RmsNorm { x, inv_rms }))] => {
                    let n = x.len() as f64;
                    let dot: f64 = x.iter().zip(&v_grad).map(|(xi, gi)| xi * gi).sum();
                    let inv_rms3 = inv_rms * inv_rms * inv_rms;
                    let mut ci = child.0.borrow_mut();
                    for (i, (g, &vg)) in ci.grad.iter_mut().zip(&v_grad).enumerate() {
                        *g += vg * inv_rms - x[i] * dot * inv_rms3 / n;
                    }
                }

                // Softmax: Jacobian-vector product
                // ∂L/∂logits[i] = p[i] * (grad[i] - sum(p[j]*grad[j]))
                [(child, Some(GradFn::Softmax { probs }))] => {
                    let dot: f64 = probs.iter().zip(&v_grad).map(|(p, g)| p * g).sum();
                    let mut ci = child.0.borrow_mut();
                    for (i, (g, &vg)) in ci.grad.iter_mut().zip(&v_grad).enumerate() {
                        *g += probs[i] * (vg - dot);
                    }
                }

                // NllLoss: ∂(-log(p[t]))/∂p[i] = -1/p[t] if i==target, else 0
                [(child, Some(GradFn::NllLoss { probs, target }))] => {
                    let mut ci = child.0.borrow_mut();
                    ci.grad[*target] += -1.0 / probs[*target] * v_grad[0];
                }

                // SelectRow: scatter grad back to the correct row
                [(child, Some(GradFn::SelectRow { row_idx, _total_rows: _, cols }))] => {
                    let offset = row_idx * cols;
                    let mut ci = child.0.borrow_mut();
                    for (i, &vg) in v_grad.iter().enumerate() {
                        ci.grad[offset + i] += vg;
                    }
                }

                // SelectCols: scatter grad back to original positions
                [(child, Some(GradFn::SelectCols { start, _full_cols: _ }))] => {
                    let mut ci = child.0.borrow_mut();
                    for (i, &vg) in v_grad.iter().enumerate() {
                        ci.grad[start + i] += vg;
                    }
                }

                // Scale
                [(child, Some(GradFn::Scale(s)))] => {
                    let mut ci = child.0.borrow_mut();
                    for (g, &vg) in ci.grad.iter_mut().zip(&v_grad) {
                        *g += vg * s;
                    }
                }

                // StackRows: split grad [N,D] back to N children of [1,D]
                _ if !children.is_empty() && matches!(&children[0].1, Some(GradFn::StackRows { .. })) => {
                    if let Some(GradFn::StackRows { _n_rows: _, cols }) = &children[0].1 {
                        let d = *cols;
                        for (i, (child, _)) in children.iter().enumerate() {
                            let mut ci = child.0.borrow_mut();
                            for j in 0..d {
                                ci.grad[j] += v_grad[i * d + j];
                            }
                        }
                    }
                }

                // CatCols: split grad [1, sum(Di)] back to children
                _ if !children.is_empty() && matches!(&children[0].1, Some(GradFn::CatCols { .. })) => {
                    if let Some(GradFn::CatCols { col_sizes }) = &children[0].1 {
                        let mut offset = 0;
                        for (i, (child, _)) in children.iter().enumerate() {
                            let sz = col_sizes[i];
                            let mut ci = child.0.borrow_mut();
                            for j in 0..sz {
                                ci.grad[j] += v_grad[offset + j];
                            }
                            offset += sz;
                        }
                    }
                }

                // Sum (1+ children, first has Add marker, rest have None)
                // All scalar children get the output grad
                _ if !children.is_empty() && matches!(&children[0].1, Some(GradFn::Add)) => {
                    for (child, _) in &children {
                        let mut ci = child.0.borrow_mut();
                        for (g, &vg) in ci.grad.iter_mut().zip(&v_grad) {
                            *g += vg;
                        }
                    }
                }

                // Leaf node or unrecognized — skip
                _ => {}
            }
        }
    }
}


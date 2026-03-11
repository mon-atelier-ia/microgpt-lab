//! Scalar autograd engine — reverse-mode autodiff via computation graph.
//!
//! Each `Value` wraps a scalar and tracks its children + local gradients.
//! Calling `backward()` on a loss node traverses the graph in reverse
//! topological order and accumulates gradients into all reachable leaves.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

struct ValueInner {
    data: f64,
    grad: f64,
    children: Vec<(Value, f64)>,
}

/// A node in the computation graph.
///
/// `Clone` is shallow (Rc reference count). `PartialEq` is intentionally
/// not derived — Rc equality would compare pointers, not values, which
/// would be misleading for a numeric type.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        f.debug_struct("Value")
            .field("data", &inner.data)
            .field("grad", &inner.grad)
            .finish()
    }
}

impl Value {
    /// Create a leaf node with the given scalar value.
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children: vec![],
        })))
    }

    fn with_children(data: f64, children: Vec<(Value, f64)>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children,
        })))
    }

    /// Current scalar value.
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    /// Current gradient (accumulated by `backward()`).
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    /// Overwrite the scalar value (used by Adam optimizer).
    pub fn set_data(&self, d: f64) {
        self.0.borrow_mut().data = d;
    }
    /// Reset gradient to zero.
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    // ── Ops ──────────────────────────────────────────────────────────────

    /// Addition: `self + other`.
    pub fn add(&self, other: &Value) -> Value {
        Value::with_children(
            self.data() + other.data(),
            vec![(self.clone(), 1.0), (other.clone(), 1.0)],
        )
    }

    /// Multiplication: `self * other`.
    pub fn mul(&self, other: &Value) -> Value {
        let (sd, od) = (self.data(), other.data());
        Value::with_children(sd * od, vec![(self.clone(), od), (other.clone(), sd)])
    }

    /// Power: `self ^ exp`.
    pub fn pow_f64(&self, exp: f64) -> Value {
        let d = self.data();
        Value::with_children(d.powf(exp), vec![(self.clone(), exp * d.powf(exp - 1.0))])
    }

    /// Natural log: `ln(self)`.
    pub fn log(&self) -> Value {
        let d = self.data();
        Value::with_children(d.ln(), vec![(self.clone(), 1.0 / d)])
    }

    /// Exponential: `e^self`.
    pub fn exp(&self) -> Value {
        let d = self.data();
        let e = d.exp();
        Value::with_children(e, vec![(self.clone(), e)])
    }

    /// ReLU activation: `max(0, self)`.
    pub fn relu(&self) -> Value {
        let d = self.data();
        Value::with_children(
            d.max(0.0),
            vec![(self.clone(), if d > 0.0 { 1.0 } else { 0.0 })],
        )
    }

    /// Negation: `-self`.
    pub fn neg(&self) -> Value {
        self.mul_f64(-1.0)
    }
    /// Subtraction: `self - other`.  ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1.
    pub fn sub(&self, o: &Value) -> Value {
        Value::with_children(
            self.data() - o.data(),
            vec![(self.clone(), 1.0), (o.clone(), -1.0)],
        )
    }
    /// Division: `self / other`.  ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b².
    pub fn div(&self, o: &Value) -> Value {
        let (a, b) = (self.data(), o.data());
        Value::with_children(
            a / b,
            vec![(self.clone(), 1.0 / b), (o.clone(), -a / (b * b))],
        )
    }

    /// Scalar multiplication: `self * s`.
    pub fn mul_f64(&self, s: f64) -> Value {
        Value::with_children(self.data() * s, vec![(self.clone(), s)])
    }

    /// Scalar subtraction: `self - s`.
    pub fn sub_f64(&self, s: f64) -> Value {
        Value::with_children(self.data() - s, vec![(self.clone(), 1.0)])
    }

    /// Scalar addition: `self + s`.
    pub fn add_f64(&self, s: f64) -> Value {
        self.add(&Value::new(s))
    }

    // ── Backward ─────────────────────────────────────────────────────────

    /// Backpropagate gradients from this node through the computation graph.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();

        fn build(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const RefCell<ValueInner>>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.insert(ptr) {
                let children: Vec<Value> =
                    v.0.borrow()
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

        self.0.borrow_mut().grad = 1.0;
        for v in topo.iter().rev() {
            let vg = v.grad();
            let children: Vec<(Value, f64)> = v.0.borrow().children.clone();
            for (child, local_grad) in &children {
                child.0.borrow_mut().grad += local_grad * vg;
            }
        }
    }
}

//! Tensor-level model: weights + Adam state.
//!
//! Wraps `TensorStateDict` and provides initialization + parameter access.
//! Mirrors `model.rs::Model` but for the tensor engine.

use crate::config::{ModelConfig, TrainConfig};
use crate::rng::Rng;
use crate::tensor::{Shape, Tensor};
use crate::tensor_forward::{TensorLayerWeights, TensorStateDict};

/// Complete tensor model: weights + optimizer state.
pub struct TensorModel {
    /// Architecture configuration.
    pub config: ModelConfig,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// All weight tensors.
    pub sd: TensorStateDict,
    /// Flat list of all parameter tensors (for Adam iteration).
    params: Vec<Tensor>,
    /// Adam first moment buffer (flat across all params).
    pub m_buf: Vec<f64>,
    /// Adam second moment buffer.
    pub v_buf: Vec<f64>,
}

fn make_tensor(rows: usize, cols: usize, std: f64, rng: &mut Rng) -> Tensor {
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.gauss(0.0, std)).collect();
    Tensor::new(data, Shape(rows, cols))
}

impl TensorModel {
    /// Create a new model with random weights (same RNG sequence as scalar `Model`).
    pub fn new(vocab_size: usize, rng: &mut Rng, mc: ModelConfig, tc: &TrainConfig) -> Self {
        let e = mc.n_embd;
        let std = tc.std_init;

        let wte = make_tensor(vocab_size, e, std, rng);
        let wpe = make_tensor(mc.block_size, e, std, rng);
        let lm_head = make_tensor(vocab_size, e, std, rng);

        let layers: Vec<TensorLayerWeights> = (0..mc.n_layer)
            .map(|_| TensorLayerWeights {
                attn_wq: make_tensor(e, e, std, rng),
                attn_wk: make_tensor(e, e, std, rng),
                attn_wv: make_tensor(e, e, std, rng),
                attn_wo: make_tensor(e, e, std, rng),
                mlp_fc1: make_tensor(4 * e, e, std, rng),
                mlp_fc2: make_tensor(e, 4 * e, std, rng),
            })
            .collect();

        let sd = TensorStateDict {
            wte,
            wpe,
            lm_head,
            layers,
        };

        // Build flat param list in same order as scalar Model::params()
        let params = Self::collect_params(&sd);
        let n: usize = params.iter().map(|t| t.shape().len()).sum();

        Self {
            config: mc,
            vocab_size,
            sd,
            params,
            m_buf: vec![0.0; n],
            v_buf: vec![0.0; n],
        }
    }

    /// Flat list of all parameter tensors.
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }

    /// Total number of scalar parameters.
    pub fn param_count(&self) -> usize {
        self.params.iter().map(|t| t.shape().len()).sum()
    }

    /// Adam update with linear LR decay (same formula as scalar engine).
    /// Zero-clone: mutates data in-place via `adam_update`.
    pub fn adam_step(&mut self, step: usize, tc: &TrainConfig) {
        let lr_t = (tc.lr * (1.0 - step as f64 / tc.n_steps as f64)).max(0.0);
        let step1 = (step + 1) as f64;

        let mut flat_idx = 0;
        for p in &self.params {
            let m_buf = &mut self.m_buf;
            let v_buf = &mut self.v_buf;
            let fi = flat_idx;
            p.adam_update(|data, grad| {
                for i in 0..data.len() {
                    let g = grad[i];
                    let idx = fi + i;
                    m_buf[idx] = tc.beta1 * m_buf[idx] + (1.0 - tc.beta1) * g;
                    v_buf[idx] = tc.beta2 * v_buf[idx] + (1.0 - tc.beta2) * g * g;
                    let m_hat = m_buf[idx] / (1.0 - tc.beta1.powf(step1));
                    let v_hat = v_buf[idx] / (1.0 - tc.beta2.powf(step1));
                    data[i] -= lr_t * m_hat / (v_hat.sqrt() + tc.eps);
                }
            });
            flat_idx += p.shape().len();
        }
    }

    /// Collect all parameter tensors in deterministic order (same as scalar engine).
    fn collect_params(sd: &TensorStateDict) -> Vec<Tensor> {
        let mut ps = vec![sd.wte.clone(), sd.wpe.clone(), sd.lm_head.clone()];
        for l in &sd.layers {
            ps.push(l.attn_wq.clone());
            ps.push(l.attn_wk.clone());
            ps.push(l.attn_wv.clone());
            ps.push(l.attn_wo.clone());
            ps.push(l.mlp_fc1.clone());
            ps.push(l.mlp_fc2.clone());
        }
        ps
    }
}

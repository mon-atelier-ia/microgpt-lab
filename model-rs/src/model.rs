//! Weight matrices, state dict, and Adam optimizer.

use crate::config::{ModelConfig, TrainConfig};
use crate::rng::Rng;
use crate::value::Value;

/// Weight matrix: `[out_dim][in_dim]` of `Value` nodes.
pub type Matrix = Vec<Vec<Value>>;

fn make_matrix(nout: usize, nin: usize, std: f64, rng: &mut Rng) -> Matrix {
    (0..nout)
        .map(|_| (0..nin).map(|_| Value::new(rng.gauss(0.0, std))).collect())
        .collect()
}

/// Attention and MLP weights for one transformer block.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Query projection `[n_embd][n_embd]`.
    pub attn_wq: Matrix,
    /// Key projection `[n_embd][n_embd]`.
    pub attn_wk: Matrix,
    /// Value projection `[n_embd][n_embd]`.
    pub attn_wv: Matrix,
    /// Output projection `[n_embd][n_embd]`.
    pub attn_wo: Matrix,
    /// MLP first layer `[4*n_embd][n_embd]`.
    pub mlp_fc1: Matrix,
    /// MLP second layer `[n_embd][4*n_embd]`.
    pub mlp_fc2: Matrix,
}

/// All model weights: embeddings, transformer layers, and output head.
#[derive(Debug, Clone)]
pub struct StateDict {
    /// Token embeddings `[vocab_size][n_embd]`.
    pub wte: Matrix,
    /// Position embeddings `[block_size][n_embd]`.
    pub wpe: Matrix,
    /// Output projection `[vocab_size][n_embd]` (no weight tying).
    pub lm_head: Matrix,
    /// Transformer block weights (one per layer).
    pub layers: Vec<LayerWeights>,
}

impl StateDict {
    /// Initialize all weights with gaussian noise at `tc.std_init`.
    pub fn new(vocab_size: usize, rng: &mut Rng, mc: &ModelConfig, tc: &TrainConfig) -> Self {
        let e = mc.n_embd;
        let std = tc.std_init;
        let wte = make_matrix(vocab_size, e, std, rng);
        let wpe = make_matrix(mc.block_size, e, std, rng);
        let lm_head = make_matrix(vocab_size, e, std, rng);
        let layers = (0..mc.n_layer)
            .map(|_| LayerWeights {
                attn_wq: make_matrix(e, e, std, rng),
                attn_wk: make_matrix(e, e, std, rng),
                attn_wv: make_matrix(e, e, std, rng),
                attn_wo: make_matrix(e, e, std, rng),
                mlp_fc1: make_matrix(4 * e, e, std, rng),
                mlp_fc2: make_matrix(e, 4 * e, std, rng),
            })
            .collect();
        StateDict {
            wte,
            wpe,
            lm_head,
            layers,
        }
    }

    /// Flat list of all parameter Values (same order as Python's state_dict).
    pub fn params(&self) -> Vec<Value> {
        let mut ps = vec![];
        let mut add = |m: &Matrix| {
            for row in m {
                for p in row {
                    ps.push(p.clone());
                }
            }
        };
        add(&self.wte);
        add(&self.wpe);
        add(&self.lm_head);
        for l in &self.layers {
            add(&l.attn_wq);
            add(&l.attn_wk);
            add(&l.attn_wv);
            add(&l.attn_wo);
            add(&l.mlp_fc1);
            add(&l.mlp_fc2);
        }
        ps
    }
}

/// Complete model: weights + Adam optimizer state.
#[derive(Debug, Clone)]
pub struct Model {
    /// Architecture configuration.
    pub config: ModelConfig,
    /// Vocabulary size (unique chars + BOS).
    pub vocab_size: usize,
    /// All weight matrices.
    pub sd: StateDict,
    /// Pre-computed flat list of all parameter Values (avoids re-allocation per step).
    params_flat: Vec<Value>,
    /// Adam first moment buffer.
    pub m_buf: Vec<f64>,
    /// Adam second moment buffer.
    pub v_buf: Vec<f64>,
}

impl Model {
    /// Create a new model with random weights.
    pub fn new(vocab_size: usize, rng: &mut Rng, mc: ModelConfig, tc: &TrainConfig) -> Self {
        let sd = StateDict::new(vocab_size, rng, &mc, tc);
        let params_flat = sd.params();
        let n = params_flat.len();
        Self {
            config: mc,
            vocab_size,
            sd,
            params_flat,
            m_buf: vec![0.0; n],
            v_buf: vec![0.0; n],
        }
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.params_flat.len()
    }

    /// Flat list of all parameter Values (pre-computed, no allocation).
    pub fn params(&self) -> &[Value] {
        &self.params_flat
    }

    /// Adam update with linear LR decay.
    pub fn adam_step(&mut self, step: usize, tc: &TrainConfig) {
        let lr_t = (tc.lr * (1.0 - step as f64 / tc.n_steps as f64)).max(0.0);
        let step1 = (step + 1) as f64;
        for i in 0..self.params_flat.len() {
            let g = self.params_flat[i].grad();
            self.m_buf[i] = tc.beta1 * self.m_buf[i] + (1.0 - tc.beta1) * g;
            self.v_buf[i] = tc.beta2 * self.v_buf[i] + (1.0 - tc.beta2) * g * g;
            let m_hat = self.m_buf[i] / (1.0 - tc.beta1.powf(step1));
            let v_hat = self.v_buf[i] / (1.0 - tc.beta2.powf(step1));
            self.params_flat[i]
                .set_data(self.params_flat[i].data() - lr_t * m_hat / (v_hat.sqrt() + tc.eps));
            self.params_flat[i].zero_grad();
        }
    }
}

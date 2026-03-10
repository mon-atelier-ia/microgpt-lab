//! GPT model architecture and training hyperparameters.

/// GPT model architecture parameters.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    /// Embedding dimension (default: 16).
    pub n_embd: usize,
    /// Number of attention heads (default: 4).
    pub n_head: usize,
    /// Number of transformer blocks (default: 1).
    pub n_layer: usize,
    /// Maximum sequence length (default: 16).
    pub block_size: usize,
}

impl ModelConfig {
    /// Per-head dimension: `n_embd / n_head`.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 16,
            n_head: 4,
            n_layer: 1,
            block_size: 16,
        }
    }
}

/// Training and inference hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Number of training iterations (default: 1000).
    pub n_steps: usize,
    /// Initial learning rate with linear decay (default: 0.01).
    pub lr: f64,
    /// Adam first moment decay (default: 0.85).
    pub beta1: f64,
    /// Adam second moment decay (default: 0.99).
    pub beta2: f64,
    /// Adam epsilon for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight initialization standard deviation (default: 0.08).
    pub std_init: f64,
    /// Inference sampling temperature (default: 0.5).
    pub temperature: f64,
    /// Number of names to generate per inference call (default: 5).
    pub n_samples: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_steps: 1000,
            lr: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
            std_init: 0.08,
            temperature: 0.5,
            n_samples: 5,
        }
    }
}

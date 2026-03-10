mod backprop_trace;
mod trace;
mod training_trace;

use std::collections::HashMap;

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize, Vocab};
use microgpt_rs::forward::{forward, new_kv_cache};
use microgpt_rs::model::Model;
use microgpt_rs::ops::softmax;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor::Tensor;
use microgpt_rs::tensor_forward::{new_tensor_kv_cache, tensor_forward_probs};
use microgpt_rs::tensor_model::TensorModel;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use training_trace::{
    build_param_options, capture_step, MatrixKind, TraceOptimizer, TraceStepParam, TrackedParam,
};

/// Install panic hook once so Rust panics show readable stack traces
/// in the browser console instead of "unreachable".
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Serialize)]
struct ConfigResult {
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
    head_dim: usize,
    vocab_size: usize,
}

#[derive(Serialize)]
struct StepResult {
    step: usize,
    loss: f64,
    word: String,
    lr: f64,
}

#[derive(Serialize)]
struct TrainingMeta {
    optimizer: TraceOptimizer,
    parameter_options: Vec<training_trace::TraceParamOption>,
    initial_step: training_trace::TraceStep,
}

fn parse_docs(names_text: &str) -> Vec<String> {
    names_text
        .lines()
        .flat_map(|l| l.split(','))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[wasm_bindgen]
pub struct WasmGpt {
    /// Scalar model — used for forward_trace (Ch4), forward_with_grads (Ch5),
    /// compute_probs (Ch7), and raw weight access (Ch3).
    /// Weights are synced from tensor_model after each training step.
    model: Model,
    /// Tensor model — owns weights, used for training (~100× faster).
    tensor_model: TensorModel,
    vocab: Vocab,
    rng: Rng,
    docs: Vec<String>,
    tc: TrainConfig,
    step_count: usize,
    /// Shuffled order for training (reset each epoch).
    train_order: Vec<usize>,
    /// Cached parameter tracking config (deterministic, depends only on vocab).
    tracked: Vec<TrackedParam>,
    /// True when tensor weights have changed and scalar model needs sync.
    weights_dirty: bool,
}

#[wasm_bindgen]
impl WasmGpt {
    /// Create model from newline-separated (or comma-separated) names.
    #[wasm_bindgen(constructor)]
    pub fn new(names_text: &str) -> Result<WasmGpt, JsError> {
        let docs = parse_docs(names_text);
        if docs.is_empty() {
            return Err(JsError::new("WasmGpt: names_text is empty — provide at least one name"));
        }
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let vocab = build_vocab(&doc_refs);
        let mc = ModelConfig::default();
        let tc = TrainConfig::default();
        let mut rng = Rng::new(42);
        let model = Model::new(vocab.size(), &mut rng, mc, &tc);
        let train_order = Self::shuffled_order(docs.len(), &mut rng);
        let (_, tracked) = build_param_options(&vocab.tokens);
        // Tensor model with same seed → identical initial weights.
        let mut rng_t = Rng::new(42);
        let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
        Ok(WasmGpt {
            model,
            tensor_model,
            vocab,
            rng,
            docs,
            tc,
            step_count: 0,
            train_order,
            tracked,
            weights_dirty: false,
        })
    }

    /// BOS token id.
    pub fn bos(&self) -> usize {
        self.vocab.bos()
    }

    /// Character list (for display). Returns JSON string array.
    pub fn vocab_tokens(&self) -> String {
        serde_json::to_string(&self.vocab.tokens).unwrap_or_default()
    }

    /// Model config as a JS object: { n_embd, n_head, n_layer, block_size, head_dim, vocab_size }.
    pub fn config(&self) -> JsValue {
        let cfg = self.model.config;
        serde_wasm_bindgen::to_value(&ConfigResult {
            n_embd: cfg.n_embd,
            n_head: cfg.n_head,
            n_layer: cfg.n_layer,
            block_size: cfg.block_size,
            head_dim: cfg.head_dim(),
            vocab_size: self.model.vocab_size,
        })
        .unwrap_or(JsValue::NULL)
    }

    // ── Ch7: Inference ──────────────────────────────────────────────

    /// Forward pass on prefix, return probs as Float64Array.
    /// Direct plug-in for useInferenceEngine's ComputeProbs type.
    pub fn compute_probs(&mut self, prefix_ids: &[u32], temperature: f64) -> Result<Vec<f64>, JsError> {
        self.ensure_scalar_synced();
        if prefix_ids.is_empty() {
            return Err(JsError::new("compute_probs: prefix_ids must not be empty"));
        }
        // Clamp temperature to avoid division by zero or negative values.
        let temp = temperature.max(1e-8);

        let cfg = self.model.config;
        let (mut keys, mut vals) = new_kv_cache(&cfg);

        let mut logits = vec![];
        for (pos, &id) in prefix_ids.iter().enumerate() {
            let token_id = id as usize;
            if token_id >= self.model.vocab_size {
                return Err(JsError::new(&format!(
                    "compute_probs: token_id {} out of range (vocab_size={})",
                    token_id, self.model.vocab_size
                )));
            }
            logits = forward(token_id, pos, &mut keys, &mut vals, &self.model.sd, &cfg);
        }

        let scaled: Vec<_> = logits.iter().map(|l| l.mul_f64(1.0 / temp)).collect();
        let probs = softmax(&scaled);
        Ok(probs.iter().map(|p| p.data()).collect())
    }

    // ── Ch4: Attention with intermediates ──────────────────────────

    /// Full forward trace at query_pos for head_idx.
    /// Returns a JS object with all intermediate vectors for Ch4 animation.
    pub fn forward_trace(
        &mut self,
        token_ids: &[u32],
        query_pos: usize,
        head_idx: usize,
    ) -> Result<JsValue, JsError> {
        let cfg = self.model.config;
        if token_ids.is_empty() {
            return Err(JsError::new("forward_trace: token_ids must not be empty"));
        }
        if query_pos >= token_ids.len() {
            return Err(JsError::new(&format!(
                "forward_trace: query_pos {} out of range (seq_len={})",
                query_pos,
                token_ids.len()
            )));
        }
        if head_idx >= cfg.n_head {
            return Err(JsError::new(&format!(
                "forward_trace: head_idx {} out of range (n_head={})",
                head_idx, cfg.n_head
            )));
        }
        self.ensure_scalar_synced();
        let ids: Vec<usize> = token_ids.iter().map(|&id| id as usize).collect();
        for (i, &id) in ids.iter().enumerate() {
            if id >= self.model.vocab_size {
                return Err(JsError::new(&format!(
                    "forward_trace: token_id {} at position {} out of range (vocab_size={})",
                    id, i, self.model.vocab_size
                )));
            }
        }

        let trace = trace::forward_with_trace(&ids, query_pos, head_idx, &self.model.sd, &cfg);
        serde_wasm_bindgen::to_value(&trace)
            .map_err(|e| JsError::new(&format!("forward_trace serialization failed: {e}")))
    }

    // ── Ch5: Backprop with autodiff ────────────────────────────────

    /// Forward + backward at `selected_pos`, returning per-position
    /// embedding gradients via Rust autodiff.
    ///
    /// Uses scalar engine (value.rs) — weights are synced from tensor model.
    pub fn forward_with_grads(
        &mut self,
        token_ids: &[u32],
        selected_pos: usize,
    ) -> Result<JsValue, JsError> {
        self.ensure_scalar_synced();
        let ids: Vec<usize> = token_ids.iter().map(|&id| id as usize).collect();
        for (i, &id) in ids.iter().enumerate() {
            if id >= self.model.vocab_size {
                return Err(JsError::new(&format!(
                    "forward_with_grads: token_id {} at position {} out of range (vocab_size={})",
                    id, i, self.model.vocab_size
                )));
            }
        }

        let result = backprop_trace::compute_backprop_trace(&self.model, &ids, selected_pos)
            .map_err(|e| JsError::new(&format!("forward_with_grads: {e}")))?;

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsError::new(&format!("forward_with_grads serialization failed: {e}")))
    }

    // ── Ch3: Raw weight access ──────────────────────────────────────

    /// Get wte row for token_id (reads from tensor model).
    pub fn wte_row(&self, token_id: usize) -> Result<Vec<f64>, JsError> {
        if token_id >= self.tensor_model.vocab_size {
            return Err(JsError::new(&format!(
                "wte_row: token_id {} out of range (vocab_size={})",
                token_id, self.tensor_model.vocab_size
            )));
        }
        Ok(self.tensor_model.sd.wte.row(token_id))
    }

    /// Get wpe row for pos_id (reads from tensor model).
    pub fn wpe_row(&self, pos_id: usize) -> Result<Vec<f64>, JsError> {
        if pos_id >= self.tensor_model.config.block_size {
            return Err(JsError::new(&format!(
                "wpe_row: pos_id {} out of range (block_size={})",
                pos_id, self.tensor_model.config.block_size
            )));
        }
        Ok(self.tensor_model.sd.wpe.row(pos_id))
    }

    /// Get lm_head row for token_id. Used by Ch5 for gradient display.
    pub fn lm_head_row(&self, token_id: usize) -> Result<Vec<f64>, JsError> {
        if token_id >= self.tensor_model.vocab_size {
            return Err(JsError::new(&format!(
                "lm_head_row: token_id {} out of range (vocab_size={})",
                token_id, self.tensor_model.vocab_size
            )));
        }
        Ok(self.tensor_model.sd.lm_head.row(token_id))
    }

    // ── Batch training (no trace) ───────────────────────────────────

    /// Batch-train the model for `n_steps` without capturing trace data.
    /// Uses tensor engine (~100× faster than scalar).
    /// After this call, the model is trained and ready for inference/forward_trace.
    pub fn train(&mut self, n_steps: usize) {
        let cfg = self.tensor_model.config;
        for _ in 0..n_steps {
            let epoch_idx = self.step_count % self.docs.len();
            if epoch_idx == 0 {
                self.train_order = Self::shuffled_order(self.docs.len(), &mut self.rng);
            }
            let doc_idx = self.train_order[epoch_idx];
            let doc = self.docs[doc_idx].clone();
            let tokens = tokenize(&doc, &self.vocab, cfg.block_size);
            microgpt_rs::tensor_train::tensor_train_step(
                &mut self.tensor_model,
                &tokens,
                self.step_count,
                &self.tc,
            );
            self.step_count += 1;
        }
        self.weights_dirty = true;
    }

    // ── Ch6: Live training ──────────────────────────────────────────

    /// Run one training step using tensor engine.
    /// Returns JS object: { step, loss, word, lr }.
    pub fn train_step(&mut self) -> JsValue {
        let epoch_idx = self.step_count % self.docs.len();
        if epoch_idx == 0 {
            self.train_order = Self::shuffled_order(self.docs.len(), &mut self.rng);
        }
        let doc_idx = self.train_order[epoch_idx];
        let doc = self.docs[doc_idx].clone();
        let cfg = self.tensor_model.config;
        let tokens = tokenize(&doc, &self.vocab, cfg.block_size);
        let loss = microgpt_rs::tensor_train::tensor_train_step(
            &mut self.tensor_model,
            &tokens,
            self.step_count,
            &self.tc,
        );
        let lr_t = (self.tc.lr * (1.0 - self.step_count as f64 / self.tc.n_steps as f64)).max(0.0);
        self.step_count += 1;
        self.weights_dirty = true;

        serde_wasm_bindgen::to_value(&StepResult {
            step: self.step_count,
            loss,
            word: doc,
            lr: lr_t,
        })
        .unwrap_or(JsValue::NULL)
    }

    // ── Ch6: Traced training step (streaming) ──────────────────────

    /// Run one training step with full gradient + parameter capture.
    /// Uses tensor engine for forward/backward/Adam.
    pub fn train_step_traced(&mut self) -> Result<JsValue, JsError> {
        let epoch_idx = self.step_count % self.docs.len();
        if epoch_idx == 0 {
            self.train_order = Self::shuffled_order(self.docs.len(), &mut self.rng);
        }
        let doc_idx = self.train_order[epoch_idx];
        let doc = self.docs[doc_idx].clone();
        let cfg = self.tensor_model.config;
        let tokens = tokenize(&doc, &self.vocab, cfg.block_size);

        let n = cfg.block_size.min(tokens.len().saturating_sub(1));
        if n == 0 {
            self.step_count += 1;
            return Ok(JsValue::NULL);
        }

        // Forward + loss using tensor engine.
        let (mut keys, mut vals) = new_tensor_kv_cache(cfg.n_layer);
        let mut losses: Vec<Tensor> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let probs = tensor_forward_probs(
                tokens[pos_id],
                pos_id,
                &mut keys,
                &mut vals,
                &self.tensor_model.sd,
                cfg.n_head,
                cfg.n_embd,
            );
            losses.push(probs.nll_loss(tokens[pos_id + 1]));
        }
        let loss = Tensor::sum_scalars(&losses).scale(1.0 / n as f64);
        loss.backward();
        let loss_val = loss.data()[0];
        let lr_t = (self.tc.lr * (1.0 - self.step_count as f64 / self.tc.n_steps as f64)).max(0.0);

        // Capture grads BEFORE adam_step zeros them.
        let mut grad_snap: HashMap<String, Vec<f64>> = HashMap::with_capacity(self.tracked.len());
        for tp in &self.tracked {
            let grads = Self::tensor_row_grad(&self.tensor_model, &tp.kind, tp.row_index);
            grad_snap.insert(tp.id.clone(), grads);
        }

        // Adam update (zeros grads, updates data).
        self.tensor_model.adam_step(self.step_count, &self.tc);
        self.weights_dirty = true;

        // Capture param values AFTER update.
        let mut params = HashMap::with_capacity(self.tracked.len());
        for tp in &self.tracked {
            let after = Self::tensor_row_data(&self.tensor_model, &tp.kind, tp.row_index);
            params.insert(
                tp.id.clone(),
                TraceStepParam {
                    grad: grad_snap.remove(&tp.id).unwrap_or_default(),
                    after,
                },
            );
        }

        self.step_count += 1;

        let step = training_trace::TraceStep {
            step: self.step_count,
            word: doc,
            loss: Some(loss_val),
            learning_rate: lr_t,
            params,
        };

        serde_wasm_bindgen::to_value(&step)
            .map_err(|e| JsError::new(&format!("train_step_traced serialization: {e}")))
    }

    /// Return training metadata: optimizer config + tracked parameter list
    /// + initial step (step 0, pre-training state).
    pub fn training_meta(&mut self) -> Result<JsValue, JsError> {
        self.ensure_scalar_synced();
        let (param_options, _) = build_param_options(&self.vocab.tokens);
        let initial = capture_step(&self.model, &self.tracked, 0, "", None, self.tc.lr, false);
        let meta = TrainingMeta {
            optimizer: TraceOptimizer {
                name: "Adam".into(),
                beta1: self.tc.beta1,
                beta2: self.tc.beta2,
                eps: self.tc.eps,
                base_learning_rate: self.tc.lr,
                schedule: "linear_decay(lr_t = lr * (1 - step / num_steps))".into(),
            },
            parameter_options: param_options,
            initial_step: initial,
        };
        serde_wasm_bindgen::to_value(&meta)
            .map_err(|e| JsError::new(&format!("training_meta serialization: {e}")))
    }

    /// Reset model weights for streaming re-training.
    pub fn reset_training(&mut self) {
        self.rng = Rng::new(42);
        self.model = Model::new(
            self.vocab.size(),
            &mut self.rng,
            ModelConfig::default(),
            &self.tc,
        );
        // Advance rng past the constructor's shuffled_order call.
        let _ = Self::shuffled_order(self.docs.len(), &mut self.rng);
        // Recreate tensor model with same seed.
        let mut rng_t = Rng::new(42);
        self.tensor_model = TensorModel::new(
            self.vocab.size(),
            &mut rng_t,
            ModelConfig::default(),
            &self.tc,
        );
        self.step_count = 0;
        self.train_order = Self::shuffled_order(self.docs.len(), &mut self.rng);
        let (_, tracked) = build_param_options(&self.vocab.tokens);
        self.tracked = tracked;
        self.weights_dirty = false;
    }

    /// Reset model with new dataset.
    pub fn reset(&mut self, names_text: &str) -> Result<(), JsError> {
        let docs = parse_docs(names_text);
        if docs.is_empty() {
            return Err(JsError::new("reset: names_text is empty — provide at least one name"));
        }
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        self.vocab = build_vocab(&doc_refs);
        self.model = Model::new(
            self.vocab.size(),
            &mut self.rng,
            ModelConfig::default(),
            &self.tc,
        );
        // Tensor model with fresh RNG (same seed as scalar model's rng state
        // won't match, but that's fine — reset() starts fresh training anyway).
        let mut rng_t = Rng::new(self.rng.next_u64());
        self.tensor_model = TensorModel::new(
            self.vocab.size(),
            &mut rng_t,
            ModelConfig::default(),
            &self.tc,
        );
        // Sync so both models have identical weights.
        Self::sync_weights(&self.tensor_model, &self.model);
        self.train_order = Self::shuffled_order(docs.len(), &mut self.rng);
        let (_, tracked) = build_param_options(&self.vocab.tokens);
        self.tracked = tracked;
        self.docs = docs;
        self.step_count = 0;
        self.weights_dirty = false;
        Ok(())
    }
}

impl WasmGpt {
    /// Sync tensor → scalar only when needed (lazy).
    fn ensure_scalar_synced(&mut self) {
        if self.weights_dirty {
            Self::sync_weights(&self.tensor_model, &self.model);
            self.weights_dirty = false;
        }
    }

    fn shuffled_order(len: usize, rng: &mut Rng) -> Vec<usize> {
        let mut order: Vec<usize> = (0..len).collect();
        rng.shuffle(&mut order);
        order
    }

    /// Copy all tensor weights → scalar model Values.
    /// Both models have parameters in the same flat order.
    /// Zero-clone: borrows tensor data via `data_ref`.
    fn sync_weights(tensor: &TensorModel, scalar: &Model) {
        let s_params = scalar.params();
        let mut offset = 0;
        for t in tensor.params() {
            t.data_ref(|data| {
                for (i, &val) in data.iter().enumerate() {
                    s_params[offset + i].set_data(val);
                }
            });
            offset += t.shape().len();
        }
    }

    /// Read gradient for a tracked parameter row from tensor model.
    fn tensor_row_grad(tm: &TensorModel, kind: &MatrixKind, row_index: usize) -> Vec<f64> {
        match kind {
            MatrixKind::Wte => tm.sd.wte.row_grad(row_index),
            MatrixKind::Wpe => tm.sd.wpe.row_grad(row_index),
            MatrixKind::LmHead => tm.sd.lm_head.row_grad(row_index),
            MatrixKind::AttnWq => tm.sd.layers[0].attn_wq.row_grad(row_index),
        }
    }

    /// Read data for a tracked parameter row from tensor model.
    fn tensor_row_data(tm: &TensorModel, kind: &MatrixKind, row_index: usize) -> Vec<f64> {
        match kind {
            MatrixKind::Wte => tm.sd.wte.row(row_index),
            MatrixKind::Wpe => tm.sd.wpe.row(row_index),
            MatrixKind::LmHead => tm.sd.lm_head.row(row_index),
            MatrixKind::AttnWq => tm.sd.layers[0].attn_wq.row(row_index),
        }
    }
}

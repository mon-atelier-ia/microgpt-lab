use serde::Deserialize;
use wasm_bindgen_test::*;

const TEST_NAMES: &str = "alice\nbob\ncharlie\ndiana\neve";

#[derive(Deserialize)]
struct Config {
    n_embd: usize,
    n_head: usize,
    block_size: usize,
    vocab_size: usize,
}

#[derive(Deserialize)]
struct TrainStepResult {
    step: usize,
    loss: f64,
    word: String,
    lr: f64,
}

#[derive(Deserialize)]
struct ForwardTrace {
    x_vectors: Vec<Vec<f64>>,
    q_vector: Vec<f64>,
    k_rows: Vec<Vec<f64>>,
    v_rows: Vec<Vec<f64>>,
    attn_weights: Vec<f64>,
    attn_output: Vec<f64>,
    head_outputs: Vec<Vec<f64>>,
    x_attn_vector: Vec<f64>,
    mha_output: Vec<f64>,
    attn_block_result: Vec<f64>,
    block_output: Vec<f64>,
    logits: Vec<f64>,
    probs: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn new_creates_model() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    assert!(gpt.bos() > 0); // BOS is always the last token
}

#[wasm_bindgen_test]
fn new_rejects_empty() {
    let result = microgpt_wasm::WasmGpt::new("");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn vocab_tokens_returns_valid_json() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let json = gpt.vocab_tokens();
    let tokens: Vec<String> = serde_json::from_str(&json).unwrap();
    // Must contain at least the unique chars + BOS
    assert!(tokens.len() > 5);
    assert!(tokens.contains(&"<BOS>".to_string()));
}

#[wasm_bindgen_test]
fn config_returns_valid_object() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let cfg: Config = serde_wasm_bindgen::from_value(gpt.config()).unwrap();
    assert_eq!(cfg.n_embd, 16);
    assert_eq!(cfg.n_head, 4);
    assert_eq!(cfg.block_size, 16);
    assert!(cfg.vocab_size > 5);
}

// ---------------------------------------------------------------------------
// compute_probs — probability distribution validity
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn compute_probs_sums_to_one() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt.bos() as u32;
    let probs = gpt.compute_probs(&[bos], 1.0).unwrap();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probs should sum to 1.0, got {sum}"
    );
    assert!(probs.iter().all(|&p| p >= 0.0), "all probs must be >= 0");
}

#[wasm_bindgen_test]
fn compute_probs_temperature_affects_entropy() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt.bos() as u32;

    let probs_cold = gpt.compute_probs(&[bos], 0.1).unwrap();
    let probs_hot = gpt.compute_probs(&[bos], 2.0).unwrap();

    // Cold temperature → lower entropy (more peaked)
    let entropy_cold: f64 = probs_cold
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let entropy_hot: f64 = probs_hot
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();

    assert!(
        entropy_cold < entropy_hot,
        "cold temp ({entropy_cold}) should have lower entropy than hot ({entropy_hot})"
    );
}

#[wasm_bindgen_test]
fn compute_probs_rejects_empty_prefix() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let result = gpt.compute_probs(&[], 1.0);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn compute_probs_rejects_oob_token() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let result = gpt.compute_probs(&[9999], 1.0);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// compute_probs — determinism (same seed → same output)
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn compute_probs_is_deterministic() {
    let mut gpt1 = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let mut gpt2 = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt1.bos() as u32;

    let probs1 = gpt1.compute_probs(&[bos], 1.0).unwrap();
    let probs2 = gpt2.compute_probs(&[bos], 1.0).unwrap();

    assert_eq!(probs1.len(), probs2.len());
    for (i, (a, b)) in probs1.iter().zip(&probs2).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "probs differ at index {i}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// wte_row / wpe_row — bounds and shape
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn wte_row_returns_n_embd_values() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let row = gpt.wte_row(0).unwrap();
    assert_eq!(row.len(), 16); // n_embd = 16
}

#[wasm_bindgen_test]
fn wte_row_rejects_oob() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    assert!(gpt.wte_row(9999).is_err());
}

#[wasm_bindgen_test]
fn wpe_row_returns_n_embd_values() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let row = gpt.wpe_row(0).unwrap();
    assert_eq!(row.len(), 16);
}

#[wasm_bindgen_test]
fn wpe_row_rejects_oob() {
    let gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    assert!(gpt.wpe_row(9999).is_err());
}

// ---------------------------------------------------------------------------
// forward_trace — shape validation
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn forward_trace_returns_correct_shapes() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt.bos() as u32;
    let tokens = gpt.vocab_tokens();
    let token_list: Vec<String> = serde_json::from_str(&tokens).unwrap();
    let a_id = token_list.iter().position(|t| t == "a").unwrap() as u32;

    let trace: ForwardTrace =
        serde_wasm_bindgen::from_value(gpt.forward_trace(&[bos, a_id], 1, 0).unwrap()).unwrap();

    // seq_len = 2 (BOS + 'a')
    assert_eq!(trace.x_vectors.len(), 2);
    assert_eq!(trace.x_vectors[0].len(), 16); // n_embd

    // head_dim = 4
    assert_eq!(trace.q_vector.len(), 4);
    assert_eq!(trace.k_rows.len(), 2); // seq_len
    assert_eq!(trace.k_rows[0].len(), 4); // head_dim
    assert_eq!(trace.v_rows.len(), 2);
    assert_eq!(trace.attn_weights.len(), 2); // seq_len
    assert_eq!(trace.attn_output.len(), 4); // head_dim

    // n_head = 4
    assert_eq!(trace.head_outputs.len(), 4);
    assert_eq!(trace.head_outputs[0].len(), 4); // head_dim

    // n_embd = 16
    assert_eq!(trace.x_attn_vector.len(), 16);
    assert_eq!(trace.mha_output.len(), 16);
    assert_eq!(trace.attn_block_result.len(), 16);
    assert_eq!(trace.block_output.len(), 16);

    // vocab_size
    assert!(trace.logits.len() > 5);
    assert_eq!(trace.probs.len(), trace.logits.len());

    // Probs should sum to ~1
    let sum: f64 = trace.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn forward_trace_rejects_oob_query_pos() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt.bos() as u32;
    assert!(gpt.forward_trace(&[bos], 5, 0).is_err());
}

#[wasm_bindgen_test]
fn forward_trace_rejects_oob_head_idx() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let bos = gpt.bos() as u32;
    assert!(gpt.forward_trace(&[bos], 0, 99).is_err());
}

// ---------------------------------------------------------------------------
// train_step — loss decreases over training
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn train_step_returns_step_result() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let result: TrainStepResult =
        serde_wasm_bindgen::from_value(gpt.train_step().unwrap()).unwrap();
    assert_eq!(result.step, 1);
    assert!(result.loss > 0.0, "initial loss should be positive");
    assert!(result.lr > 0.0, "lr should be positive");
    assert!(!result.word.is_empty(), "word should not be empty");
}

#[wasm_bindgen_test]
fn train_reduces_loss() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();

    let r1: TrainStepResult = serde_wasm_bindgen::from_value(gpt.train_step().unwrap()).unwrap();
    let loss_1 = r1.loss;

    // Train 50 more steps
    for _ in 0..50 {
        gpt.train_step().unwrap();
    }

    let r_last: TrainStepResult =
        serde_wasm_bindgen::from_value(gpt.train_step().unwrap()).unwrap();
    let loss_last = r_last.loss;

    assert!(
        loss_last < loss_1,
        "loss should decrease: {loss_1} → {loss_last}"
    );
}

// ---------------------------------------------------------------------------
// reset — model reinitializes correctly
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn reset_changes_vocab() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    let tokens_before = gpt.vocab_tokens();

    gpt.reset("xyz\nwww").unwrap();
    let tokens_after = gpt.vocab_tokens();

    assert_ne!(
        tokens_before, tokens_after,
        "vocab should change after reset"
    );
}

#[wasm_bindgen_test]
fn reset_rejects_empty() {
    let mut gpt = microgpt_wasm::WasmGpt::new(TEST_NAMES).unwrap();
    assert!(gpt.reset("").is_err());
}

// ---------------------------------------------------------------------------
// new_with_config — configurable constructor
// ---------------------------------------------------------------------------

#[test]
fn test_new_with_config_custom_params() {
    let gpt = microgpt_wasm::WasmGpt::new_with_config("alice\nbob\ncharlie", 8, 2, 1, 8)
        .expect("should create with custom config");
    assert_eq!(gpt.model_config().n_embd, 8);
    assert_eq!(gpt.model_config().n_head, 2);
    assert_eq!(gpt.model_config().n_layer, 1);
    assert_eq!(gpt.model_config().block_size, 8);
}

#[test]
fn test_new_with_config_invalid_head_dim() {
    let result = microgpt_wasm::WasmGpt::new_with_config("alice\nbob", 8, 3, 1, 16);
    assert!(result.is_err(), "n_embd=8, n_head=3 should fail (8%3!=0)");
}

// ---------------------------------------------------------------------------
// set_lr / current_lr
// ---------------------------------------------------------------------------

#[test]
fn test_set_lr() {
    let mut gpt = microgpt_wasm::WasmGpt::new("alice\nbob\ncharlie").expect("create");
    gpt.set_lr(0.05);
    assert!((gpt.current_lr() - 0.05).abs() < 1e-10);
}

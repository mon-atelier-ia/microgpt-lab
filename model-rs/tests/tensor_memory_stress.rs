//! Stress test: verify memory doesn't grow linearly across training steps.
//!
//! Runs 3000 tensor training steps and checks that peak allocator usage
//! stays bounded (no Rc cycle leaks, GradFn data freed after backward).

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor_model::TensorModel;
use microgpt_rs::tensor_train::tensor_train_step;

#[cfg(target_os = "windows")]
fn resident_bytes() -> usize {
    // On Windows, use GetProcessMemoryInfo via winapi-compatible approach.
    // Fallback: not available in std. Return 0 to skip memory assertion.
    0
}

#[cfg(not(target_os = "windows"))]
fn resident_bytes() -> usize {
    // Read RSS from /proc/self/statm (Linux) — field 1 × page size.
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[test]
fn tensor_3000_steps_no_oom() {
    let docs: Vec<&str> = vec![
        "tyrannosaurus",
        "triceratops",
        "velociraptor",
        "stegosaurus",
        "brachiosaurus",
        "pteranodon",
        "diplodocus",
        "allosaurus",
    ];
    let vocab = build_vocab(&docs);
    let mc = ModelConfig {
        n_embd: 32,
        n_head: 1,
        n_layer: 1,
        block_size: 32,
    };
    let tc = TrainConfig {
        n_steps: 5000,
        ..TrainConfig::default()
    };
    let mut rng = Rng::new(42);
    let mut model = TensorModel::new(vocab.size(), &mut rng, mc, &tc);

    // Warm up: run 100 steps to let allocator settle.
    for step in 0..100 {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        tensor_train_step(&mut model, &tokens, step, &tc);
    }

    let mem_after_warmup = resident_bytes();

    // Run 2900 more steps.
    for step in 100..3000 {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        tensor_train_step(&mut model, &tokens, step, &tc);
    }

    let mem_after_3000 = resident_bytes();

    // Verify loss is reasonable (not NaN/Inf).
    let tokens = tokenize("tyrannosaurus", &vocab, mc.block_size);
    let final_loss = tensor_train_step(&mut model, &tokens, 3000, &tc);
    assert!(
        final_loss.is_finite(),
        "Loss should be finite: {final_loss}"
    );
    assert!(final_loss < 5.0, "Loss should have decreased: {final_loss}");

    // Memory check (only on Linux where RSS is available).
    if mem_after_warmup > 0 && mem_after_3000 > 0 {
        let growth = mem_after_3000.saturating_sub(mem_after_warmup);
        // Allow up to 10MB growth for allocator overhead / fragmentation.
        assert!(
            growth < 10 * 1024 * 1024,
            "Memory grew too much: {growth} bytes ({} MB). Likely a leak.",
            growth / (1024 * 1024)
        );
    }

    println!(
        "3000 steps complete. Final loss: {final_loss:.4}. \
         Memory: warmup={}, final={} (delta={})",
        mem_after_warmup,
        mem_after_3000,
        mem_after_3000.saturating_sub(mem_after_warmup)
    );
}

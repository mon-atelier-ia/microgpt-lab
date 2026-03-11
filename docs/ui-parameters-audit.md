# UI Parameters Audit

> Which parameters from the TS project UIs need support in the Rust engine?

**Date**: 2026-02-28

## Source Projects

| Project | UI Type | Configurable params |
|---|---|---|
| microgpt-visualizer-fr | Pedagogical visualizer | 4 (temperature, dataset, steps, theme) |
| microgpt-ts-fr | Training playground | 10+ (full model/train/inference config) |

## Parameter Coverage

### Already in microgpt-rs

| Parameter | UI Source | Rust Location |
|---|---|---|
| n_embd | ts-fr dropdown [8,16,32] | `ModelConfig.n_embd` |
| n_head | ts-fr dropdown [1,2,4] | `ModelConfig.n_head` |
| n_layer | ts-fr dropdown [1,2,4] | `ModelConfig.n_layer` |
| block_size | ts-fr dropdown [8,16,32,64] | `ModelConfig.block_size` |
| lr | ts-fr log slider 0.001-0.5 | `TrainConfig.lr` |
| n_steps | Both (ts-fr slider, vis-fr presets) | `TrainConfig.n_steps` |
| beta1 | Hard-coded in both | `TrainConfig.beta1` |
| beta2 | Hard-coded in both | `TrainConfig.beta2` |
| temperature | Both (sliders) | `TrainConfig.temperature` |

### Missing from microgpt-rs

| Parameter | UI Source | What's needed | Priority |
|---|---|---|---|
| Dataset selection | Both | Embedded datasets + selection API | Phase 3 (roadmap) |
| Generation prefix | ts-fr text input | `prefix: &str` param in `generate()` | Small addition |
| n_samples in config | ts-fr slider 1-30 | Currently a `generate()` arg, not in config | Minor |

### UI-only (no engine support needed)

| Parameter | UI Source | Why |
|---|---|---|
| Theme (dark/light) | vis-fr | Pure CSS/UI |
| Mode explore/batch | ts-fr | UI logic, calls `train_step()` in loop |
| Character/position selectors | vis-fr | UI + future `get_activations()` (Phase 4) |

## Conclusion

The Rust engine already covers ~90% of UI-configurable parameters via `ModelConfig` and `TrainConfig`. The WASM binding (Phase 1) just needs to serialize these structs from/to JSON.

Three gaps remain:
1. **Datasets** — planned in Phase 3
2. **Generation prefix** — small API addition to `generate()`
3. **n_samples in config** — trivial refactor

No architectural changes needed.

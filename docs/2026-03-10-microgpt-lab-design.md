<!-- LLM-NAV
doc-id: lab-design
status: Validated
related: ui-parameters-audit (microgpt-rs), architecture (microgpt-rs)
summary: Design spec for microgpt-lab — parameter experimentation playground
-->

# microgpt-lab — Design Spec

**Date**: 2026-03-10
**Status**: Validated

## Purpose

An experimentation playground for microGPT. Users tweak hyperparameters, train models in real time, and compare inference results. The core value proposition: **change a parameter → see the impact immediately**, powered by Rust/WASM speed.

Target audience: learners who have used microgpt-visualizer-fr (pedagogy) or microgpt-xray (internals) and now want hands-on experimentation to verify their understanding.

## Modes

### Solo (default)

Full-width layout with three horizontal panels following a **40/25/35** ratio:

- **Parameters (40%)** — sidebar with all interactive controls: dataset preset selector, model config (n_embd, n_head, n_layer, block_size), training config (lr slider, temperature slider), Train and Generate buttons.
- **Loss curve (25%)** — real-time Chart.js canvas showing train and eval loss as the model trains step by step. Acts as a visual indicator that the configuration is converging — not the main event, but essential feedback.
- **Inference (35%)** — grid of generated words/names. This is the payoff: the user sees what the model learned.

### Compare (A/B)

Split vertical layout — two mirror columns side by side. Each column stacks the same three sections **vertically** with a **35/25/40** ratio:

- **Parameters (35%)** — fully editable, independent per column. Each column has its own Train button.
- **Loss curve (25%)** — independent real-time chart per column.
- **Inference (40%)** — word grid, slightly more space than solo since this is where comparison happens.

Each column has a distinct color identity (see Color section). Compare gives more vertical space to Inference (40% vs 35% in Solo) since side-by-side output comparison is the primary use case of this mode.

### Navigation

- No landing page — the app opens directly in Solo mode.
- A toggle/button in the top bar switches between Solo and Compare.
- Mode switch preserves Model A state; Compare adds Model B.

## Training Architecture

### WASM Engine

The `WasmGpt` class from microgpt-rs (already built and available in `app/wasm-pkg/`) provides:

- `constructor(names_text: string)` — create model from dataset text
- `train_step()` → `{ step, loss, word, lr }` — one training step (tensor engine, fast)
- `train(n_steps)` — batch training (no per-step feedback)
- `compute_probs(prefix_ids, temperature)` — inference probabilities
- `vocab_tokens()` — character list (returns JSON-encoded string array, must be `JSON.parse()`'d)
- `config()` — model config
- `reset(names_text)` — reset with new dataset
- `reset_training()` — reset weights only

### Web Workers

Each model instance runs in its own Web Worker to avoid blocking the UI:

- Worker loads WASM, creates `WasmGpt` instance
- Main thread sends: `{ type: "train", n_steps }` or `{ type: "generate", temperature, n_samples }`
- Worker calls `train_step()` in a loop, posting `{ type: "step", step, loss, word, lr }` back per step
- After training, worker runs autoregressive generation: start with BOS (`bos()`), call `compute_probs(prefix, temperature)`, sample token from returned distribution, append to prefix, repeat until end token or max length
- In Compare mode: two independent workers run in parallel

### Performance

- Training: `train_step()` tensor engine is ~100× faster than scalar. ~1ms/step on desktop, ~5-10ms on mobile.
- Two parallel workers: fine on any device with 2+ cores (all modern phones).
- Memory: each `WasmGpt` instance is a few KB (tiny vocab + small matrices). No concern even on low-end Android.

## Prerequisites (Phase 0 — blocking)

The following Rust/WASM changes **must be implemented before any frontend work begins**:

1. **Configurable constructor** — `WasmGpt::new_with_config(names_text, n_embd, n_head, n_layer, block_size)` or equivalent
2. **Learning rate setter** — `WasmGpt::set_lr(lr: f64)` to allow mid-training adjustment

Without these, the lab frontend cannot function. These changes live in `model-rs/crates/microgpt-wasm/`.

## WASM API Gap (detail)

The current `WasmGpt` constructor takes `names_text` (dataset) but model config (n_embd, n_head, etc.) is hardcoded. For the lab, we need a configurable constructor.

**Required change to `model-rs/crates/microgpt-wasm/`:**

Add a constructor or configuration method that accepts `ModelConfig` parameters:
```
WasmGpt::new_with_config(names_text, n_embd, n_head, n_layer, block_size)
```

Or a JS-friendly variant:
```
WasmGpt::new(names_text, config_json)
```

Additionally, `train_step()` currently takes no arguments — learning rate is internal. We need a setter:
```
WasmGpt::set_lr(lr: f64)
```

**Config change = full reset.** Any change to model config (n_embd, n_head, n_layer, block_size) requires destroying the `WasmGpt` instance and creating a new one (new dataset tokenization + weight initialization). This resets all training progress. The UI must make this clear — either auto-reset with a warning, or disable config changes while trained.

Training-time parameters (lr) can be changed mid-training via the setter without reset.

## Parameters (v1)

### Model Config (change = full reset)

| Parameter | Control | Values | Source |
|-----------|---------|--------|--------|
| dataset | Select dropdown | Preset list (see below) | `WasmGpt::new()` |
| n_embd | Select | 8, 16, 32 | `ModelConfig` |
| n_head | Select | 1, 2, 4 | `ModelConfig` |
| n_layer | Select | 1, 2, 4 | `ModelConfig` |
| block_size | Select | 8, 16, 32, 64 | `ModelConfig` |

**Constraint:** `n_embd % n_head == 0` must hold (head_dim must be integer). The UI must validate this and disable invalid combinations.

### Training Config (changeable mid-training)

| Parameter | Control | Values | Source |
|-----------|---------|--------|--------|
| lr | Slider (log scale) | 0.001 – 0.5 | `set_lr()` |

### Inference Config (post-training)

| Parameter | Control | Values | Source |
|-----------|---------|--------|--------|
| temperature | Slider | 0.1 – 2.0 | `compute_probs()` |

### Dataset Presets

Imported from microgpt-ts-fr. Minimum viable set:

- prénoms-fr (French first names)
- baby-names (English)
- dinosaures (French dinosaur names)

Additional presets can be added later from the ts-fr collection (13 datasets available).

### Future Parameters (nice-to-have)

- beta1, beta2 (Adam optimizer)
- n_steps (training duration)
- seed (reproducibility)
- generation prefix
- Custom dataset upload (text input or .txt file)

## Color System

### Palette Architecture

OKLCH tetradric palette (double complementary, ±15° shift):

- **Model A primary** (hue H) + **Model A accent** (hue H+15°)
- **Model B primary** (hue H+180°, complement) + **Model B accent** (hue H+195°)

Exact hue values to be determined during implementation with `frontend-design` skill. The principle: A and B must be instantly distinguishable at a glance.

### Design Principles

- OKLCH color space for perceptual uniformity
- Palette is unique to microgpt-lab — not inherited from xray or other projects
- Each project in the ecosystem has its own harmonic identity
- Dark background (neutral-950), content on elevated surfaces

## Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | React 19 + Vite 7 | Already scaffolded, fast HMR |
| Styling | Tailwind CSS 4 | Utility-first, consistent with ecosystem |
| Components | shadcn/ui (Radix + Tailwind) | Accessible, copy-paste, only install what's used (~3-4 Radix packages for slider, select, tooltip) |
| Charts | Chart.js + react-chartjs-2 | Canvas-based, lightweight (~30KB gzip), good real-time perf |
| Model | microgpt-rs WASM | Already built, tensor engine ~100× faster than scalar |
| Deploy | Vercel | Config optimized for static + WASM |
| Quality | ESLint 9 + Prettier + jscpd + Husky | Already configured |

## Deployment

Vercel with optimized config:
- Static export (no SSR needed)
- WASM files served with correct MIME type
- Cache headers for WASM binary

## Training Feedback (Gamification)

After generating words, the inference panel displays a dual-tone feedback badge based on three ML metrics:

- **Memorization** — Levenshtein fuzzy match against dataset (with temperature normalization)
- **Quality** — bigram cosine similarity + word length plausibility
- **Diversity** — Distinct-1/Distinct-2 (Li et al. 2016) + unique ratio

Seven feedback levels, evaluated in priority order:

| Level | Emoji | Technical message | Ado-friendly hint |
|-------|-------|-------------------|-------------------|
| `untrained` | — | (none) | (none) |
| `random` | 🎲 | Le modèle génère du bruit | Le modèle tâtonne encore — patience ! |
| `learning` | 📈 | Le modèle apprend les patterns du langage… | Ça progresse ! Le modèle commence à comprendre… |
| `sweet-spot` | 🎯 | Bonne généralisation ! | Bravo ! Le modèle invente des mots crédibles ! |
| `low-diversity` | 🔁 | Le modèle manque de créativité | Toujours les mêmes mots… Monte la température ! |
| `overfitting` | 🧠 | Le modèle mémorise le dataset | Le modèle triche — il récite au lieu d'inventer ! |
| `underpowered` | ⚡ | Capacité limitée : X params pour Y noms | Modèle trop petit pour ce dataset — augmente n_embd ! |

Design decisions:
- `DatasetProfile` (vocabSize, wordSet, bigramDist) pre-computed once per dataset change
- Dynamic capacity ratio (`params/dataset_size`) replaces hardcoded thresholds
- OLS trend detection for overfitting (rising memorization + falling diversity)
- No val split (WASM limitation), no Self-BLEU (overkill at this scale)
- Badge has `role="status"` + `aria-live="polite"` for accessibility
- Benchmark: `docs/benchmark-loss-thresholds.md`

## Implementation Notes

- shadcn/ui components: Slider, Select, Button, AlertDialog (4 Radix packages)
- Chart.js + react-chartjs-2 for loss curves (lazy-loaded, ~55KB gzip)
- Web Worker communication via `postMessage` / `onmessage` (no shared state)
- Solo and Compare share the same underlying `ModelPanel` component; Solo renders one, Compare renders two
- WASM guards: NaN/Inf loss detection, prefix clamped to block_size in compute_probs

# Architecture — microgpt-lab

> **LLM instruction**: Use this document as a reference for project structure, file roles, and data flow. When modifying code, verify your changes are consistent with the architecture described here. If you add files or change responsibilities, update this document.

## Overview

microgpt-lab is a parameter experimentation playground for microGPT. It lets users tweak hyperparameters, train models in-browser, and compare inferences side by side. The model engine is written in Rust, compiled to WASM.

Two layers:
1. **`model-rs/`** — Rust ML engine (zero dependencies, std-only) compiled to WASM via wasm-pack
2. **`app/`** — React 19 + Vite 7 + Tailwind 4 frontend

## `model-rs/` — Rust/WASM Engine

| File | Role |
|------|------|
| `src/value.rs` | Autograd scalar engine — `Value` with forward/backward ops |
| `src/model.rs` | GPT architecture (scalar engine) — embeddings, attention, MLP, Adam optimizer |
| `src/tensor.rs` | Tensor autograd engine — batched operations |
| `src/tensor_model.rs` | GPT architecture (tensor engine) — same arch, tensor-backed for performance |
| `src/config.rs` | `ModelConfig` + `TrainConfig` with defaults |
| `crates/microgpt-wasm/src/lib.rs` | WASM bindings — `WasmGpt` class exposed to JS via wasm-bindgen |

Key design:
- Architecture faithful to [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Zero dependencies (std-only Rust)
- Dynamic tokenizer: extracts unique characters from dataset at training time
- Adam optimizer with linear LR decay: `lr_t = lr * (1 - step/n_steps)`

## `app/` — React Frontend

### Entry & Layout

| File | Role |
|------|------|
| `src/main.tsx` | Entry point, renders `<App />` |
| `src/App.tsx` | Root: mode toggle (Solo/Compare) + panel routing via `<TopBar />` |
| `src/index.css` | Tailwind import + OKLCH design tokens + sRGB fallbacks |

### Components

| File | Role |
|------|------|
| `src/components/top-bar.tsx` | Mode tabs (Solo/Compare) + dataset selector, APG roving tabindex pattern |
| `src/components/error-boundary.tsx` | React error boundary with French messages |
| **Model Panel (triptyque)** | |
| `src/components/model-panel/model-panel.tsx` | Orchestrator: params → worker → loss/inference. Arch reset confirmation dialog |
| `src/components/model-panel/params-panel.tsx` | Hyperparameter controls (n_embd, n_head, n_layer, block_size, lr, temperature) |
| `src/components/model-panel/loss-panel.tsx` | Real-time loss curve (Chart.js) with raw + EMA lines, animated first draw |
| `src/components/model-panel/loss-chart.tsx` | Lazy-loaded Chart.js wrapper (react-chartjs-2) |
| `src/components/model-panel/inference-panel.tsx` | Generated words display |
| `src/components/model-panel/error-banner.tsx` | Worker error display |
| **UI primitives (shadcn/Radix)** | |
| `src/components/ui/button.tsx` | Button with variants |
| `src/components/ui/slider.tsx` | Slider control |
| `src/components/ui/select.tsx` | Select dropdown |
| `src/components/ui/alert-dialog.tsx` | Confirmation dialog |

### Hooks

| File | Role |
|------|------|
| `src/hooks/use-model-worker.ts` | Web Worker lifecycle — init, train, generate, set_lr, dispose. Exports `WorkerHandle` type |
| `src/hooks/use-loss-data.ts` | EMA computation + Chart.js dataset formatting. Returns `{ chartData, lastEma }` |

### Workers

| File | Role |
|------|------|
| `src/workers/model-worker.ts` | Web Worker: synchronous training loop, WASM instance management, validation, crash handling |

### Data & Lib

| File | Role |
|------|------|
| `src/data/presets.ts` | Dataset registry with lazy `import()` loading |
| `src/data/prenoms-simple.ts` | ~50 French first names |
| `src/data/baby-names.ts` | ~4500 English baby names |
| `src/data/dinosaures.ts` | ~100 dinosaur names |
| `src/data/pokemon-fr.ts` | ~150 Pokémon FR names |
| `src/lib/types.ts` | Shared types only (ModelParams, StepResult, TrainState, ColorVar, WorkerMessage, WorkerResponse) |
| `src/lib/constants.ts` | DEFAULT_PARAMS, DEFAULT_N_SAMPLES |
| `src/lib/utils.ts` | `cn()`, `resolveVar()`, `modelAccent()`, `modelColor()` |
| `src/lib/validation.ts` | `validHeadCounts()`, `isArchChange()` |

## Import Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  Main thread (App.tsx, UI components, hooks)             │
│  Imports: types only from lib/types.ts                   │
│  No direct WASM access                                   │
└────────────────────┬────────────────────────────────────┘
                     │ postMessage (structured clone)
┌────────────────────▼────────────────────────────────────┐
│  Web Worker (model-worker.ts)                            │
│  Imports: @wasm/microgpt_wasm (WasmGpt)                  │
│  Synchronous training loop, no setTimeout chunking       │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

```
User clicks "Entraîner"
  → use-model-worker.ts posts { type: "train", n_steps }
  → model-worker.ts: synchronous loop, posts { type: "step", data } for each step
  → posts { type: "train_done" }
  → hook updates trainState = "trained"

User clicks "Générer"
  → use-model-worker.ts posts { type: "generate", temperature, n_samples }
  → model-worker.ts: compute_probs + sample per token, posts { type: "generated", words }

Architecture change while trained:
  → model-panel.tsx detects isArchChange() → shows ArchResetDialog
  → on confirm: re-init worker with new config (resets weights)

WASM crash (OOM / panic):
  → model-worker.ts catches "unreachable" → posts French error message
  → user must re-init model
```

## Design Tokens

OKLCH tetradric palette with sRGB fallbacks:
- `--model-a` / `--model-a-accent` — Model A color identity
- `--model-b` / `--model-b-accent` — Model B color identity
- CSS vars resolved via `resolveVar()` for Canvas 2D (Chart.js)
- Tailwind `@theme` block maps CSS vars to utility classes

## Quality Gates

- **pre-commit**: ESLint --fix --max-warnings=0 + Prettier
- **pre-push**: tsc --noEmit + vite build + jscpd (5% threshold)
- **Tests**: Vitest (36 unit/component) + Playwright (5 E2E)
- **Rust**: cargo test + cargo fmt --check + cargo clippy -- -D warnings

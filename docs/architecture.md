# Architecture

> **LLM instruction**: Use this document as a reference for project structure, file roles, and data flow. When modifying code, verify your changes are consistent with the architecture described here. If you add files or change responsibilities, update this document.

## Overview

microgpt-ts is a complete GPT built from scratch in TypeScript with zero runtime dependencies. The project has two distinct layers:

1. **`microgpt/`** — Standalone ML library (~400 lines)
2. **`web/`** — Next.js playground that imports the library

The library runs anywhere (Node, browser, workers). The playground is a teaching tool that lets users train and interact with the model in-browser.

## `microgpt/` — ML Library

| File | Role |
|------|------|
| `value.ts` | Autograd engine — `Value` class with forward/backward ops |
| `model.ts` | GPT-2 architecture — embeddings, multi-head attention, MLP, residual connections, rmsnorm. Exports `buildTokenizer`, `initStateDict`, `forward`, `inference`, `inferenceStepwise` |
| `train.ts` | Adam optimizer with linear LR decay. Exports `trainStep`, `initAdamState` |
| `utils.ts` | Math helpers (`softmax`, `multinomial`, `emaSmooth`, `parseDocs`, `splitDocs`) |
| `browser.ts` | Web Worker serialization — `snapshotWeights`/`restoreStateDict` + message types (`TrainWorkerIn`, `TrainWorkerOut`) |

Key design: the tokenizer is **dynamic** — it extracts unique characters from the dataset at training time. Vocab is not hardcoded to a-z; any Unicode works.

## `web/` — Next.js Playground

### App Router pages

| Path | File | Description |
|------|------|-------------|
| `/` | `app/page.tsx` | Landing page with hero CTA |
| `/playground` | `app/playground/page.tsx` | Main demo (wraps `TrainDemo`) |
| `/about` | `app/about/page.tsx` | Project explanation |

### `web/components/demo/` — Playground UI

The playground is a 3-tab interface (Dataset → Train → Generate). Each tab has a **sidebar** (controls) and a **content area** (visualization).

| File | Role |
|------|------|
| `demo.tsx` | Main orchestrator — state, worker lifecycle, tab routing |
| `presets.ts` | Dataset presets with per-preset model/training configs |
| `types.ts` | Shared types (`Status`, `GenerateMode`) and utilities |
| **Dataset tab** | |
| `dataset-sidebar.tsx` | Preset selector + word count + train button |
| `dataset-tab.tsx` | Dataset text preview / custom text editor |
| **Train tab** | |
| `train-sidebar.tsx` | Model config selects + LR/steps sliders + action buttons |
| `train-tab.tsx` | Training stats, loss chart, live generation samples |
| `train-status.tsx` | 4-stat grid (step, time, train loss, eval loss) |
| `loss-chart.tsx` | Recharts loss curve (train + eval) |
| `live-gen-stream.tsx` | Rolling list of generated words during training |
| **Generate tab** | |
| `generate-sidebar.tsx` | Mode toggle, temperature, samples, prefix, action buttons |
| `generate-tab.tsx` | Batch output list or step-by-step explore view |
| `explore-view.tsx` | Token probability visualization for step-by-step inference |
| `token-prob-chart.tsx` | Bar chart of per-token probabilities |

### `web/workers/` — Bridge between UI and library

| File | Role |
|------|------|
| `train-worker.ts` | Training loop — receives `init` message, runs chunked training (10 steps/chunk via `setTimeout`), posts progress, handles generation requests post-training |
| `eval-worker.ts` | Evaluation — receives weight snapshots, computes eval loss on held-out data |

### `web/lib/`

| File | Role |
|------|------|
| `strings.ts` | All French UI strings (single source of truth for i18n) |
| `utils.ts` | Tailwind `cn()` merge helper |

## `datasets/` — Training data

TypeScript files exporting `string[]`. Lowercase a-z only, sorted alphabetically, min 3 chars per entry. See `AGENTS.md` for the full dataset table.

## `scripts/`

| File | Role |
|------|------|
| `demo.ts` | CLI proof that `microgpt/` works standalone (no web dependency) |
| `fetch-baby-names.ts` | Data fetcher for baby names dataset |
| `fetch-movie-titles.ts` | Data fetcher for movie titles dataset |

## Frontend ← Model API surface

The frontend never calls model functions directly on the main thread. All heavy computation runs in Web Workers. The main thread only imports **types** for type-safety.

### `microgpt/model.ts`

| Export | Kind | Consumers |
|--------|------|-----------|
| `ModelConfig` | type | `demo.tsx`, `presets.ts`, `train-sidebar.tsx`, `eval-worker.ts`, `train-worker.ts` |
| `DEFAULT_CONFIG` | const | `train-worker.ts` |
| `StateDict` | type | `train-worker.ts` |
| `Tokenizer` | type | `train-worker.ts` |
| `buildTokenizer` | fn | `train-worker.ts` |
| `initStateDict` | fn | `train-worker.ts` |
| `getParams` | fn | `train-worker.ts` |
| `forward` | fn | `eval-worker.ts`, `train-worker.ts` |
| `InferenceStep` | type | `demo.tsx`, `explore-view.tsx`, `generate-tab.tsx`, `explore-view.stories.tsx` |
| `inferenceStepwise` | fn | `train-worker.ts` |
| `inference` | fn | `train-worker.ts` |

### `microgpt/browser.ts`

Serialization layer: converts `Value` objects ↔ plain numbers for Worker `postMessage`.

| Export | Kind | Consumers |
|--------|------|-----------|
| `TrainWorkerIn` | type | `demo.tsx` |
| `TrainWorkerOut` | type | `demo.tsx`, `eval-worker.ts`, `train-worker.ts` |
| `NumericStateDict` | type | `eval-worker.ts`, `train-worker.ts` |
| `snapshotWeights` | fn | `train-worker.ts` |
| `restoreStateDict` | fn | `eval-worker.ts`, `train-worker.ts` |

### `microgpt/train.ts`

| Export | Kind | Consumers |
|--------|------|-----------|
| `AdamConfig` | type | `train-worker.ts` |
| `DEFAULT_ADAM_CONFIG` | const | `train-worker.ts` |
| `AdamState` | type | `train-worker.ts` |
| `StepInfo` | type | `train-worker.ts` |
| `initAdamState` | fn | `train-worker.ts` |
| `trainStep` | fn | `train-worker.ts` |

### `microgpt/utils.ts`

| Export | Kind | Consumers |
|--------|------|-----------|
| `emaSmooth` | fn | `train-worker.ts` |
| `parseDocs` | fn | `train-worker.ts` |
| `splitDocs` | fn | `train-worker.ts` |

### `microgpt/value.ts`

Not imported directly by any frontend file. Consumed indirectly via `model.ts` and `browser.ts`.

### Import boundaries

```
┌─────────────────────────────────────────────────────────┐
│  Main thread (demo.tsx, UI components)                  │
│  Imports: types only (ModelConfig, InferenceStep,       │
│           TrainWorkerIn, TrainWorkerOut)                 │
└────────────────────┬────────────────────────────────────┘
                     │ postMessage (structured clone)
┌────────────────────▼────────────────────────────────────┐
│  Workers (train-worker.ts, eval-worker.ts)              │
│  Imports: all functions + types from microgpt/*          │
│  browser.ts converts Value ↔ number at the boundary     │
└─────────────────────────────────────────────────────────┘
```

## Data flow

```
User clicks "Train"
  → demo.tsx creates Web Worker (train-worker.ts)
  → sends { type: "init", datasetText, modelConfig, ... }
  → train-worker.ts: buildTokenizer → initStateDict → runTraining()
    → chunked loop: trainStep() → post("steps") / post("live-gen") / post("eval-snapshot")
    → eval-worker.ts receives snapshots, returns avgLoss
    → if loss is NaN → post("error", "nan-divergence") → stop immediately
    → if training ends with high loss → post("warning", "high-loss")
  → post("done")
  → demo.tsx sets status = "trained" | "diverged"

  Error path (NaN):
    → worker posts "error" → demo.tsx terminateWorkers() + setStatus("diverged")
    → train-tab.tsx shows red pedagogical encart (what is NaN, why it happened, what to do)
    → "Ré-entraîner" visible, "Générer →" hidden

  Warning path (high loss):
    → worker posts "warning" before "done" → demo.tsx stores trainingWarning
    → train-tab.tsx shows amber encart (limited learning, suggestions)
    → generation still available (model is usable but degraded)

User clicks "Generate"
  → demo.tsx sends { type: "generate" | "explore-start" }
  → train-worker.ts: inference() → post("gen-word") or inferenceStepwise() → post("explore-step")
```

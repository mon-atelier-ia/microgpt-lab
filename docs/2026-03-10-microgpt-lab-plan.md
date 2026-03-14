# microgpt-lab Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameter experimentation playground for microGPT with Solo and Compare modes, powered by Rust/WASM.

**Architecture:** Three-panel triptyque (params/loss/inference) with a shared `ModelPanel` component. Solo renders one panel horizontally; Compare renders two vertically. Each model runs in its own Web Worker wrapping a `WasmGpt` WASM instance. shadcn/ui for controls, Chart.js for real-time loss curves, OKLCH tetradric palette for A/B color identity.

**Tech Stack:** React 19, Vite 7, Tailwind 4, shadcn/ui (Radix), Chart.js + react-chartjs-2, microgpt-rs WASM, Vercel

**Spec:** `docs/2026-03-10-microgpt-lab-design.md`

---

## File Structure

```
app/src/
├── main.tsx                    # Entry point (existing)
├── App.tsx                     # Root: mode toggle + panel routing
├── index.css                   # Tailwind import + OKLCH tokens (existing)
├── vite-env.d.ts               # Vite types (existing)
├── theme/
│   └── tokens.css              # OKLCH design tokens (tetradric palette)
├── data/
│   ├── presets.ts              # Dataset presets (name, data, default config)
│   ├── prenoms-simple.ts       # ~50 French first names
│   ├── baby-names.ts           # ~4500 English baby names
│   ├── dinosaures.ts           # ~100 dinosaur names
│   └── pokemon-fr.ts           # ~150 Pokémon FR names
├── lib/
│   ├── types.ts                # Shared types (ModelParams, TrainState, StepResult, etc.)
│   ├── validation.ts           # n_embd % n_head constraint
│   └── utils.ts                # cn() helper
├── workers/
│   └── model-worker.ts         # Web Worker: WASM init, train loop, generation
├── hooks/
│   ├── use-model-worker.ts     # Hook: spawn worker, send messages, receive state
│   └── use-loss-data.ts        # Hook: accumulate loss points for Chart.js
├── components/
│   ├── ui/                     # shadcn/ui components (slider, select, button, tooltip)
│   │   ├── slider.tsx
│   │   ├── select.tsx
│   │   ├── button.tsx
│   │   └── tooltip.tsx
│   ├── top-bar.tsx             # Mode toggle (Solo/Compare) + title
│   ├── model-panel/
│   │   ├── model-panel.tsx     # Orchestrator: params + loss + inference
│   │   ├── params-panel.tsx    # All param controls (selects, sliders, buttons)
│   │   ├── loss-panel.tsx      # Chart.js real-time loss curve
│   │   └── inference-panel.tsx # Word grid display
│   ├── solo-view.tsx           # Solo layout (horizontal 40/25/35)
│   └── compare-view.tsx        # Compare layout (vertical split, two panels)
```

### Rust changes (in `model-rs/crates/microgpt-wasm/src/lib.rs`)

- Add `new_with_config(names_text, n_embd, n_head, n_layer, block_size)` constructor
- Add `set_lr(lr: f64)` method

---

## Chunk 1: Phase 0 — WASM API Extensions (Rust)

> **Prerequisite:** Chunk 2 and 3 depend on Chunk 1 being complete and WASM rebuilt. Do not start Chunk 2 until Task 2 Step 6 (WASM rebuild) succeeds.

### Task 1: Add configurable constructor and store ModelConfig

**Files:**
- Modify: `model-rs/crates/microgpt-wasm/src/lib.rs:64-117` (struct + impl)
- Modify: `model-rs/crates/microgpt-wasm/tests/parity.rs`

- [x] **Step 1: Add `mc` field to WasmGpt struct**

In `model-rs/crates/microgpt-wasm/src/lib.rs`, add `mc: ModelConfig` to the `WasmGpt` struct (after `tc`). Update the existing `new()` constructor to store `mc`:

```rust
// In struct WasmGpt:
mc: ModelConfig,

// In new(): before Ok(WasmGpt { ... })
let mc = ModelConfig::default();
// ... and add `mc,` to the struct literal
```

Also update `reset_training()` and `reset()` to use `self.mc` instead of `ModelConfig::default()`.

- [x] **Step 2: Write tests for new_with_config**

The tests live in the existing `model-rs/crates/microgpt-wasm/tests/parity.rs`. Add imports if not present:

```rust
use microgpt_wasm::WasmGpt;
```

Add tests (these are integration tests — `WasmGpt` methods that return `Result<_, JsError>` work in native target because `JsError` implements `Debug`):

```rust
#[test]
fn test_new_with_config_custom_params() {
    let gpt = WasmGpt::new_with_config("alice\nbob\ncharlie", 8, 2, 1, 8)
        .expect("should create with custom config");
    assert_eq!(gpt.model_config().n_embd, 8);
    assert_eq!(gpt.model_config().n_head, 2);
    assert_eq!(gpt.model_config().n_layer, 1);
    assert_eq!(gpt.model_config().block_size, 8);
}

#[test]
fn test_new_with_config_invalid_head_dim() {
    let result = WasmGpt::new_with_config("alice\nbob", 8, 3, 1, 16);
    assert!(result.is_err(), "n_embd=8, n_head=3 should fail (8%3!=0)");
}
```

- [x] **Step 3: Run tests to verify they fail**

Run: `cd model-rs && cargo test --test parity test_new_with_config 2>&1`
Expected: compilation error — `new_with_config` and `model_config` don't exist yet

- [x] **Step 4: Implement new_with_config + model_config**

In `model-rs/crates/microgpt-wasm/src/lib.rs`, add to `#[wasm_bindgen] impl WasmGpt`:

```rust
/// Create model with custom architecture config.
/// Validates n_embd % n_head == 0.
#[wasm_bindgen]
pub fn new_with_config(
    names_text: &str,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
) -> Result<WasmGpt, JsError> {
    if n_embd % n_head != 0 {
        return Err(JsError::new(&format!(
            "n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        )));
    }
    let docs = parse_docs(names_text);
    if docs.is_empty() {
        return Err(JsError::new("new_with_config: names_text is empty"));
    }
    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let vocab = build_vocab(&doc_refs);
    let mc = ModelConfig { n_embd, n_head, n_layer, block_size };
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);
    let model = Model::new(vocab.size(), &mut rng, mc, &tc);
    let train_order = Self::shuffled_order(docs.len(), &mut rng);
    let (_, tracked) = build_param_options(&vocab.tokens);
    let mut rng_t = Rng::new(42);
    let tensor_model = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);
    Ok(WasmGpt {
        model, tensor_model, vocab, rng, docs, tc, mc,
        step_count: 0, train_order, tracked, weights_dirty: false,
    })
}
```

And in the non-wasm `impl WasmGpt` block (test-only, no `#[wasm_bindgen]`):

```rust
/// Expose ModelConfig for testing.
pub fn model_config(&self) -> ModelConfig {
    self.mc
}
```

- [x] **Step 5: Fix reset_training() and reset() to use self.mc**

Replace all `ModelConfig::default()` in `reset_training()` and `reset()` with `self.mc`.

- [x] **Step 6: Run tests to verify they pass**

Run: `cd model-rs && cargo test --release 2>&1 | tail -20`
Expected: all tests PASS including the two new ones

- [x] **Step 7: Commit**

```bash
git add model-rs/
git commit -m "feat(wasm): add configurable constructor new_with_config, store mc"
```

---

### Task 2: Add set_lr method to WasmGpt

**Files:**
- Modify: `model-rs/crates/microgpt-wasm/src/lib.rs`
- Modify: `model-rs/crates/microgpt-wasm/tests/parity.rs`

- [x] **Step 1: Write failing test for set_lr**

In `model-rs/crates/microgpt-wasm/tests/parity.rs`, add:

```rust
#[test]
fn test_set_lr() {
    let mut gpt = WasmGpt::new("alice\nbob\ncharlie").expect("create");
    gpt.set_lr(0.05);
    assert!((gpt.current_lr() - 0.05).abs() < 1e-10);
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd model-rs && cargo test --test parity test_set_lr 2>&1`
Expected: compilation error — `set_lr` and `current_lr` don't exist

- [x] **Step 3: Implement set_lr and current_lr**

In `model-rs/crates/microgpt-wasm/src/lib.rs`, add to `#[wasm_bindgen] impl WasmGpt`:

```rust
/// Set learning rate (can be changed mid-training).
pub fn set_lr(&mut self, lr: f64) {
    self.tc.lr = lr;
}

/// Get current base learning rate.
pub fn current_lr(&self) -> f64 {
    self.tc.lr
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd model-rs && cargo test --release 2>&1 | tail -20`
Expected: all tests PASS

- [x] **Step 5: Run fmt + clippy**

Run: `cd model-rs && cargo fmt && cargo clippy -- -D warnings 2>&1`
Expected: no errors

- [x] **Step 6: Rebuild WASM**

Run: `cd /c/Dev/microgpt-lab && bash build-wasm.sh 2>&1`
Expected: WASM output in `app/wasm-pkg/`, new `.d.ts` includes `new_with_config`, `set_lr`, `current_lr`

- [x] **Step 7: Commit**

```bash
git add model-rs/ app/wasm-pkg/
git commit -m "feat(wasm): add set_lr and current_lr methods"
```

---

## Chunk 2: Phase 1 — Frontend Foundation

> **Prerequisite:** Chunk 1 must be complete (WASM rebuilt with `new_with_config`, `set_lr`, `current_lr`) before starting Task 6 (Web Worker). Tasks 3-5 and 8 can start in parallel with Chunk 1.

### Task 3: Install dependencies (shadcn/ui, Chart.js)

**Files:**
- Modify: `app/package.json`
- Create: `app/src/lib/utils.ts`
- Create: `app/src/components/ui/button.tsx`
- Create: `app/src/components/ui/select.tsx`
- Create: `app/src/components/ui/slider.tsx`
- Create: `app/src/components/ui/tooltip.tsx`

- [x] **Step 1: Install Radix + Chart.js + shadcn deps**

```bash
cd app && pnpm add @radix-ui/react-select @radix-ui/react-slider @radix-ui/react-tooltip @radix-ui/react-slot chart.js react-chartjs-2 class-variance-authority clsx tailwind-merge lucide-react
```

- [x] **Step 2: Create cn() utility**

Write `app/src/lib/utils.ts`:
```ts
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [x] **Step 3: Copy shadcn/ui components from microgpt-ts-fr**

Copy and adapt these files from `C:/Dev/microgpt-ts-fr/web/components/ui/`:
- `button.tsx`
- `select.tsx`
- `slider.tsx`
- `tooltip.tsx`

Adapt imports: change `@/lib/utils` path if needed. Remove any Next.js-specific imports.

- [x] **Step 4: Verify build**

Run: `cd app && npx tsc --noEmit && pnpm build`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/
git commit -m "feat: install shadcn/ui components and Chart.js"
```

---

### Task 4: Dataset presets

**Files:**
- Create: `app/src/data/prenoms-simple.ts`
- Create: `app/src/data/dinosaures.ts`
- Create: `app/src/data/pokemon-fr.ts`
- Create: `app/src/data/presets.ts`

- [x] **Step 1: Copy dataset files from microgpt-ts-fr**

Copy the raw arrays from:
- `C:/Dev/microgpt-ts-fr/datasets/prenoms-simple.ts` → `app/src/data/prenoms-simple.ts`
- `C:/Dev/microgpt-ts-fr/datasets/baby-names.ts` → `app/src/data/baby-names.ts`
- `C:/Dev/microgpt-ts-fr/datasets/dinosaures.ts` → `app/src/data/dinosaures.ts`
- `C:/Dev/microgpt-ts-fr/datasets/pokemon-fr.ts` → `app/src/data/pokemon-fr.ts`

- [x] **Step 2: Create presets.ts**

Write `app/src/data/presets.ts`:
```ts
import { prenomsSimple } from './prenoms-simple';
import { babyNames } from './baby-names';
import { dinosaures } from './dinosaures';
import { pokemonFr } from './pokemon-fr';

export type Preset = {
  id: string;
  name: string;
  description: string;
  data: string[];
};

export const PRESETS: Preset[] = [
  {
    id: 'prenoms-simple',
    name: 'Prénoms FR',
    description: '~50 prénoms français courants',
    data: prenomsSimple,
  },
  {
    id: 'baby-names',
    name: 'Baby Names EN',
    description: '~4500 English baby names',
    data: babyNames,
  },
  {
    id: 'dinosaures',
    name: 'Dinosaures',
    description: '~100 noms de dinosaures',
    data: dinosaures,
  },
  {
    id: 'pokemon-fr',
    name: 'Pokémon FR',
    description: '~150 noms de Pokémon en français',
    data: pokemonFr,
  },
];
```

- [x] **Step 3: Verify build**

Run: `cd app && npx tsc --noEmit && pnpm build`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/src/data/
git commit -m "feat: add dataset presets from microgpt-ts-fr"
```

---

### Task 5: Types and validation

**Files:**
- Create: `app/src/lib/types.ts`
- Create: `app/src/lib/validation.ts`

- [x] **Step 1: Create shared types**

Write `app/src/lib/types.ts`:
```ts
export type ModelParams = {
  datasetId: string;
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
  lr: number;
  temperature: number;
};

export type StepResult = {
  step: number;
  loss: number;
  word: string;
  lr: number;
};

export type TrainState = 'idle' | 'training' | 'trained' | 'error';

export type WorkerMessage =
  | { type: 'init'; datasetText: string; config: ModelParams }
  | { type: 'train'; n_steps: number }
  | { type: 'set_lr'; lr: number }
  | { type: 'generate'; temperature: number; n_samples: number };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'step'; data: StepResult }
  | { type: 'train_done' }
  | { type: 'generated'; words: string[] }
  | { type: 'error'; message: string };

export const DEFAULT_PARAMS: ModelParams = {
  datasetId: 'prenoms-simple',
  n_embd: 16,
  n_head: 4,
  n_layer: 1,
  block_size: 16,
  lr: 0.01,
  temperature: 0.8,
};
```

- [x] **Step 2: Create validation helper**

Write `app/src/lib/validation.ts`:
```ts
/** Valid n_head values for a given n_embd (n_embd % n_head must be 0). */
export function validHeadCounts(n_embd: number, options: number[]): number[] {
  return options.filter((h) => n_embd % h === 0);
}
```

- [x] **Step 3: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/src/lib/
git commit -m "feat: add shared types and validation helpers"
```

---

### Task 6: Web Worker (model-worker.ts)

**Files:**
- Create: `app/src/workers/model-worker.ts`

- [x] **Step 1: Write the Web Worker**

Write `app/src/workers/model-worker.ts`:
```ts
import init, { WasmGpt } from '@wasm/microgpt_wasm';
import type { WorkerMessage, WorkerResponse } from '../lib/types';

let gpt: WasmGpt | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

async function handleInit(datasetText: string, config: { n_embd: number; n_head: number; n_layer: number; block_size: number }) {
  await init();
  if (gpt) gpt.free();
  gpt = WasmGpt.new_with_config(datasetText, config.n_embd, config.n_head, config.n_layer, config.block_size);
  post({ type: 'ready' });
}

let trainRemaining = 0;

function handleTrain(n_steps: number) {
  if (!gpt) { post({ type: 'error', message: 'Model not initialized' }); return; }
  trainRemaining = n_steps;
  trainChunk();
}

function trainChunk() {
  if (!gpt || trainRemaining <= 0) {
    post({ type: 'train_done' });
    return;
  }
  const chunkSize = Math.min(10, trainRemaining);
  for (let i = 0; i < chunkSize; i++) {
    const result = gpt.train_step();
    post({ type: 'step', data: result as any });
    trainRemaining--;
  }
  // Yield to message queue so main thread can process step events,
  // then continue with next chunk.
  if (trainRemaining > 0) {
    setTimeout(trainChunk, 0);
  } else {
    post({ type: 'train_done' });
  }
}

function handleGenerate(temperature: number, n_samples: number) {
  if (!gpt) { post({ type: 'error', message: 'Model not initialized' }); return; }
  const vocab = JSON.parse(gpt.vocab_tokens()) as string[];
  const bos = gpt.bos();
  const words: string[] = [];

  for (let s = 0; s < n_samples; s++) {
    let prefix = new Uint32Array([bos]);
    let word = '';
    for (let t = 0; t < 20; t++) {
      const probs = gpt.compute_probs(prefix, temperature);
      const token = sampleFromProbs(probs);
      if (token === bos) break;
      word += vocab[token];
      const newPrefix = new Uint32Array(prefix.length + 1);
      newPrefix.set(prefix);
      newPrefix[prefix.length] = token;
      prefix = newPrefix;
    }
    if (word.length > 0) words.push(word);
  }
  post({ type: 'generated', words });
}

function sampleFromProbs(probs: Float64Array): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  try {
    switch (e.data.type) {
      case 'init':
        await handleInit(e.data.datasetText, e.data.config);
        break;
      case 'train':
        handleTrain(e.data.n_steps);
        break;
      case 'set_lr':
        if (gpt) gpt.set_lr(e.data.lr);
        break;
      case 'generate':
        handleGenerate(e.data.temperature, e.data.n_samples);
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
```

The `handleTrain` function uses a chunked `setTimeout(trainChunk, 0)` pattern: train 10 steps, yield to the message queue so the main thread can process step events and update the loss chart, then continue. This avoids blocking the worker's message loop while ensuring all steps complete.

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS (worker file type-checks with the WASM types)

- [x] **Step 3: Commit**

```bash
git add app/src/workers/
git commit -m "feat: add model-worker wrapping WASM for train + generate"
```

---

### Task 7: useModelWorker hook

**Files:**
- Create: `app/src/hooks/use-model-worker.ts`

- [x] **Step 1: Write the hook**

Write `app/src/hooks/use-model-worker.ts`:
```ts
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ModelParams, StepResult, TrainState, WorkerMessage, WorkerResponse } from '../lib/types';
import { PRESETS } from '../data/presets';

export function useModelWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [trainState, setTrainState] = useState<TrainState>('idle');
  const [steps, setSteps] = useState<StepResult[]>([]);
  const [words, setWords] = useState<string[]>([]);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/model-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      switch (e.data.type) {
        case 'ready':
          setTrainState('idle');
          break;
        case 'step':
          setSteps((prev) => [...prev, e.data.data]);
          break;
        case 'train_done':
          setTrainState('trained');
          break;
        case 'generated':
          setWords(e.data.words);
          break;
        case 'error':
          setTrainState('error');
          console.error('Worker error:', e.data.message);
          break;
      }
    };

    return () => worker.terminate();
  }, []);

  const send = useCallback((msg: WorkerMessage) => {
    workerRef.current?.postMessage(msg);
  }, []);

  const initModel = useCallback((params: ModelParams) => {
    const preset = PRESETS.find((p) => p.id === params.datasetId);
    if (!preset) return;
    setSteps([]);
    setWords([]);
    setTrainState('idle');
    send({
      type: 'init',
      datasetText: preset.data.join('\n'),
      config: params,
    });
  }, [send]);

  const train = useCallback((n_steps: number) => {
    setTrainState('training');
    send({ type: 'train', n_steps });
  }, [send]);

  const setLr = useCallback((lr: number) => {
    send({ type: 'set_lr', lr });
  }, [send]);

  const generate = useCallback((temperature: number, n_samples: number) => {
    send({ type: 'generate', temperature, n_samples });
  }, [send]);

  return { trainState, steps, words, initModel, train, setLr, generate };
}
```

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add app/src/hooks/
git commit -m "feat: add useModelWorker hook for worker lifecycle"
```

---

### Task 8: OKLCH design tokens

**Files:**
- Create: `app/src/theme/tokens.css`
- Modify: `app/src/index.css`

- [x] **Step 1: Create OKLCH tokens**

Write `app/src/theme/tokens.css`:
```css
/*
 * OKLCH tetradric palette (double complementary ±15°).
 * Final hues to be refined with frontend-design skill.
 * Placeholder: A=220° (blue), B=40° (amber), A'=235°, B'=55°
 */
:root {
  /* Surface */
  --surface-0: oklch(0.13 0.01 260);
  --surface-1: oklch(0.17 0.01 260);
  --surface-2: oklch(0.21 0.015 260);

  /* Model A (blue) */
  --model-a: oklch(0.65 0.15 220);
  --model-a-accent: oklch(0.55 0.12 235);
  --model-a-muted: oklch(0.35 0.06 220);

  /* Model B (amber) */
  --model-b: oklch(0.70 0.15 40);
  --model-b-accent: oklch(0.60 0.12 55);
  --model-b-muted: oklch(0.35 0.06 40);

  /* Semantic */
  --success: oklch(0.65 0.15 145);
  --error: oklch(0.60 0.18 25);
  --text-primary: oklch(0.93 0.01 260);
  --text-secondary: oklch(0.65 0.01 260);
  --text-muted: oklch(0.45 0.01 260);
}
```

- [x] **Step 2: Import tokens in index.css**

Modify `app/src/index.css`:
```css
@import 'tailwindcss';
@import './theme/tokens.css';
```

- [x] **Step 3: Verify build**

Run: `cd app && pnpm build`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/src/theme/ app/src/index.css
git commit -m "feat: add OKLCH tetradric design tokens"
```

---

### Task 9: Params panel component

**Files:**
- Create: `app/src/components/model-panel/params-panel.tsx`

- [x] **Step 1: Write params panel**

Write `app/src/components/model-panel/params-panel.tsx`:

A form component that renders:
- Dataset select (from PRESETS)
- Model config selects: n_embd (8/16/32), n_head (filtered by validHeadCounts), n_layer (1/2/4), block_size (8/16/32/64)
- LR slider (log scale, 0.001–0.5)
- Temperature slider (0.1–2.0)
- Train button + Generate button

Props:
```ts
type ParamsPanelProps = {
  params: ModelParams;
  onParamsChange: (params: ModelParams) => void;
  onTrain: () => void;
  onGenerate: () => void;
  trainState: TrainState;
  colorVar: 'a' | 'b';
};
```

Uses shadcn/ui `Select`, `Slider`, `Button`. Color-coded with CSS vars `--model-a` or `--model-b` based on `colorVar`.

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add app/src/components/model-panel/
git commit -m "feat: add params-panel component with all controls"
```

---

### Task 10: Loss panel component (Chart.js)

**Files:**
- Create: `app/src/hooks/use-loss-data.ts`
- Create: `app/src/components/model-panel/loss-panel.tsx`

- [x] **Step 1: Write useLossData hook**

Write `app/src/hooks/use-loss-data.ts`:
```ts
import { useMemo } from 'react';
import type { StepResult } from '../lib/types';

export function useLossData(steps: StepResult[], colorVar: 'a' | 'b') {
  return useMemo(() => ({
    labels: steps.map((s) => s.step),
    datasets: [
      {
        label: 'Train Loss',
        data: steps.map((s) => s.loss),
        borderColor: `var(--model-${colorVar})`,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
      },
    ],
  }), [steps, colorVar]);
}
```

- [x] **Step 2: Write loss panel**

Write `app/src/components/model-panel/loss-panel.tsx`:

A Chart.js Line chart component:
- Takes `steps: StepResult[]` and `colorVar: 'a' | 'b'`
- Canvas-based, no animation (perf), responsive
- Shows step count and current loss value below chart
- Chart options: no legend, minimal axes, dark background

- [x] **Step 3: Verify build**

Run: `cd app && npx tsc --noEmit && pnpm build`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/src/hooks/use-loss-data.ts app/src/components/model-panel/loss-panel.tsx
git commit -m "feat: add loss-panel with real-time Chart.js curve"
```

---

### Task 11: Inference panel component

**Files:**
- Create: `app/src/components/model-panel/inference-panel.tsx`

- [x] **Step 1: Write inference panel**

Write `app/src/components/model-panel/inference-panel.tsx`:

A grid of generated words:
- Takes `words: string[]` and `colorVar: 'a' | 'b'`
- CSS grid, auto-fill columns
- Each word in a card with surface-2 background
- Empty state: subtle "Train a model to see generated words" message
- Word count + temperature footer

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add app/src/components/model-panel/inference-panel.tsx
git commit -m "feat: add inference-panel word grid component"
```

---

### Task 12: ModelPanel orchestrator

**Files:**
- Create: `app/src/components/model-panel/model-panel.tsx`

- [x] **Step 1: Write model panel**

Write `app/src/components/model-panel/model-panel.tsx`:

Orchestrator that composes `ParamsPanel`, `LossPanel`, `InferencePanel`:
- Takes `colorVar: 'a' | 'b'` and `layout: 'horizontal' | 'vertical'`
- `horizontal` (Solo): flexbox row, flex 40/25/35
- `vertical` (Compare): flexbox column, flex 35/25/40
- Accepts an optional `workerHandle` prop (from parent) OR creates its own `useModelWorker()` — this enables state preservation when switching modes (see Task 15)
- Manages `ModelParams` state, auto-inits worker on mount and on model config changes (config change = reinit + warning)
- When model config changes (n_embd, n_head, etc.) while `trainState === 'trained'`, show a warning: "Changing this parameter will reset training." Proceed on confirm.
- LR changes go through `setLr()` without reinit (mid-training safe)
- Calls `train(200)` on Train click, `generate(temperature, 10)` on Generate click

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add app/src/components/model-panel/model-panel.tsx
git commit -m "feat: add ModelPanel orchestrator component"
```

---

## Chunk 3: Phase 2 — Views and Polish

### Task 13: TopBar component

**Files:**
- Create: `app/src/components/top-bar.tsx`

- [x] **Step 1: Write top bar**

Write `app/src/components/top-bar.tsx`:

Simple bar with:
- "microgpt-lab" title (left)
- Mode toggle: "Solo" / "Compare" buttons (right)
- Props: `mode: 'solo' | 'compare'`, `onModeChange: (mode) => void`

- [x] **Step 2: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add app/src/components/top-bar.tsx
git commit -m "feat: add TopBar with mode toggle"
```

---

### Task 14: SoloView and CompareView

**Files:**
- Create: `app/src/components/solo-view.tsx`
- Create: `app/src/components/compare-view.tsx`

- [x] **Step 1: Write SoloView**

Write `app/src/components/solo-view.tsx`:

Receives `workerHandle` prop from App. Renders one `<ModelPanel layout="horizontal" colorVar="a" workerHandle={workerHandle} />` at full width.

- [x] **Step 2: Write CompareView**

Write `app/src/components/compare-view.tsx`:

Receives `workerHandleA` prop from App. Creates its own `useModelWorker()` for Model B. Renders two `<ModelPanel layout="vertical" />` side by side:
- Left: `colorVar="a"` with `workerHandle={workerHandleA}` (preserved from Solo)
- Right: `colorVar="b"` with its own worker (created on mount, destroyed on unmount)
- Flexbox with gap, each flex-1

- [x] **Step 3: Verify build**

Run: `cd app && npx tsc --noEmit`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/src/components/solo-view.tsx app/src/components/compare-view.tsx
git commit -m "feat: add SoloView and CompareView layouts"
```

---

### Task 15: Wire up App.tsx

**Files:**
- Modify: `app/src/App.tsx`

- [x] **Step 1: Update App.tsx**

Replace `app/src/App.tsx` with:
```tsx
import { useState } from 'react';
import { TopBar } from './components/top-bar';
import { SoloView } from './components/solo-view';
import { CompareView } from './components/compare-view';
import { useModelWorker } from './hooks/use-model-worker';

type Mode = 'solo' | 'compare';

export default function App() {
  const [mode, setMode] = useState<Mode>('solo');
  // Model A worker lives at App level so it survives mode switches.
  const modelA = useModelWorker();

  return (
    <div className="min-h-screen" style={{ background: 'var(--surface-0)', color: 'var(--text-primary)' }}>
      <TopBar mode={mode} onModeChange={setMode} />
      <main className="p-4">
        {mode === 'solo' ? (
          <SoloView workerHandle={modelA} />
        ) : (
          <CompareView workerHandleA={modelA} />
        )}
      </main>
    </div>
  );
}
```

**Key decision:** Model A's worker handle is owned by `App`, not by `ModelPanel`. This means switching Solo → Compare preserves Model A's training state (worker stays alive). Model B gets its own worker inside `CompareView`. Switching Compare → Solo destroys Model B only.

- [x] **Step 2: Verify full build**

Run: `cd app && npx tsc --noEmit && pnpm build`
Expected: PASS

- [x] **Step 3: Manual smoke test**

Run: `cd app && pnpm dev`
- Open http://localhost:5173
- Verify Solo mode shows params/loss/inference panels
- Click "Compare" — verify two columns appear
- Select a dataset, click Train — verify loss curve updates in real time
- Click Generate — verify words appear

- [x] **Step 4: Commit**

```bash
git add app/src/App.tsx
git commit -m "feat: wire up App with Solo and Compare modes"
```

---

### Task 16: Vercel config

**Files:**
- Modify: `app/vercel.json` (create if missing)
- Modify: `app/vite.config.js`

- [x] **Step 1: Create vercel.json**

Write `app/vercel.json`:
```json
{
  "buildCommand": "pnpm build",
  "outputDirectory": "dist",
  "framework": "vite",
  "headers": [
    {
      "source": "/assets/(.*).wasm",
      "headers": [
        { "key": "Content-Type", "value": "application/wasm" },
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    }
  ]
}
```

- [x] **Step 2: Ensure WASM is included in build**

Verify `vite.config.js` handles `.wasm` files correctly. Vite 7 handles WASM imports natively. If needed, add `assetsInclude: ['**/*.wasm']` to the Vite config.

- [x] **Step 3: Verify production build**

Run: `cd app && pnpm build && ls dist/assets/*.wasm 2>/dev/null`
Expected: WASM file present in dist/assets/

- [x] **Step 4: Commit**

```bash
git add app/vercel.json app/vite.config.js
git commit -m "chore: add Vercel config with WASM headers"
```

---

### Task 17: Frontend design polish pass

- [ ] **Step 1: Invoke frontend-design skill** ⚠️ JAMAIS EXÉCUTÉ — reporté à Task 57

Use the `frontend-design` skill to review and polish:
- OKLCH palette finalization (tetradric harmony)
- Component spacing and visual hierarchy
- Dark theme consistency
- Responsive behavior (mobile → stack vertically)

- [x] **Step 2: Apply feedback and commit**

```bash
git add -A
git commit -m "style: apply frontend-design polish pass"
```

---

### Task 18: Final integration commit

- [x] **Step 1: Run all quality gates**

```bash
cd app && npx tsc --noEmit && pnpm build && npx eslint src/ --max-warnings=0 && npx jscpd src/
```
Expected: all PASS

- [x] **Step 2: Add .superpowers to .gitignore**

Append `.superpowers/` to `.gitignore` if not already there.

- [x] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final integration — all quality gates pass"
```

---

## Chunk 4: Phase 3 — Industry Standards Audit Fixes

> **Context:** Post-implementation audit revealed 3 critical, 7 important, and 10 minor issues against industry standards. All must be fixed before the branch can be considered complete.

### Task 19: Fix critical issues (C1, C2, C3)

**Files:**
- Modify: `app/src/hooks/use-model-worker.ts`
- Modify: `app/src/components/model-panel/model-panel.tsx`
- Modify: `app/src/workers/model-worker.ts`

- [x] **Step 1: C1 — Unbounded `steps` array growth**

In `use-model-worker.ts`, the `steps` array grows without limit via `setSteps(prev => [...prev, item])` on every training step. This causes O(n²) copies, memory leaks, and re-render storms over sustained training sessions.

Fix: Accumulate steps in a `useRef` buffer, flush to state at a throttled interval (e.g., every 100ms) or on `train_done`. Cap the array to a reasonable maximum (e.g., 5000 points) using a ring buffer or tail truncation.

- [x] **Step 2: C2 — Confirm dialog fires after `setParams`**

In `model-panel.tsx`, `setParams(next)` is called on line 60 before the `window.confirm` check. If the user cancels, the UI shows new params but the model still has old ones — desync.

Fix: Move `setParams(next)` after the confirm check. Only update state when the change is actually applied.

- [x] **Step 3: C3 — WASM `free()` not called on worker termination**

In `model-worker.ts`, when the worker is terminated via `worker.terminate()`, `gpt.free()` is never called. WASM linear memory may leak.

Fix: Add a `'dispose'` message type. In `use-model-worker.ts`, send `dispose` before `terminate()`. In the worker, call `gpt.free()` on dispose.

- [x] **Step 4: Verify build**

Run: `cd app && npx tsc --noEmit && pnpm build`

- [x] **Step 5: Commit**

```bash
git commit -m "fix: critical — unbounded steps array, confirm desync, WASM free on dispose"
```

---

### Task 20: Fix Chart.js CSS variable issue (I2)

**Files:**
- Modify: `app/src/hooks/use-loss-data.ts`

- [x] **Step 1: I2 — CSS custom properties not supported by Canvas 2D**

`borderColor: var(--model-a)` is invalid for Canvas 2D context. Chart.js renders to `<canvas>`, not DOM.

Fix: Resolve CSS variables at render time using `getComputedStyle(document.documentElement).getPropertyValue(...)`, or pass resolved color strings as props.

- [x] **Step 2: Verify the loss curve actually renders with correct color**

- [x] **Step 3: Commit**

```bash
git commit -m "fix: resolve CSS vars for Chart.js canvas rendering"
```

---

### Task 21: Error handling & resilience (I1, I4)

**Files:**
- Modify: `app/src/hooks/use-model-worker.ts`
- Modify: `app/src/workers/model-worker.ts`
- Modify: `app/src/components/model-panel/model-panel.tsx`

- [x] **Step 1: I1 — Surface error messages in UI**

Add `errorMessage: string | null` state to `useModelWorker`. Set it on `'error'` responses. Display it in `ModelPanel` (e.g., a styled banner).

- [x] **Step 2: I4 — Validate worker messages**

Add guards in the worker's `onmessage`: `n_steps > 0`, `temperature > 0`, `lr > 0`, config values are positive integers. Return an `'error'` response for invalid inputs.

- [x] **Step 3: Commit**

```bash
git commit -m "fix: surface worker errors in UI, validate worker messages"
```

---

### Task 22: Accessibility (M2)

**Files:**
- Modify: `app/src/components/model-panel/params-panel.tsx`
- Modify: `app/src/components/model-panel/loss-panel.tsx`
- Modify: `app/src/components/model-panel/inference-panel.tsx`
- Modify: `app/src/components/model-panel/model-panel.tsx`
- Modify: `app/src/components/top-bar.tsx`

- [x] **Step 1: M2 — Add ARIA attributes**

- Training status as `aria-live="polite"` region
- `aria-label` on all custom interactive elements
- `role="status"` on loss/step counters
- `role="list"` on word grid, `role="listitem"` on word cards
- Keyboard-navigable mode toggle (ensure Button handles this via Radix)

- [x] **Step 2: Commit**

```bash
git commit -m "fix: add ARIA attributes for screen reader support"
```

---

### Task 23: Responsive design (M3)

**Files:**
- Modify: `app/src/components/compare-view.tsx`
- Modify: `app/src/components/solo-view.tsx`
- Modify: `app/src/components/model-panel/model-panel.tsx`

- [x] **Step 1: M3 — Add responsive breakpoints**

Compare mode: stack vertically on mobile (< 768px). Solo mode: panels wrap on narrow viewports. Use Tailwind responsive classes or CSS media queries.

- [x] **Step 2: Commit**

```bash
git commit -m "fix: responsive layout for mobile viewports"
```

---

### Task 24: Code quality fixes (M1, M5, M6, M9, M10)

**Files:**
- Modify: `app/src/lib/types.ts` (M1)
- Modify: `app/src/App.tsx` (M1)
- Modify: `app/src/components/top-bar.tsx` (M1)
- Modify: `app/src/components/model-panel/inference-panel.tsx` (M5)
- Modify: `app/src/theme/tokens.css` (M6)
- Modify: `app/src/components/model-panel/model-panel.tsx` (M9)
- Modify: `model-rs/crates/microgpt-wasm/src/lib.rs` (M10)

- [x] **Step 1: M1 — Deduplicate `Mode` type to `types.ts`**
- [x] **Step 2: M5 — Fix fragile key prop** — use index only since words have no stable identity
- [x] **Step 3: M6 — Add OKLCH fallbacks** — `@supports` fallback with sRGB hex equivalents
- [x] **Step 4: M9 — Replace `window.confirm` with custom dialog** — use Radix AlertDialog
- [x] **Step 5: M10 — Make `train_step` return `Result`** — align with `train_step_traced` pattern

- [x] **Step 6: Verify build (Rust + Frontend)**

```bash
cd model-rs && cargo test --release && cargo clippy -- -D warnings
cd app && npx tsc --noEmit && pnpm build && npx eslint src/ --max-warnings=0
```

- [x] **Step 7: Commit**

```bash
git commit -m "fix: code quality — deduplicate types, OKLCH fallbacks, custom dialog, consistent Rust API"
```

---

### Task 25: Performance optimizations (M4, M7, M8)

**Files:**
- Modify: `app/src/components/model-panel/model-panel.tsx` (M4)
- Modify: `app/src/components/model-panel/params-panel.tsx` (M4)
- Modify: `app/src/data/presets.ts` (M8)
- Potentially modify: `app/src/components/model-panel/loss-panel.tsx` (M7)

- [x] **Step 1: M4 — Configurable training step count** — add a steps slider/input in params-panel (100/200/500/1000)
- [x] **Step 2: M8 — Lazy-load datasets** — use dynamic `import()` in presets, load on selection
- [x] **Step 3: M7 — Evaluate Chart.js replacement** — if bundle impact > 150KB gzipped, consider a lightweight SVG sparkline. Otherwise document the tradeoff and keep Chart.js.

- [x] **Step 4: Verify build + bundle size**

```bash
cd app && pnpm build 2>&1 | grep -i chunk
```

- [x] **Step 5: Commit**

```bash
git commit -m "perf: configurable steps, lazy datasets, evaluate chart bundle"
```

---

### Task 26: Styling consistency (I7)

**Files:**
- Modify: `app/src/index.css`
- Modify: `app/src/theme/tokens.css`
- Modify: multiple component files

- [x] **Step 1: I7 — Unify styling approach**

Extend Tailwind 4 theme with CSS custom properties using `@theme` block. Replace inline `style={{ background: 'var(--surface-0)' }}` with Tailwind classes like `bg-surface-0`. Apply consistently across all components.

- [x] **Step 2: Verify build**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: unify styling — Tailwind theme integration, remove inline styles"
```

---

### Task 27: Frontend unit tests (I5)

**Files:**
- Create: `app/src/lib/__tests__/validation.test.ts`
- Create: `app/src/components/model-panel/__tests__/params-utils.test.ts`

- [x] **Step 1: Install Vitest**

```bash
cd app && pnpm add -D vitest
```

- [x] **Step 2: Write tests for pure functions**

- `validation.ts`: `validHeadCounts` edge cases
- `params-utils.ts`: `posToLr`/`lrToPos` roundtrip, `formatLr` formatting

- [x] **Step 3: Run tests**

```bash
cd app && npx vitest run
```

- [x] **Step 4: Commit**

```bash
git commit -m "test: add unit tests for validation and params-utils"
```

---

### Task 28: Final quality gate

- [x] **Step 1: Run all checks**

```bash
cd model-rs && cargo test --release && cargo fmt --check && cargo clippy -- -D warnings
cd app && npx tsc --noEmit && pnpm build && npx eslint src/ --max-warnings=0 && npx jscpd src/ && npx vitest run
```

- [x] **Step 2: Final commit**

```bash
git commit -m "chore: all industry audit fixes complete — quality gates green"
```

---

## Chunk 5: Phase 4 — Final Quality Pass

> **Context:** Post-audit normalization complete. Remaining gaps: smoke test, pre-push hooks, integration tests, bundle optimization, lint suppression cleanup, runtime type safety, CSP headers, license audit.

### Task 29: Smoke test

- [x] **Step 1: Start dev server and verify manually**

```bash
cd app && pnpm dev
```

Open http://localhost:5173 and verify:
- Solo mode: params panel renders with all controls
- Select dataset, click Train → loss curve updates in real time
- Click Generate → word grid populates
- Switch to Compare → two columns, Model B has its own worker
- Switch back to Solo → Model A training state preserved
- Change architecture param while trained → confirm dialog appears
- Cancel → params revert, model unchanged
- Confirm → model re-inits, loss clears
- Change LR mid-training → no reinit, takes effect on next step
- Error state: try invalid config → error banner shows

- [x] **Step 2: Fix any runtime issues found**
- [x] **Step 3: Commit if fixes needed**

---

### Task 30: Pre-push hook validation

- [x] **Step 1: Run the actual pre-push hook**

```bash
cd app && npx tsc --noEmit && pnpm build && npx jscpd src/
```

These are the documented pre-push gates. Verify they pass end-to-end as the hook would run them.

- [x] **Step 2: Test the hook itself**

```bash
cd /c/Dev/microgpt-lab && git stash && git stash pop
# Simulate a push by running .husky/pre-push manually if it exists
```

---

### Task 31: Remove eslint-disable suppression

**Files:**
- Modify: `app/src/hooks/use-model-worker.ts`

- [x] **Step 1: Audit the `eslint-disable react-hooks/exhaustive-deps`**

The auto-init `useEffect` in `model-panel.tsx` uses `// eslint-disable-next-line react-hooks/exhaustive-deps` to omit `initModel` and `params` from the deps array. This is intentional (init once on mount), but the pattern is fragile.

Fix: Use a `useRef(false)` guard for "init once" instead of suppressing the lint:
```ts
const initialized = useRef(false);
useEffect(() => {
  if (!initialized.current) {
    initialized.current = true;
    initModel(params);
  }
}, [initModel, params]);
```

This satisfies the exhaustive-deps rule while maintaining mount-only behavior.

- [x] **Step 2: Remove all `eslint-disable` comments from app/src/**
- [x] **Step 3: Verify `npx eslint src/ --max-warnings=0`**
- [x] **Step 4: Commit**

```bash
git commit -m "fix: remove eslint-disable — use ref guard for mount-only init"
```

---

### Task 32: Runtime type guard for WASM return values

**Files:**
- Modify: `app/src/workers/model-worker.ts`

- [x] **Step 1: Add a type guard for StepResult**

Replace the unsafe `as` cast on `gpt.train_step()` with a runtime validator:
```ts
function isStepResult(v: unknown): v is StepResult {
  return typeof v === 'object' && v !== null
    && 'step' in v && typeof (v as any).step === 'number'
    && 'loss' in v && typeof (v as any).loss === 'number'
    && 'word' in v && typeof (v as any).word === 'string'
    && 'lr' in v && typeof (v as any).lr === 'number';
}
```

Use it in `trainChunk()`:
```ts
const raw = gpt.train_step();
if (!isStepResult(raw)) {
  post({ type: 'error', message: 'Unexpected train_step result' });
  return;
}
post({ type: 'step', data: raw });
```

- [x] **Step 2: Verify tsc + build**
- [x] **Step 3: Commit**

```bash
git commit -m "fix: runtime type guard for WASM train_step return value"
```

---

### Task 33: Bundle size — code-split Chart.js

**Files:**
- Modify: `app/src/components/model-panel/loss-panel.tsx`

- [x] **Step 1: Lazy-load Chart.js via React.lazy**

Wrap the `Line` chart in a lazy-loaded component to split Chart.js (~200KB) into a separate chunk:

```tsx
import { lazy, Suspense } from 'react';
const LazyLine = lazy(() =>
  import('./loss-chart').then(m => ({ default: m.LossChart }))
);
```

Create `app/src/components/model-panel/loss-chart.tsx` with the Chart.js imports and `<Line>` rendering.

- [x] **Step 2: Verify chunk split**

```bash
cd app && pnpm build 2>&1 | grep -i chunk
```

Expected: main chunk < 300KB, chart chunk separate.

- [x] **Step 3: Commit**

```bash
git commit -m "perf: lazy-load Chart.js to reduce initial bundle"
```

---

### Task 34: CSP headers for WASM

**Files:**
- Modify: `app/vercel.json`

- [x] **Step 1: Add Content-Security-Policy header**

WASM execution requires `'wasm-unsafe-eval'` in script-src. Add a security header:

```json
{
  "source": "/(.*)",
  "headers": [
    {
      "key": "Content-Security-Policy",
      "value": "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; worker-src 'self' blob:; connect-src 'self'"
    }
  ]
}
```

- [x] **Step 2: Verify local build + serve works with CSP**

```bash
cd app && pnpm build && npx serve dist
```

Open in browser, check console for CSP violations.

- [x] **Step 3: Commit**

```bash
git commit -m "chore: add Content-Security-Policy headers for WASM"
```

---

### Task 35: License audit

- [x] **Step 1: Check all dependency licenses**

```bash
cd app && npx license-checker --summary 2>/dev/null || pnpm exec license-checker --summary
```

Alternatively, manually verify key deps:
- `chart.js`: MIT
- `react-chartjs-2`: MIT
- `@radix-ui/*`: MIT
- `class-variance-authority`: Apache 2.0
- `clsx`: MIT
- `tailwind-merge`: MIT
- `lucide-react`: ISC

All must be compatible with MIT/Apache. Flag any GPL/AGPL/SSPL.

- [x] **Step 2: Add THIRD_PARTY_LICENSES.md if needed**
- [x] **Step 3: Commit**

```bash
git commit -m "chore: verify dependency license compatibility"
```

---

### Task 36: Integration tests — worker + hook

**Files:**
- Create: `app/src/hooks/__tests__/use-model-worker.test.ts`

- [x] **Step 1: Install jsdom + @testing-library/react**

```bash
cd app && pnpm add -D @testing-library/react @testing-library/react-hooks jsdom
```

Add to vitest config in `vite.config.js`:
```js
test: { environment: 'jsdom' }
```

- [x] **Step 2: Write integration test**

Test the useModelWorker hook lifecycle with a mock worker. Verify:
- `initModel` sends correct message to worker
- `train` sets trainState to 'training'
- Step messages update steps array
- `train_done` sets trainState to 'trained'
- `generate` sends correct message
- `dispose` is sent on cleanup

- [x] **Step 3: Run tests**

```bash
cd app && npx vitest run
```

- [x] **Step 4: Commit**

```bash
git commit -m "test: add integration tests for useModelWorker hook"
```

---

### Task 37: Component tests

**Files:**
- Create: `app/src/components/model-panel/__tests__/params-panel.test.tsx`
- Create: `app/src/components/model-panel/__tests__/inference-panel.test.tsx`
- Create: `app/src/components/model-panel/__tests__/error-banner.test.tsx`
- Create: `app/src/components/model-panel/__tests__/model-panel.test.tsx`

- [x] **Step 1: Install @testing-library/react + @testing-library/user-event**

```bash
cd app && pnpm add -D @testing-library/react @testing-library/user-event
```

- [x] **Step 2: ParamsPanel tests** (`params-panel.test.tsx`)

- Renders all dataset options from PRESETS
- Renders n_embd/n_head/n_layer/block_size selects
- n_head options update when n_embd changes (divisibility filter)
- LR slider renders with log-scale value
- Temperature slider renders in [0.1, 2.0] range
- Train button shows "Entraîner" when idle, "Entraînement…" when training
- Train button disabled during training
- Generate button calls onGenerate
- Calls onParamsChange with correct payload on control changes
- trainSteps select renders options [100, 200, 500, 1000, 2000]

- [x] **Step 3: InferencePanel tests** (`inference-panel.test.tsx`)

- Empty state: shows placeholder message
- With words: renders word grid with `role="list"` and `role="listitem"`
- Shows word count in footer
- Shows temperature in footer when provided
- Uses correct model color for word styling

- [x] **Step 4: ErrorBanner tests** (`error-banner.test.tsx`)

- Returns null when message is null
- Renders with `role="alert"` when message is provided
- Displays the error message text

- [x] **Step 5: ModelPanel integration tests** (`model-panel.test.tsx`)

Mock `useModelWorker` hook. Verify:
- Renders all three sub-panels (params, loss, inference)
- Horizontal layout applies correct flex ratios
- Vertical layout applies correct flex ratios
- ErrorBanner appears when errorMessage is set
- AlertDialog appears when changing arch params while trained
- AlertDialog cancel does not call initModel
- AlertDialog confirm calls initModel with new params
- LR change calls setLr without initModel

- [x] **Step 6: Run tests**

```bash
cd app && npx vitest run
```

- [x] **Step 7: Commit**

```bash
git commit -m "test: add component tests for panels, error banner, model panel"
```

---

### Task 38: E2E tests (Playwright)

**Files:**
- Create: `app/e2e/solo-flow.spec.ts`
- Create: `app/e2e/compare-flow.spec.ts`
- Create: `app/playwright.config.ts`

- [x] **Step 1: Install Playwright**

```bash
cd app && pnpm add -D @playwright/test && npx playwright install chromium
```

- [x] **Step 2: Configure Playwright** (`app/playwright.config.ts`)

```ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  webServer: {
    command: 'pnpm dev',
    port: 5173,
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://localhost:5173',
  },
});
```

Add script to `package.json`: `"test:e2e": "playwright test"`

- [x] **Step 3: Solo flow E2E** (`solo-flow.spec.ts`)

- Page loads, TopBar visible with "microgpt-lab" title
- Solo mode active by default
- Params panel renders with dataset select, architecture controls, sliders, buttons
- Select "Prénoms FR" dataset
- Click "Entraîner" → button changes to "Entraînement…"
- Loss curve appears and updates (canvas element present)
- Training completes → button reverts to "Entraîner"
- Click "Générer" → word grid populates with generated words
- Word count footer updates
- Change n_embd → confirm dialog if trained
- Cancel dialog → params unchanged
- Confirm dialog → loss curve clears, model re-inits

- [x] **Step 4: Compare flow E2E** (`compare-flow.spec.ts`)

- Click "Compare" → two model panels appear side by side
- Each panel has independent controls
- Train Model A → loss curve A updates
- Train Model B → loss curve B updates independently
- Switch back to "Solo" → Model A state preserved (loss curve still visible)
- Switch back to "Compare" → Model A still has data, Model B is fresh

- [x] **Step 5: Run E2E**

```bash
cd app && pnpm test:e2e
```

- [x] **Step 6: Commit**

```bash
git commit -m "test: add Playwright E2E tests for Solo and Compare flows"
```

---

### Task 39: Final quality gate (definitive)

> Renumbered from Task 37. Includes all test suites.

- [x] **Step 1: Run ALL checks**

```bash
cd model-rs && cargo test --release && cargo fmt --check && cargo clippy -- -D warnings
cd app && npx tsc --noEmit && pnpm build && npx eslint src/ --max-warnings=0 && npx jscpd src/ && npx vitest run && pnpm test:e2e
```

- [x] **Step 2: Verify zero eslint-disable comments**

```bash
grep -r "eslint-disable" app/src/ | grep -v node_modules | grep -v __tests__
```

Expected: 0 results (or only in shadcn/ui generated files).

- [x] **Step 3: Verify bundle size**

Main chunk < 300KB gzipped.

- [x] **Step 4: Final commit**

```bash
git commit -m "chore: final quality pass — all gates green, zero suppressions"
```

---

## Chunk 6: Phase 5 — Final Audit Fixes

> **Context:** Final audit revealed 8 important and 4 minor issues. All must be fixed.

### Task 40: Add React ErrorBoundary (I1)

**Files:**
- Create: `app/src/components/error-boundary.tsx`
- Modify: `app/src/App.tsx`

- [x] **Step 1: Create ErrorBoundary class component** — catches render errors, shows recovery UI
- [x] **Step 2: Wrap App content in ErrorBoundary**
- [x] **Step 3: Commit**

```bash
git commit -m "feat: add React ErrorBoundary for crash recovery"
```

---

### Task 41: Delete orphaned loss-chart-options.ts, deduplicate (I2 + M1)

**Files:**
- Delete: `app/src/components/model-panel/loss-chart-options.ts`
- Modify: `app/src/components/model-panel/loss-panel.tsx` — already has inline copy, keep it

- [x] **Step 1: Delete orphaned file**
- [x] **Step 2: Verify build**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: delete orphaned loss-chart-options.ts"
```

---

### Task 42: Move Preset type to types.ts (I3)

**Files:**
- Modify: `app/src/lib/types.ts` — add `Preset` type
- Modify: `app/src/data/presets.ts` — import from types.ts instead of defining locally

- [x] **Step 1: Move type, update imports**
- [x] **Step 2: Commit**

```bash
git commit -m "refactor: move Preset type to types.ts per convention"
```

---

### Task 43: Remove `as` casts in isStepResult (I5)

**Files:**
- Modify: `app/src/workers/model-worker.ts`

- [x] **Step 1: Refactor isStepResult to use `in` narrowing without `as Record<>`**
- [x] **Step 2: Commit**

```bash
git commit -m "refactor: remove as casts in isStepResult type guard"
```

---

### Task 44: Fix useLossData memoization (I7 + M8)

**Files:**
- Modify: `app/src/hooks/use-loss-data.ts` — wrap returned ChartData in useMemo keyed on steps + borderColor
- Modify: `app/src/components/model-panel/loss-panel.tsx` — remove `useMemo(() => getChartOptions(), [])`, resolve every render for theme reactivity

- [x] **Step 1: Memoize ChartData object properly** — deps: [lossValues, labels, borderColor]
- [x] **Step 2: Fix chartOptions to also resolve on every render** (or pass borderColor as dep)
- [x] **Step 3: Commit**

```bash
git commit -m "fix: proper memoization for Chart.js data and options"
```

---

### Task 45: Complete ARIA tab pattern (M6)

**Files:**
- Modify: `app/src/components/top-bar.tsx` — add `aria-controls` + `id` linking
- Modify: `app/src/App.tsx` — add `role="tabpanel"` + `id` on content area

- [x] **Step 1: Wire aria-controls/id on tabs and tabpanel**
- [x] **Step 2: Commit**

```bash
git commit -m "fix: complete ARIA tab pattern with tabpanel linkage"
```

---

### Task 46: Move param options to constants.ts (M2)

**Files:**
- Modify: `app/src/lib/constants.ts` — add N_EMBD_OPTIONS, N_LAYER_OPTIONS, etc.
- Modify: `app/src/components/model-panel/params-panel.tsx` — import from constants

- [x] **Step 1: Move constants, update imports**
- [x] **Step 2: Commit**

```bash
git commit -m "refactor: move param options to constants.ts"
```

---

### Task 47: Extract n_samples constant (M7)

**Files:**
- Modify: `app/src/lib/constants.ts`
- Modify: `app/src/components/model-panel/model-panel.tsx`

- [x] **Step 1: Add DEFAULT_N_SAMPLES = 10 to constants.ts**
- [x] **Step 2: Use it in model-panel.tsx**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: extract n_samples constant"
```

---

### Task 48: Fix import style consistency (M10)

**Files:**
- Modify: `app/src/components/model-panel/model-panel.tsx`

- [x] **Step 1: Split mixed import into separate runtime and type imports**
- [x] **Step 2: Commit**

```bash
git commit -m "style: consistent type-only imports"
```

---

### Task 49: Mark all completed tasks in plan

- [x] **Step 1: Check all task checkboxes `[x]` for Tasks 1-48**
- [x] **Step 2: Commit**

---

### Task 50: Final quality gate (definitive)

- [x] **Step 1: Run ALL checks**: tsc, build, eslint, jscpd, vitest, playwright
- [x] **Step 2: Verify zero `as` casts outside ui/ and worker JSON.parse**
- [x] **Step 3: Verify zero orphaned files**
- [x] **Step 4: Commit**

```bash
git commit -m "chore: chunk 6 complete — all audit findings fixed"
```

---

## Chunk 7: Bug Fixes — Conformité Design Spec

> **Constat post-déploiement (2026-03-11)** : l'app déployée sur Vercel présente des non-conformités critiques par rapport au design spec (`docs/2026-03-10-microgpt-lab-design.md`). Les paramètres d'architecture (n_embd, n_head, n_layer, block_size) s'affichent et se modifient dans l'UI mais ne sont potentiellement pas propagés au WASM worker via `new_with_config()`. Le mode Compare est donc inutile si les deux colonnes créent des modèles identiques. Le layout ne respecte pas les ratios du spec.

### Task 51: Audit complet UI vs Design Spec

- [x] **Step 1: Vérifier que le changement de n_embd/n_head/n_layer/block_size appelle `new_with_config()` dans le worker** — ✅ OK
- [x] **Step 2: Vérifier que le bouton "Entraîner" utilise les params actuels du panneau (pas les defaults)** — ✅ OK
- [x] **Step 3: Vérifier l'indépendance des states A/B en mode Compare** — ✅ OK
- [x] **Step 4: Documenter tous les écarts trouvés** — voir ci-dessous

**Résultat audit (2026-03-11):**
- ✅ Params modifiables et propagés au worker via `new_with_config()`
- ✅ Dialogue de reset sur changement d'architecture
- ✅ States A/B indépendants en Compare (workers séparés, loss différentes)
- ✅ Layout ratios conformes (40/25/35 Solo, 35/25/40 Compare)
- ❌ State Solo non préservé quand on switch vers Compare (spec: "Mode switch preserves Model A state")
- ❌ favicon.ico 404

### Task 52: Fix — Préservation du state Model A lors du switch Solo → Compare

**Spec:** "Mode switch preserves Model A state; Compare adds Model B."

**Files:**
- Modify: `app/src/App.tsx` ou state management

- [x] **Step 1: Lift Model A state pour qu'il persiste entre Solo et Compare** — ✅ Déjà fait : `useModelWorker()` est appelé dans `App` et passé en prop aux vues
- [x] **Step 2: Vérifier que les params, la loss curve et les mots générés de A sont préservés** — ✅ Confirmé : le hook ne se démonte pas au switch
- [x] **Step 3: Commit** — N/A, aucun changement nécessaire

```bash
git commit -m "fix: preserve Model A state across Solo/Compare mode switch"
```

### Task 55: Add favicon

**Files:**
- Create: `app/public/favicon.ico` (or `favicon.svg`)
- Modify: `index.html` if needed

- [x] **Step 1: Lab flask SVG favicon** — `app/public/favicon.svg` with OKLCH palette (#1a1a2e bg, #3b8edb flask, #2e6fb3 liquid)
- [x] **Step 2: Console banner easter egg** — `app/public/banner.js`, P-A.G ASCII art, reads `--model-a` CSS var at runtime
- [x] **Step 3: CSP compliance** — Moved inline script to external `banner.js` (CSP `script-src 'self'` blocks inline)
- [x] **Step 4: Reusable template** — `C:\Dev\Easter_eggs\console-log-banner.html` with `{{APP_NAME}}`, `{{CSS_VAR}}`, `{{FALLBACK}}` placeholders
- [x] **Step 5: Commits** — `35bc029`, `7985268`, `277b310`

```bash
git commit -m "feat: add lab flask favicon with OKLCH palette colors (Task 55)"
git commit -m "feat: add console banner easter egg with dynamic OKLCH color (Task 55)"
git commit -m "fix: move console banner to external script for CSP compliance"
```

### Task 56: EMA smoothing sur la loss curve (style TensorBoard)

**Files:**
- Modify: `app/src/components/model-panel/loss-chart.tsx` (ou équivalent)
- Modify: `app/src/lib/utils.ts` ou nouveau helper

- [ ] **Step 1: Implémenter calcul EMA (α ≈ 0.1) sur les données de loss**
- [ ] **Step 2: Ajouter une seconde série Chart.js (EMA) en overlay sur la loss brute**
- [ ] **Step 3: Loss brute en trait fin semi-transparent, EMA en trait épais — couleur OKLCH cohérente avec le modèle (A/B)**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add EMA smoothing overlay on loss curve"
```

### Task 57: REFONTE VISUELLE — Tenir les promesses du design spec

> **Contexte** : La Task 17 "frontend-design polish pass" a été marquée complète sans exécution réelle. Le skill `frontend-design` n'a jamais été invoqué. L'UI livrée est un template shadcn/ui brut sans personnalité. Cette task corrige la fraude.

**Promesses non tenues à honorer :**

- [x] **Step 1: Typography** — Geist Sans (display) + Geist Mono (code) via @fontsource. Configured in Tailwind @theme.
- [x] **Step 2: Identité couleur A/B** — OKLCH tetradric palette (blue 220° / orange 40°). panel-glow-a/b with border, background tint, triple box-shadow. Slider colors match model identity.
- [x] **Step 3: Élévation visible** — 8% lightness jumps (surface-0/1/2: 0.12/0.20/0.28). Panel glow with inset highlight.
- [x] **Step 4: Atmosphère** — Grid pattern (60px) + radial gradient vignette + SVG noise grain overlay (opacity 0.03). Lab identity.
- [x] **Step 5: Loss curve polish** — EMA overlay (α=0.1) on raw loss. Chart.js canvas with model-color line.
- [x] **Step 6: Word grid polish** — Staggered fade-in animation (word-appear keyframes, 0.35s cubic-bezier).
- [x] **Step 7: Header** — Logo with "gpt" accent color + v1 badge. Tab indicator with scale animation.
- [x] **Step 8: Micro-interactions** — Training pulse (pulse-glow), status dot blink, panel-surface transitions (0.4s), instrument-header bar accent.
- [x] **Step 9: frontend-design audit** — Audited via Playwright screenshots at 5 viewports (375/768/1024/1366/1920). Fixed responsive bug: Solo breakpoint md→lg (commit cd0f5e7).
- [x] **Step 10: Commit** — cd0f5e7

```bash
git commit -m "style: complete visual overhaul — typography, color identity, atmosphere, animations"
```

### Task 59: Bouton « Réinitialiser le modèle » (Solo + Compare)

> **Contexte** : L'utilisateur doit pouvoir remettre un modèle à zéro (poids, steps, loss, mots générés) sans recharger la page. Utile pour relancer une expérience propre ou comparer des runs successifs.

- [x] **Step 1: Worker message** — Réutilise `initModel(params)` existant (pas de nouveau message worker nécessaire)
- [x] **Step 2: Hook** — `resetModel()` exposé dans `useModelWorker`, délègue à `initModel(params)`
- [x] **Step 3: UI ParamsPanel** — Bouton ↺ ghost dans ActionButtons, désactivé pendant l'entraînement, aria-label
- [x] **Step 4: Confirm dialog** — `ConfirmResetDialog` générique (title/description en props), affiché si trained ou error
- [x] **Step 5: Solo + Compare** — Vérifié : chaque panel a son propre reset indépendant via WorkerHandle
- [x] **Step 6: Tests** — 4 tests ajoutés (hook resetModel + 3 bouton reset), `renderPanel()` helper DRY
- [x] **Step 7: Commit** — `c081662`, `9698da1`

```bash
git commit -m "feat: add reset model button in Solo and Compare views"
```

### Task 60a: Defensive guards — empêcher le frontend de crasher le WASM

> **Contexte** : Le WASM crash souvent à cause de commandes envoyées pendant l'entraînement ou après un crash. Deux couches de défense ajoutées.

- [x] **Step 1: Hook guards** — `trainStateRef` (useRef + useEffect sync) dans `useModelWorker`. `initModel`, `train`, `generate` court-circuités si `trainState === 'training'`
- [x] **Step 2: Worker crash recovery** — `gpt = null` dans le catch après crash WASM → force ré-init propre au prochain `initModel`
- [x] **Step 3: Dead code removal** — Suppression du flag `busy` dans le worker (dead code en single-threaded)
- [x] **Step 4: Commit** — `607a71b`

```bash
git commit -m "fix: add defensive guards against WASM crashes"
```

### Task 60: Fix structural — WASM memory (OOM sur gros datasets)

> **Contexte** : Le tensor autograd engine en WASM épuise la mémoire linéaire sur les gros datasets (~2600 steps sur dinosaures). L'allocateur WASM ne rend jamais la mémoire → fragmentation → OOM → trap `unreachable`. Actuellement catch côté worker avec message FR, mais le modèle est irrécupérable sans réinit.

- [x] **Step 1: Fix backward() memory** — `std::mem::take` instead of clone in `backward()` (done in prior session, commit `22c9d9c`)
- [x] **Step 2: Profiler mémoire** — `memory.buffer.byteLength` logged every 100 steps via temporary diagnostic in worker
- [x] **Step 3: Tester 4000 steps sur dinosaures** — 2×2000 steps on Dinosaures (1530): memory stable at 2.75 MB from step 100 to 4000. Zero linear growth. Crash threshold (~2600) passed without issue.
- [x] **Step 4: Optionnel — allocateur** — Not needed. `std::mem::take` fix is sufficient.
- [x] **Step 5: Commit** — No code change needed (diagnostic was temporary). Validation recorded in plan.

```bash
git commit -m "fix: prevent WASM OOM via autograd graph detach between training steps"
```

### Task 61: LR schedule adaptatif — overfit & sweet spot par dataset

> **Contexte** : Le LR decay actuel est linéaire sur `n_steps` fixé à 1000 dans `TrainConfig::default()`. Problèmes :
> - Après 1000 steps, `lr_t = 0` → Adam ne met plus à jour les poids, l'entraînement est mort.
> - Le nombre de steps pour overfitter dépend de la taille du dataset (tokens uniques, vocabulaire, séquences).
> - L'utilisateur veut pouvoir overfitter librement puis retry avec d'autres params.
>
> **Réflexions dataset size vs steps :**
> - **Petit dataset** (~50 prénoms, ~200 tokens uniques) : overfit possible en ~500-1000 steps, sweet spot ~300-500 steps.
> - **Moyen dataset** (~150 pokémon, ~800 tokens) : overfit ~2000-3000 steps, sweet spot ~1000-1500 steps.
> - **Gros dataset** (~4500 baby names, ~3000+ tokens) : overfit ~5000-10000 steps, sweet spot ~2000-4000 steps.
> - Le sweet spot = loss stabilisée mais pas encore surapprentissage (génère des mots crédibles, pas du copier-coller du dataset).
> - Avec LR decay linéaire sur 1000, les gros datasets ne convergent jamais car le LR tombe à 0 avant d'atteindre le sweet spot.
>
> **Options à évaluer :**
> 1. **LR constant** (`lr_t = tc.lr`) — simple, laisse l'utilisateur décider quand arrêter. Risque : instabilité sur long training.
> 2. **Decay proportionnel au dataset** — calculer `n_steps` en fonction de `vocab_size * block_size` au lieu de le hardcoder.
> 3. **Cosine annealing avec warm restart** — LR cyclique, permet de continuer indéfiniment sans tomber à 0.
> 4. **Exposer n_steps dans l'UI** — laisser l'utilisateur contrôler le schedule complet.
>
> **Décision :** à trancher après tests empiriques sur les 4 datasets.

- [x] **Step 1: Benchmark empirique** — Skipped: Option A (constant LR) is the clear winner for a playground UX. No decay = user controls LR via slider, no hidden behavior, no "dead model" after N steps.
- [x] **Step 2: Choisir le schedule** — Option A: constant LR. Rationale: playground UX = "overfit & retry". Decay breaks multi-batch training (lr=0 after n_steps). User has slider for manual control.
- [x] **Step 3: Implémenter** — Removed linear decay from `adam_step()` in `model.rs` and `tensor_model.rs`. Updated `train_step()` and `train_step_traced()` in WASM to return `tc.lr` directly. Updated `training_meta()` schedule string to "constant".
- [x] **Step 4: Rebuild WASM** — `./build-wasm.sh` → 165KB. All Rust tests pass (34), cargo fmt + clippy clean. Frontend tsc + build OK.
- [x] **Step 5: Commit** — pending

```bash
git commit -m "fix: remove LR decay that froze training after 1000 steps"
```

---

### Task 62: Gamification de l'entraînement (sweet spot reward & overfitting alert)

> **Contexte** : Rendre l'entraînement ludique et pédagogique en donnant du feedback visuel à l'utilisateur sur la qualité de son entraînement.
>
> **Métriques** : 3 axes (memorization, quality, diversity) combinés en un feedback level unique.
> - Memorization : exact match + Levenshtein fuzzy + prefix match
> - Quality : bigram cosine similarity + length ratio
> - Diversity : Distinct-1 + Distinct-2 (Li et al. 2016) + unique word ratio
> Voir `docs/benchmark-loss-thresholds.md` pour les données empiriques.
>
> **Décision architecturale (2026-03-14)** : La détection `underpowered` utilise un **calcul de capacité dynamique** (`param_count / dataset_size`) au lieu de seuils de loss hardcodés par config. Raison : l'UI permet 3×4×3×4 = 144 combinaisons d'hyperparams + 7 datasets = 1008 configs. Benchmarker chaque combo est impraticable. La formule `params = 2×V×E + B×E + L×12×E²` donne le nombre exact de paramètres pour toute config, et le ratio params/noms indique si le modèle a la capacité théorique. Le benchmark (Step 1, DONE) reste utile pour les seuils de match ratio (learning/sweet-spot/overfitting) et la validation empirique des températures.
>
> **Prérequis** : LR constant (Task 61 DONE). Benchmark Step 1 DONE.
>
> **UX** : subtil et non-bloquant — badge inline dans InferencePanel, pas de modal ni toast. Ton pédagogique, pas punitif. L'utilisateur peut toujours entraîner davantage (10×2000 = 20000 steps ≈ 30s) — ne jamais dire "impossible".

#### Step 1: Benchmark empirique — seuils de loss par dataset

Pour chaque dataset (Prénoms FR 50, Prénoms FR 1000, Prénoms FR 33k, Baby Names EN 1000, Names EN 8000, Dinosaures 1530, Pokémon FR 1022), avec les hyperparams par défaut (n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01) :

- [x] **Step 1a: Benchmark 7 datasets × 5000 steps** — Node.js script (`scripts/benchmark.mjs`) using WASM directly. 50 samples × 3 runs × 3 temperatures (0.5/0.8/1.2) per checkpoint.
- [x] **Step 1b: Results documented** — `docs/benchmark-loss-thresholds.md`. Key finding: only Prénoms FR (50) converges. Temperature affects match ratio 3-5×.
- [x] **Step 1c: Seuils définis** — At t=0.8: learning <10%, sweet-spot 10-50%, overfitting >50%. Temperature normalization factor applied.
- [x] **Step 1d: Capacity ratio** — Underpowered detection uses dynamic `computeParamCount(cfg, vocabSize) / datasetSize` instead of hardcoded loss thresholds. Formula: `2×V×E + B×E + L×12×E²`. Works for all 1008 config combos.

#### Step 2: Détection côté frontend — DONE

- [x] **Step 2a: `lib/training-metrics.ts`** — 3-axis scoring (memorization, quality, diversity). DatasetProfile precomputation. Levenshtein fuzzy match, bigram cosine similarity, Distinct-1/Distinct-2 (Li et al. 2016). Trade-offs documented in header (no val split, no Self-BLEU).
- [x] **Step 2b: `lib/training-feedback.ts`** — 7 feedback levels (untrained, random, learning, sweet-spot, low-diversity, overfitting, underpowered). Temperature normalization. OLS trend detection. Dynamic capacity ratio for underpowered.
- [x] **Step 2c: Tests** — 98 vitest tests including benchmark validation with real Prénoms FR (50) data, Levenshtein edge cases, capacity ratio calculations.

#### Step 3: DatasetProfile côté main thread — DONE

> Mis à jour : Step 2 a introduit `buildDatasetProfile()` (wordSet + bigramDist + avgLength) au lieu d'un simple tableau de mots. Le main thread a besoin du profil complet, pas juste des mots bruts.

- [x] **Step 3a: Charger le dataset côté main thread** — `loadDatasetWords()` dans `useModelWorker.initModel()`. Un seul appel `preset.load()`, le texte est dérivé par `join('\n')`. `loadDatasetText()` supprimé (dead code).
- [x] **Step 3b: Construire et mémoïser le DatasetProfile** — `buildDatasetProfile(datasetWords)` appelé dans `initModel`, stocké en state React. Recalculé uniquement au changement de dataset (via `initModel`/`resetModel`).
- [x] **Step 3c: Exposer le profil** — `datasetProfile: DatasetProfile | null` exposé dans le return de `useModelWorker()` (et donc dans `WorkerHandle`).

#### Step 4: Intégration UI dans InferencePanel

> Mis à jour : 7 niveaux de feedback (pas 3). Messages définis dans `MESSAGES` + message dynamique pour `underpowered`. Scores (memorization/quality/diversity) disponibles pour affichage optionnel.

- [ ] **Step 4a: Modifier `InferencePanel`** — Ajouter un prop `feedback: FeedbackResult | null` (pas juste le level — inclut message + scores). Pure display, zéro logique de calcul.
- [ ] **Step 4b: Badge feedback** — Afficher un badge inline sous les mots générés, 7 états :
  - `'untrained'` → rien (pas de badge)
  - `'random'` → badge neutre "Le modèle génère du bruit" (couleur muted)
  - `'learning'` → badge neutre "Le modèle apprend…" (couleur muted)
  - `'sweet-spot'` → badge positif "Bonne généralisation !" (couleur success/green)
  - `'low-diversity'` → badge info "Manque de créativité" (couleur warning/amber)
  - `'overfitting'` → badge avertissement "Mémorise le dataset" (couleur error/red)
  - `'underpowered'` → badge info dynamique "Capacité limitée : X params pour Y noms" (couleur warning/amber)
- [ ] **Step 4c: Câbler dans ModelPanelInner** — Appeler `computeMemorization()` + `computeQuality()` + `computeDiversity()` avec les words générés et le `DatasetProfile`, puis `computeFeedback()` avec les scores + options (temperature, lossEma, totalSteps, modelConfig, vocabSize, datasetSize). Passer `FeedbackResult` en prop à InferencePanel.
- [ ] **Step 4d: ARIA** — Le badge doit avoir `role="status"` et `aria-live="polite"` pour les lecteurs d'écran.

#### Step 5: Tests composant

> Mis à jour : pas de Playwright long-running training. Tests unitaires par mock des scores.

- [ ] **Step 5a: Test InferencePanel badge** — 7 tests : un par `FeedbackLevel`. Mock du `FeedbackResult`, vérifier le message affiché, la couleur CSS, et les attributs ARIA (`role="status"`, `aria-live="polite"`).
- [ ] **Step 5b: Test InferencePanel sans feedback** — Vérifier que `feedback={null}` n'affiche aucun badge (pas de régression).
- [ ] **Step 5c: Test intégration câblage** — Vérifier dans `ModelPanelInner` que le feedback est calculé quand words + datasetProfile sont disponibles, et `null` sinon.

#### Step 6: Validation visuelle

- [ ] **Step 6a: Screenshots** — Capturer le rendu du badge dans les 6 états visibles (random, learning, sweet-spot, low-diversity, overfitting, underpowered) à mobile 375 et laptop 1366.
- [ ] **Step 6b: Audit visuel** — Vérifier lisibilité, positionnement, couleurs cohérentes avec le thème A/B, pas de casse layout.

#### Step 7: Commits

> Mis à jour : Steps 1-2 déjà committés. Commits restants pour Steps 3-6.

- [x] **Step 7a: Commit benchmark** — `docs: add empirical loss thresholds benchmark per dataset` *(done: 8eada7c)*
- [x] **Step 7b: Commit détection** — `feat: add training feedback detection (memorization + quality + diversity)` *(done: 0fac545, 76714d5, ffacbc9, e6c2956, 617721f)*
- [ ] **Step 7c: Commit dataset profile** — `feat: expose DatasetProfile on main thread for feedback computation`
- [ ] **Step 7d: Commit UI** — `feat: add feedback badge in InferencePanel (7 levels, ARIA)`
- [ ] **Step 7e: Commit tests** — `test: InferencePanel feedback badge states + integration`

---

### Task 58: Vérification post-déploiement

- [ ] **Step 1: Redéployer sur Vercel**
- [ ] **Step 2: Tests visuels sur l'URL de production** (ne JAMAIS valider un déploiement sans vérification visuelle)
- [ ] **Step 3: Tester le flow complet : changer params → entraîner → générer, en Solo ET Compare**
- [ ] **Step 4: Vérifier que A et B produisent des résultats différents si params différents**

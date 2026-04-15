# microgpt-lab

Parameter experimentation playground for microGPT — tweak hyperparameters, compare inferences side by side, powered by Rust/WASM.

## Project context

- **Org:** [mon-atelier-ia](https://github.com/mon-atelier-ia)
- **Related repos:** microgpt-rs (Rust model), microgpt-xray (scrollytelling visualizer), microgpt-visualizer-fr (React visualizer), microgpt-ts-fr (TypeScript GPT)
- **Reference:** [Karpathy microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## Stack

- **Frontend:** React 19 + Vite 7 + Tailwind CSS 4
- **Language:** TypeScript 5.9 (strict mode, `allowJs: false`)
- **Package manager:** pnpm
- **Quality:** ESLint 9 (complexity ≤10, cognitive ≤12, max-lines ≤300, max-fn ≤100) + sonarjs + @typescript-eslint + Prettier + jscpd
- **Model:** microgpt-rs compiled to WASM (`model-rs/`), imported from microgpt-xray

## Conventions

- Language: French for exchanges, English for code and technical docs
- Commits: conventional, atomic (`feat:`, `fix:`, `refactor:`, `docs:`, etc.)
- Git: never push without explicit request
- Always document findings in `docs/` BEFORE taking action

## Project structure

```
microgpt-lab/
├── app/                  # Frontend (React + Vite + Tailwind)
│   ├── src/
│   │   ├── components/
│   │   │   ├── model-panel/  # Triptyque: params, loss, inference
│   │   │   ├── ui/           # Shared UI (button, slider, select, alert-dialog, confirm-reset-dialog)
│   │   │   ├── top-bar.tsx
│   │   │   ├── solo-view.tsx
│   │   │   └── compare-view.tsx
│   │   ├── hooks/            # use-model-worker, use-loss-data, use-loss-chart-options
│   │   ├── workers/          # model-worker.ts (WASM bridge)
│   │   ├── lib/              # types, constants, utils(cn), model-colors, validation, dataset-loader, worker-utils, training-metrics, training-feedback
│   │   ├── data/             # Dataset presets (lazy-loaded)
│   │   └── theme/            # OKLCH tokens
│   ├── wasm-pkg/             # Pre-built WASM bindings
│   └── eslint.config.js      # Strict rules (complexity ≤10, max-lines ≤300)
├── model-rs/                 # Rust/WASM engine (from microgpt-xray)
│   ├── src/                  # Core: config, model, tensor, tensor_model, forward, inference
│   └── crates/microgpt-wasm/ # WASM bindings: lib.rs, trace.rs, trace_types.rs, training_trace.rs
├── docs/                     # Documentation (design, plan, architecture, SRP audit)
├── build-wasm.sh
└── package.json              # Root (husky + lint-staged)
```

## Commands

```bash
# Frontend
cd app && pnpm install    # Install dependencies
cd app && pnpm dev        # Dev server
cd app && pnpm build      # Production build

# WASM (requires wasm-pack)
./build-wasm.sh           # Rebuild WASM from Rust sources

# Rust
cd model-rs && cargo test --release    # Run Rust tests
cd model-rs && cargo fmt --check       # Check formatting
cd model-rs && cargo clippy -- -D warnings  # Lint
```

## Quality gates

**pre-commit** (staged files):
- ESLint --fix --max-warnings=0 (complexity ≤10, cognitive ≤12, max-lines ≤300, max-fn ≤100, max-depth ≤4)
- Prettier --write

**pre-push** (full project):
- `tsc --noEmit` — type-check
- `vite build` — production build
- `jscpd src/` — duplication detection (threshold 5%)

## Key rules

- **Zero Rust dependencies**: the model is std-only
- **Karpathy-faithful**: architecture matches the reference gist (deviations documented in `docs/reference-microgpt-karpathy.md` §13)
- Run `cargo fmt` and `cargo clippy -- -D warnings` before committing Rust changes

## Git rules — STRICT

- NEVER run `git push` without explicit user request
- NEVER run destructive git commands (force push, reset --hard, etc.)
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `style:`, `chore:`, `docs:`


## Wiki

Syntheses wiki : `C:\Dev\wiki\topics\ia\` (subdomain microgpt) et `C:\Dev\wiki\entities\microgpt-lab.md`

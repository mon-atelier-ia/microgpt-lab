# microgpt-lab

Parameter experimentation playground for [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — tweak hyperparameters, compare inferences side by side.

Powered by a pure Rust GPT engine compiled to WebAssembly for instant results in the browser.

**[Live demo](https://microgpt-lab.vercel.app)**

## Features

- **Solo mode** — train a single model, tune hyperparameters, observe loss and inference in real time
- **Compare mode** — run two models side by side with different configurations
- **Preset datasets** — baby names, dinosaurs, Pokémon, French first names (INSEE), English names
- **Live loss chart** — Chart.js visualization of training loss over epochs
- **Zero backend** — everything runs in-browser via Rust/WASM

## Part of the mon-atelier-ia ecosystem

| Project | Role |
|---------|------|
| [microgpt-rs](https://github.com/mon-atelier-ia/microgpt-rs) | Core Rust engine (zero deps) |
| [microgpt-xray](https://github.com/mon-atelier-ia/microgpt-xray) | Scrollytelling visualizer |
| [microgpt-visualizer-fr](https://github.com/mon-atelier-ia/microgpt-visualizer-fr) | Pedagogical visualizer |
| [microgpt-ts-fr](https://github.com/mon-atelier-ia/microgpt-ts-fr) | TypeScript playground |
| **microgpt-lab** | **Parameter experimentation** |

## Tech stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Vite 7, Tailwind CSS 4 |
| UI | Radix UI (slider, select, alert-dialog, tooltip), Lucide icons |
| Charts | Chart.js + react-chartjs-2 |
| ML engine | Rust (std-only) compiled to WASM via wasm-pack |
| Quality | ESLint 9 (strict), Prettier, jscpd, Husky |
| Tests | 108 Vitest + 5 Playwright |

## Quick start

```bash
cd app
pnpm install
pnpm dev
```

## Project structure

```
microgpt-lab/
├── app/                  # React + Vite frontend
│   ├── src/
│   │   ├── components/   # solo-view, compare-view, model-panel/, ui/
│   │   ├── hooks/        # use-model-worker, use-loss-data
│   │   ├── workers/      # WASM bridge (Web Worker)
│   │   ├── data/         # Dataset presets
│   │   └── theme/        # OKLCH design tokens
│   └── wasm-pkg/         # Pre-built WASM bindings
├── model-rs/             # Rust/WASM engine (from microgpt-rs)
└── docs/                 # Architecture & design docs
```

## Scripts

```bash
pnpm dev              # Dev server
pnpm build            # Production build
pnpm test             # Vitest unit tests
pnpm test:e2e         # Playwright E2E tests
pnpm lint             # ESLint (zero warnings)
```

## License

MIT

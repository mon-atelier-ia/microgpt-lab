# Design Decisions — microgpt-lab

> Aesthetic decisions implemented in CSS/theme. Companion to the design spec (`2026-03-10-microgpt-lab-design.md`).

**Date**: 2026-04-10

## Typography

**Geist + Geist Mono** (Vercel fonts). Not Space Grotesk (flagged as "AI default" in the ecosystem), not JetBrains Mono (reserved for microgpt-xray's oscilloscope identity). Geist is clean, geometric, modern — fits the "laboratory instrument" vibe without being as techno-heavy as JetBrains.

- `--font-sans: 'Geist', system-ui, sans-serif` — UI labels, body text
- `--font-mono: 'Geist Mono', ui-monospace, monospace` — numerical values, model output
- Weights: 400/500/600/700 imported via `@fontsource`

## Lab Grid Background

Three-layer texture creating a "laboratory graph paper" feel:

1. **Radial vignette** — blue tint at top center, fades to transparent (60% radius)
2. **60px grid** — horizontal + vertical lines at 0.3 opacity, subtle graph paper
3. **SVG noise grain** — `feTurbulence` fractalNoise at 3% opacity, fixed overlay via `body::after`

All layers use OKLCH colors with sRGB fallbacks.

## Panel Glow — Territorial Color Identity

Each model panel (A/B) has a luminous "territory" created by layered box-shadows:

```css
.panel-glow-a, .panel-glow-b {
  border: 1px solid color-mix(in oklch, var(--glow-muted), transparent 40%);
  background: color-mix(in oklch, var(--glow-muted), transparent 70%);
  box-shadow:
    0 0 30px -4px color-mix(in oklch, var(--glow-accent), transparent 75%),  /* inner glow */
    0 0 60px -8px color-mix(in oklch, var(--glow-muted), transparent 70%),   /* outer aura */
    inset 0 1px 0 0 color-mix(in oklch, var(--glow-muted), transparent 80%); /* top edge */
}
```

DRY via `--glow-accent` / `--glow-muted` CSS vars set per model. `color-mix(in oklch)` for all transparency — no rgba.

## OKLCH Tetradric Palette

Double complementary at ±15° shift:

| Token | OKLCH | Hue | Role |
|-------|-------|-----|------|
| `--model-a` | `oklch(0.65 0.15 220)` | 220° (blue) | Model A primary |
| `--model-a-accent` | `oklch(0.55 0.12 235)` | H+15° | Model A accent |
| `--model-b` | `oklch(0.7 0.15 40)` | 40° (orange) | Model B primary (complement) |
| `--model-b-accent` | `oklch(0.6 0.12 55)` | H+15° | Model B accent |

Surface elevation: 8% lightness jumps (`0.12 → 0.20 → 0.28`), all at hue 260° with minimal chroma.

## Micro-animations (CSS-only)

No GSAP — all animations are pure CSS for performance:

| Animation | Target | Duration | Easing |
|-----------|--------|----------|--------|
| `word-appear` | Generated words (stagger) | 0.35s | `cubic-bezier(0.22, 1, 0.36, 1)` (spring) |
| `tab-scale` | Active tab indicator | 0.25s | spring |
| `pulse-glow` | Train button during training | 1.5s | ease-in-out, infinite |
| `status-blink` | Status dot (training active) | 1s | ease-in-out, infinite |

Panel transitions (box-shadow, border, background): 0.4s ease.

## Instrument Header

Vertical color bar (3px × 14px, `border-radius: 1px`) before panel titles via `::before`. Creates an oscilloscope readout aesthetic — each instrument section is "tagged" with its model's accent color.

## Progressive Enhancement

Every OKLCH value has a sRGB fallback:

```css
:root { --model-a: #3b8edb; }           /* fallback */
@supports (color: oklch(0 0 0)) {
  :root { --model-a: oklch(0.65 0.15 220); } /* enhancement */
}
```

Shadcn/ui compatibility: all project tokens are mapped to shadcn's expected `--color-*` namespace in `@theme` block.

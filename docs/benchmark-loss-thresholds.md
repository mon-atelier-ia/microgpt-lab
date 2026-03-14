<!-- LLM-NAV
doc-id: benchmark-loss-thresholds
status: Active
related: lab-design, srp-audit
summary: Empirical loss EMA + match ratio benchmarks per dataset for gamification thresholds
-->

# Benchmark — Loss Thresholds per Dataset

**Date**: 2026-03-14
**Config**: n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01
**Method**: 5000 steps per dataset, 50 samples × 3 runs per checkpoint (150 words), temperatures 0.5/0.8/1.2. Random baseline computed per dataset (200 random character strings matched against dataset).
**Script**: `scripts/benchmark.mjs`

---

## Raw Results

### Prénoms FR (50) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 | Samples (t=0.8) |
|------|----------|-------|-------|-------|-----------------|
| 200 | 2.1828 | 3% | 1% | 1% | arccstics, n, geand, maauelice, kanid |
| 500 | 1.5975 | 23% | 11% | 4% | dari, juliel, anrtic, jacques, palal |
| 1000 | 1.1458 | 61% | 42% | 24% | pare, rene, re, jedereinnne, david |
| 2000 | 1.0621 | 74% | 45% | 27% | stephane, ricannc, phephae, jeannlte, joseph |
| 3000 | 0.9282 | 62% | 58% | 44% | jene, jacques, anne, pierre, matheric |
| 5000 | 0.9187 | 66% | 56% | 41% | philippe, alaopis, elarric, aean, dathale |

**Analysis**: Converges fast. At t=0.5, hits 74% at step 2000 — close to overfitting but never reaches 80%+ because the model still makes spelling errors. At t=0.8, plateaus ~55% at 3000+. At t=1.2, plateaus ~40%. True overfitting (>80%) may require 10000+ steps or isn't achievable with this model size.

### Prénoms FR (1000) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200 | 2.6512 | 4% | 1% | 0% |
| 500 | 2.2090 | 6% | 3% | 0% |
| 1000 | 2.1741 | 5% | 1% | 1% |
| 2000 | 2.1667 | 9% | 1% | 1% |
| 3000 | 2.2141 | 16% | 5% | 1% |
| 5000 | 2.2011 | 11% | 9% | 2% |

**Analysis**: Barely learning. At t=0.5, reaches 16% at 3000 then drops. At t=0.8, stays <10%. Model too small for 1000 unique names.

### Baby Names EN (1000) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200-5000 | ~2.3-2.5 | 1-2% | 0-1% | 0-1% |

**Analysis**: No learning. Loss plateaus. Match ratio indistinguishable from noise.

### Dinosaures (1530) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200-5000 | ~1.8-2.1 | 0-1% | 0% | 0% |

**Analysis**: Loss decreases slightly (learns morphology like "-saurus" suffix) but never generates real dinosaur names. Names too long and varied.

### Pokémon FR (1022) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200-5000 | ~2.5-2.8 | 0% | 0% | 0% |

**Analysis**: Zero matches at any checkpoint or temperature. Pokémon names have complex mixed-case patterns, hyphens, accents. Completely out of reach.

### Names EN (8000) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200 | 2.3017 | 25% | 7% | 3% |
| 500 | 2.2670 | 28% | 11% | 2% |
| 1000 | 2.3003 | 18% | 7% | 2% |
| 2000 | 2.3753 | 15% | 11% | 5% |
| 3000 | 2.2877 | 23% | 5% | 6% |
| 5000 | 2.2056 | 31% | 11% | 7% |

**Analysis**: Surprising — 25-31% at t=0.5 despite no real convergence (loss ~2.2-2.3). This is because with 8000 short English names, many 3-4 letter combinations happen to match real names. The model learns basic English phonotactics early, and short generated names accidentally match. High variance between checkpoints confirms this is partially stochastic. NOT the same as "good learning" — the model isn't memorizing names, it's generating plausible short strings that happen to be names.

### Prénoms FR (33k) — random baseline: 0.0%

| Step | Loss EMA | t=0.5 | t=0.8 | t=1.2 |
|------|----------|-------|-------|-------|
| 200 | 2.5637 | 16% | 7% | 2% |
| 500 | 2.3541 | 39% | 16% | 7% |
| 1000 | 2.4987 | 25% | 9% | 4% |
| 2000 | 2.1565 | 25% | 8% | 5% |
| 3000 | 2.3167 | 27% | 11% | 5% |
| 5000 | 2.4519 | 23% | 11% | 5% |

**Analysis**: Same phenomenon as Names EN (8000). 33k names = huge coverage. Generated strings accidentally match because the dataset covers so many name variants. At t=0.5, the "match" is mostly accidental overlap with a massive dataset, not memorization. The loss doesn't converge (stays ~2.2-2.5).

---

## Key Findings

### 1. Temperature dramatically affects match ratio

At the same training step, match ratio varies 3-5× between t=0.5 and t=1.2. **Any threshold system MUST account for the user's chosen temperature.**

| Temperature | Effect on match ratio |
|-------------|----------------------|
| 0.5 | High — concentrated distribution picks "safe" common patterns |
| 0.8 | Medium — default, balanced |
| 1.2 | Low — diverse, more creative, fewer exact matches |

### 2. Only 1 of 7 datasets converges meaningfully

Prénoms FR (50) is the only dataset where the model genuinely learns to reproduce names. All others either:
- Don't converge (1000+ names = too many for 4192 params)
- Show accidental matches (8000+ names = so many that random strings match)

### 3. Overfitting threshold (>80%) is never reached

Even Prénoms FR (50) at 5000 steps never exceeds 74% at t=0.5. The model makes enough spelling errors to stay below 80%. The overfitting threshold should be **lowered to 70%** or adjusted per temperature.

### 4. Large datasets need a different feedback message

For datasets where the model can't converge, the feedback shouldn't say "garbage" — it should explain WHY: "Ce dataset est trop grand pour ce modèle (4192 paramètres). Essayez un plus petit dataset ou augmentez n_embd/n_layer."

---

## Threshold Definition

Based on data at **t=0.8** (default):

| FeedbackLevel | Match ratio (t=0.8) | Condition |
|---------------|---------------------|-----------|
| `none` | — | No generation yet |
| `learning` | 0-10% | Model producing character patterns, not real words |
| `sweet-spot` | 10-50% | Mix of real and invented words — generalization |
| `overfitting` | >50% | Majority are dataset copies (at t=0.8, >50% means near-memorization) |

**Temperature adjustment**: the thresholds above are for t=0.8. For other temperatures:
- t≤0.5: multiply thresholds by 1.5× (higher match is expected)
- t≥1.2: multiply thresholds by 0.5× (lower match is expected)

**Large dataset caveat**: for datasets with 5000+ words AND loss EMA > 2.0, match ratio is likely accidental overlap, not learning. In this case, show "learning" regardless of match ratio.

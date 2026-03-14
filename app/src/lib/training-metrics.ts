/**
 * Training evaluation metrics — industry-standard ML metrics adapted for
 * character-level name generation.
 *
 * Three independent axes, each scored 0-1:
 *   - Memorization: is the model copying the dataset?
 *   - Quality: are generated words linguistically plausible?
 *   - Diversity: is the model creative or collapsing?
 *
 * References:
 *   - Levenshtein fuzzy match: standard memorization detection (Carlini et al., MemHunter)
 *   - Distinct-1/Distinct-2: Li et al. 2016 "A Diversity-Promoting Objective Function"
 *   - Bigram cosine similarity: standard corpus similarity metric
 *
 * Known trade-offs vs full ML eval:
 *   - No train/val split — WASM engine doesn't support held-out validation.
 *     Match ratio against full dataset is used as overfitting proxy.
 *   - No Self-BLEU — overkill for 3-10 char words. Distinct-2 + unique ratio
 *     is sufficient for diversity measurement at this scale.
 *   - Loss EMA ≈ log(perplexity) — used in feedback but not as an explicit
 *     perplexity metric here. The relationship is exact for cross-entropy loss.
 *
 * All functions are pure — no side effects, no hooks.
 */

// ── Dataset Profile ─────────────────────────────────────────────────

/**
 * Pre-computed dataset properties. Build once per dataset with
 * `buildDatasetProfile()` and reuse across all metric calls.
 */
export type DatasetProfile = {
  /** Lowercase words from the dataset. */
  words: string[];
  /** Set of lowercase words for O(1) exact-match lookup. */
  wordSet: Set<string>;
  /** Bigram frequency distribution (Map<bigram, frequency>). */
  bigramDist: Map<string, number>;
  /** Average word length. */
  avgLength: number;
};

/** Build a reusable dataset profile from raw dataset words. */
export function buildDatasetProfile(datasetWords: string[]): DatasetProfile {
  const words = datasetWords.map((w) => w.toLowerCase());
  const wordSet = new Set(words);
  const bigramDist = bigramDistribution(words);
  const avgLength = words.length > 0 ? words.reduce((s, w) => s + w.length, 0) / words.length : 0;
  return { words, wordSet, bigramDist, avgLength };
}

// ── Levenshtein ─────────────────────────────────────────────────────

/** Levenshtein edit distance between two strings. */
function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[] = Array.from({ length: n + 1 }, (_, i) => i);

  for (let i = 1; i <= m; i++) {
    let prev = i - 1;
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, dp[j], dp[j - 1]);
      prev = tmp;
    }
  }
  return dp[n];
}

/**
 * Normalized Levenshtein distance (0 = identical, 1 = completely different).
 * Exported for direct testing (Issue 5).
 */
export function normalizedEditDistance(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 0;
  return levenshtein(a, b) / maxLen;
}

// ── Memorization Score ──────────────────────────────────────────────

/** Check if word has a fuzzy match (edit distance ≤ 20%) in dataset. */
function hasFuzzyMatch(word: string, dsArray: string[]): boolean {
  for (const dw of dsArray) {
    if (Math.abs(dw.length - word.length) > 2) continue;
    if (normalizedEditDistance(word, dw) <= 0.2) return true;
  }
  return false;
}

/** Check if word starts with any dataset word >= 3 chars. */
function hasPrefixMatch(word: string, dsArray: string[]): boolean {
  return dsArray.some((dw) => dw.length >= 3 && word.startsWith(dw));
}

/** Classify a generated word: 'exact' | 'fuzzy' | 'prefix' | 'novel'. */
function classifyWord(
  word: string,
  dsSet: Set<string>,
  dsArray: string[],
): 'exact' | 'fuzzy' | 'prefix' | 'novel' {
  if (dsSet.has(word)) return 'exact';
  if (hasFuzzyMatch(word, dsArray)) return 'fuzzy';
  if (hasPrefixMatch(word, dsArray)) return 'prefix';
  return 'novel';
}

/**
 * Memorization score (0-1). Higher = more memorization.
 *
 * Weighted combination:
 *   - 0.5 x exact match ratio
 *   - 0.3 x fuzzy match ratio (Levenshtein distance <= 0.2 of word length)
 *   - 0.2 x prefix match ratio (generated word starts with a dataset word >= 3 chars)
 */
export function computeMemorization(words: string[], profile: DatasetProfile): number {
  if (words.length === 0) return 0;

  const classes = words.map((w) => classifyWord(w.toLowerCase(), profile.wordSet, profile.words));

  const n = words.length;
  const exact = classes.filter((c) => c === 'exact').length;
  const fuzzy = classes.filter((c) => c === 'fuzzy').length;
  const prefix = classes.filter((c) => c === 'prefix').length;

  return 0.5 * (exact / n) + 0.3 * (fuzzy / n) + 0.2 * (prefix / n);
}

// ── Quality Score ───────────────────────────────────────────────────

/** Extract character bigrams from a word. */
function charBigrams(word: string): string[] {
  const bigrams: string[] = [];
  for (let i = 0; i < word.length - 1; i++) {
    bigrams.push(word.slice(i, i + 2));
  }
  return bigrams;
}

/** Build bigram frequency distribution from word list. Returns Map<bigram, frequency>. */
function bigramDistribution(words: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  let total = 0;
  for (const w of words) {
    for (const bg of charBigrams(w.toLowerCase())) {
      counts.set(bg, (counts.get(bg) ?? 0) + 1);
      total++;
    }
  }
  if (total > 0) {
    for (const [k, v] of counts) counts.set(k, v / total);
  }
  return counts;
}

/** Cosine similarity between two bigram frequency distributions. */
function bigramCosineSimilarity(genDist: Map<string, number>, dsDist: Map<string, number>): number {
  const allBigrams = new Set([...genDist.keys(), ...dsDist.keys()]);
  let dotProduct = 0;
  let normGen = 0;
  let normDs = 0;

  for (const bg of allBigrams) {
    const g = genDist.get(bg) ?? 0;
    const d = dsDist.get(bg) ?? 0;
    dotProduct += g * d;
    normGen += g * g;
    normDs += d * d;
  }

  const denom = Math.sqrt(normGen) * Math.sqrt(normDs);
  return denom > 0 ? dotProduct / denom : 0;
}

/**
 * Per-word bigram coverage: for each word, what fraction of its character
 * bigrams exist in the dataset? Words with < 2 chars score 0 (no bigrams).
 * Returns the average across ALL words (including 0-scoring short ones).
 */
function perWordBigramCoverage(words: string[], dsDist: Map<string, number>): number {
  let total = 0;
  for (const w of words) {
    const bgs = charBigrams(w.toLowerCase());
    if (bgs.length === 0) continue; // scores 0, doesn't add to total
    const covered = bgs.filter((b) => dsDist.has(b)).length;
    total += covered / bgs.length;
  }
  return words.length > 0 ? total / words.length : 0;
}

/**
 * Quality score (0-1). Higher = more linguistically plausible.
 *
 * Formula (tuned against Prenoms FR benchmark — see training-metrics-benchmark.test.ts):
 *   score = pwCoverage^1.5 x cosineSim + 0.08 x lenScore
 *
 * - pwCoverage: per-word bigram coverage gates the score — words containing
 *   alien bigrams (not in dataset) are penalized aggressively via the ^1.5 exponent.
 * - cosineSim: cosine similarity between generated and dataset bigram distributions.
 * - lenScore: small additive bonus for matching dataset word lengths.
 *
 * The multiplicative (pwCov^1.5 x cos) structure was chosen over a linear
 * combination because cosine similarity alone cannot distinguish garbage from
 * plausible output when the dataset covers many character bigrams (e.g., 50
 * French names). The per-word gating ensures that even one alien bigram per
 * word drags quality down.
 */
export function computeQuality(words: string[], profile: DatasetProfile): number {
  if (words.length === 0 || profile.words.length === 0) return 0;

  const genDist = bigramDistribution(words);
  const cosineSim = bigramCosineSimilarity(genDist, profile.bigramDist);
  const pwCov = perWordBigramCoverage(words, profile.bigramDist);

  // Length ratio: penalize deviation from dataset avg length
  const avgGenLen = words.reduce((s, w) => s + w.length, 0) / words.length;
  const lenRatio = profile.avgLength > 0 ? avgGenLen / profile.avgLength : 0;
  const lenScore = Math.max(0, 1 - Math.abs(1 - lenRatio));

  return Math.pow(pwCov, 1.5) * cosineSim + 0.08 * lenScore;
}

// ── Diversity Score ─────────────────────────────────────────────────

/**
 * Diversity score (0-1). Higher = more creative/diverse.
 *
 * Weighted combination:
 *   - 0.4 x Distinct-1: unique unigrams / total unigrams (Li et al. 2016)
 *   - 0.3 x Distinct-2: unique char bigrams / total char bigrams
 *   - 0.3 x unique word ratio (unique words / total words)
 */
export function computeDiversity(words: string[]): number {
  if (words.length === 0) return 0;

  const lower = words.map((w) => w.toLowerCase());

  // Distinct-1 (Li et al. 2016): unique unigrams / total unigrams
  const chars = new Set<string>();
  let totalChars = 0;
  for (const w of lower) {
    for (const c of w) {
      chars.add(c);
      totalChars++;
    }
  }
  const distinct1 = totalChars > 0 ? chars.size / totalChars : 0;

  // Distinct-2: unique character bigrams / total character bigrams
  const bigrams = new Set<string>();
  let totalBigrams = 0;
  for (const w of lower) {
    for (const bg of charBigrams(w)) {
      bigrams.add(bg);
      totalBigrams++;
    }
  }
  const distinct2 = totalBigrams > 0 ? bigrams.size / totalBigrams : 0;

  // Unique word ratio
  const uniqueWords = new Set(lower);
  const uniqueRatio = uniqueWords.size / lower.length;

  return 0.4 * distinct1 + 0.3 * distinct2 + 0.3 * uniqueRatio;
}

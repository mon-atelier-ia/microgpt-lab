/**
 * Training evaluation metrics — industry-standard ML metrics adapted for
 * character-level name generation.
 *
 * Three independent axes, each scored 0-1:
 *   - Memorization: is the model copying the dataset?
 *   - Quality: are generated words linguistically plausible?
 *   - Diversity: is the model creative or collapsing?
 *
 * All functions are pure — no side effects, no hooks.
 */

// ── Memorization Score ──────────────────────────────────────────────

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

/** Normalized Levenshtein distance (0 = identical, 1 = completely different). */
function normalizedEditDistance(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 0;
  return levenshtein(a, b) / maxLen;
}

/** Check if word has a fuzzy match (edit distance ≤ 20%) in dataset. */
function hasFuzzyMatch(word: string, dsArray: string[]): boolean {
  for (const dw of dsArray) {
    if (Math.abs(dw.length - word.length) > 2) continue;
    if (normalizedEditDistance(word, dw) <= 0.2) return true;
  }
  return false;
}

/** Check if word starts with any dataset word ≥ 3 chars. */
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
 *   - 0.5 × exact match ratio
 *   - 0.3 × fuzzy match ratio (Levenshtein distance ≤ 0.2 of word length)
 *   - 0.2 × prefix match ratio (generated word starts with a dataset word ≥ 3 chars)
 */
export function computeMemorization(words: string[], datasetWords: string[]): number {
  if (words.length === 0) return 0;

  const dsSet = new Set(datasetWords.map((w) => w.toLowerCase()));
  const dsArray = datasetWords.map((w) => w.toLowerCase());
  const classes = words.map((w) => classifyWord(w.toLowerCase(), dsSet, dsArray));

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
  // Normalize to frequencies
  if (total > 0) {
    for (const [k, v] of counts) counts.set(k, v / total);
  }
  return counts;
}

/**
 * Quality score (0-1). Higher = more linguistically plausible.
 *
 * Weighted combination:
 *   - 0.7 × bigram plausibility (cosine similarity between generated bigram
 *     distribution and dataset bigram distribution)
 *   - 0.3 × length ratio (penalize if avg generated length deviates >50% from dataset)
 */
export function computeQuality(words: string[], datasetWords: string[]): number {
  if (words.length === 0 || datasetWords.length === 0) return 0;

  // Bigram plausibility: cosine similarity
  const genDist = bigramDistribution(words);
  const dsDist = bigramDistribution(datasetWords);

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
  const cosineSim = denom > 0 ? dotProduct / denom : 0;

  // Length ratio: penalize deviation from dataset avg length
  const avgGenLen = words.reduce((s, w) => s + w.length, 0) / words.length;
  const avgDsLen = datasetWords.reduce((s, w) => s + w.length, 0) / datasetWords.length;
  const lenRatio = avgDsLen > 0 ? avgGenLen / avgDsLen : 0;
  // Score 1.0 when ratio = 1.0, drops to 0 when ratio deviates >100%
  const lenScore = Math.max(0, 1 - Math.abs(1 - lenRatio));

  return 0.7 * cosineSim + 0.3 * lenScore;
}

// ── Diversity Score ─────────────────────────────────────────────────

/**
 * Diversity score (0-1). Higher = more creative/diverse.
 *
 * Weighted combination:
 *   - 0.4 × Distinct-1 (unique characters used / total vocab)
 *   - 0.3 × Distinct-2 (unique char bigrams / total char bigrams) — Li et al. 2016
 *   - 0.3 × unique word ratio (unique words / total words)
 */
export function computeDiversity(words: string[], vocabSize: number): number {
  if (words.length === 0) return 0;

  const lower = words.map((w) => w.toLowerCase());

  // Distinct-1: unique characters
  const chars = new Set<string>();
  for (const w of lower) {
    for (const c of w) chars.add(c);
  }
  // Normalize by vocab size (minus BOS token)
  const distinct1 = vocabSize > 1 ? chars.size / (vocabSize - 1) : 0;

  // Distinct-2: unique character bigrams
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

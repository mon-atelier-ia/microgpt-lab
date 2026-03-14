/**
 * Training feedback system — combines metrics into actionable feedback levels.
 *
 * Uses memorization, quality, and diversity scores with temperature normalization
 * and trend detection to produce pedagogically useful feedback.
 */

export type FeedbackLevel =
  | 'untrained'
  | 'random'
  | 'learning'
  | 'sweet-spot'
  | 'low-diversity'
  | 'overfitting'
  | 'underpowered';

export type FeedbackResult = {
  level: FeedbackLevel;
  message: string;
  memorization: number;
  quality: number;
  diversity: number;
};

export type MetricScores = {
  memorization: number;
  quality: number;
  diversity: number;
};

/**
 * Temperature normalization factor for memorization thresholds.
 *
 * Lower temperatures → higher match ratios (more conservative generation).
 * We normalize so that thresholds are relative to t=0.8 (default).
 *
 * Based on empirical benchmark data (docs/benchmark-loss-thresholds.md):
 *   - t=0.5: match ratio ~1.5× higher than t=0.8
 *   - t=1.2: match ratio ~0.5× lower than t=0.8
 */
function temperatureNormFactor(temperature: number): number {
  // Linear interpolation: t=0.5 → 1.5, t=0.8 → 1.0, t=1.2 → 0.6
  // Clamped to [0.4, 2.0] for extreme temperatures
  const factor = 1.0 + (0.8 - temperature) * (5 / 3);
  return Math.max(0.4, Math.min(2.0, factor));
}

/** Normalize memorization score by temperature. */
function normalizeMemorization(raw: number, temperature: number): number {
  return raw / temperatureNormFactor(temperature);
}

type TrendEntry = MetricScores;

/**
 * Detect overfitting trend: memorization increasing AND diversity decreasing
 * over consecutive generations.
 */
export function detectOverfittingTrend(history: TrendEntry[]): boolean {
  if (history.length < 3) return false;

  const recent = history.slice(-3);
  const memIncreasing =
    recent[1].memorization > recent[0].memorization &&
    recent[2].memorization > recent[1].memorization;
  const divDecreasing =
    recent[1].diversity < recent[0].diversity && recent[2].diversity < recent[1].diversity;

  return memIncreasing && divDecreasing;
}

export type FeedbackOptions = {
  temperature: number;
  lossEma: number | null;
  totalSteps: number;
  wordCount: number;
  trendHistory?: TrendEntry[];
};

const MESSAGES: Record<FeedbackLevel, string> = {
  untrained: '',
  random: 'Le modèle génère du bruit — entraînez-le davantage.',
  learning: 'Le modèle apprend les patterns du langage\u2026',
  'sweet-spot': 'Bonne généralisation !',
  'low-diversity': 'Le modèle manque de créativité — essayez d\u2019augmenter la température.',
  overfitting: 'Le modèle mémorise le dataset — il ne généralise plus.',
  underpowered: 'Ce dataset est trop grand pour ce modèle — augmentez n_embd ou n_layer.',
};

function isUnderpowered(scores: MetricScores, opts: FeedbackOptions): boolean {
  return (
    scores.quality < 0.3 && opts.totalSteps >= 1000 && opts.lossEma !== null && opts.lossEma > 2.0
  );
}

function isSweetSpot(scores: MetricScores, normMem: number): boolean {
  return scores.quality > 0.4 && normMem >= 0.1 && normMem <= 0.5 && scores.diversity > 0.4;
}

/**
 * Compute feedback from metric scores.
 *
 * Decision tree (evaluated in priority order):
 *   1. No words → untrained
 *   2. Quality < 0.2 → random (noise)
 *   3. Memorization (temp-adjusted) > 0.5 OR overfitting trend → overfitting
 *   4. Diversity < 0.3 → low-diversity (mode collapse)
 *   5. Underpowered check → underpowered
 *   6. Sweet-spot check → sweet-spot
 *   7. Default → learning
 */
export function computeFeedback(scores: MetricScores, opts: FeedbackOptions): FeedbackResult {
  const base = { ...scores };

  if (opts.wordCount === 0) return { level: 'untrained', message: '', ...base };

  const normMem = normalizeMemorization(scores.memorization, opts.temperature);
  const hasTrend = detectOverfittingTrend(opts.trendHistory ?? []);

  let level: FeedbackLevel = 'learning';

  if (scores.quality < 0.2) level = 'random';
  else if (normMem > 0.5 || hasTrend) level = 'overfitting';
  else if (scores.diversity < 0.3) level = 'low-diversity';
  else if (isUnderpowered(scores, opts)) level = 'underpowered';
  else if (isSweetSpot(scores, normMem)) level = 'sweet-spot';

  return { level, message: MESSAGES[level], ...base };
}

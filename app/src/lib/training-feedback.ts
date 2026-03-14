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
 * Lower temperatures -> higher match ratios (more conservative generation).
 * We normalize so that thresholds are relative to t=0.8 (default).
 *
 * Based on empirical benchmark data (docs/benchmark-loss-thresholds.md):
 *   - t=0.5: match ratio ~1.5x higher than t=0.8
 *   - t=1.2: match ratio ~0.5x lower than t=0.8
 */
function temperatureNormFactor(temperature: number): number {
  // Linear interpolation: t=0.5 -> 1.5, t=0.8 -> 1.0, t=1.2 -> 0.6
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
 * Compute linear regression slope for an array of numeric values.
 * For N points indexed 0..N-1, returns the OLS slope.
 */
function linearSlope(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  const meanI = (n - 1) / 2;
  const meanY = values.reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const di = i - meanI;
    num += di * (values[i] - meanY);
    den += di * di;
  }
  return den > 0 ? num / den : 0;
}

/**
 * Detect overfitting trend using linear regression over the last 3+ entries.
 *
 * Instead of strict monotonicity, we compute the OLS slope of memorization
 * and diversity. This handles noise like 0.1 -> 0.3 -> 0.29 -> 0.5.
 *
 * Returns true if memorization slope > 0.02 AND diversity slope < -0.02.
 */
export function detectOverfittingTrend(history: TrendEntry[]): boolean {
  if (history.length < 3) return false;

  const recent = history.slice(-3);
  const memValues = recent.map((e) => e.memorization);
  const divValues = recent.map((e) => e.diversity);

  const memSlope = linearSlope(memValues);
  const divSlope = linearSlope(divValues);

  return memSlope > 0.02 && divSlope < -0.02;
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
  random: 'Le mod\u00e8le g\u00e9n\u00e8re du bruit \u2014 entra\u00eenez-le davantage.',
  learning: 'Le mod\u00e8le apprend les patterns du langage\u2026',
  'sweet-spot': 'Bonne g\u00e9n\u00e9ralisation !',
  'low-diversity':
    'Le mod\u00e8le manque de cr\u00e9ativit\u00e9 \u2014 essayez d\u2019augmenter la temp\u00e9rature.',
  overfitting: 'Le mod\u00e8le m\u00e9morise le dataset \u2014 il ne g\u00e9n\u00e9ralise plus.',
  underpowered: 'Ce dataset est trop grand pour ce mod\u00e8le \u2014 augmentez n_embd ou n_layer.',
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
 *   1. No words -> untrained
 *   2. Quality < 0.2 -> random (noise)
 *   3. Memorization (temp-adjusted) > 0.5 OR overfitting trend -> overfitting
 *   4. Diversity < 0.3 -> low-diversity (mode collapse)
 *   5. Underpowered check -> underpowered
 *   6. Sweet-spot check -> sweet-spot
 *   7. Default -> learning
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

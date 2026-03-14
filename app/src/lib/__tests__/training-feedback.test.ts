import { describe, it, expect } from 'vitest';
import {
  computeFeedback,
  computeParamCount,
  computeCapacityRatio,
  detectOverfittingTrend,
} from '../training-feedback';
import type { MetricScores } from '../training-feedback';

const smallModel = { n_embd: 16, n_head: 4, n_layer: 1, block_size: 16 };
const bigModel = { n_embd: 32, n_head: 8, n_layer: 4, block_size: 16 };

const defaultOpts = {
  temperature: 0.8,
  lossEma: 1.5,
  totalSteps: 500,
  wordCount: 10,
  datasetSize: 50,
  vocabSize: 27,
  modelConfig: smallModel,
};

describe('computeFeedback', () => {
  it('returns untrained when no words', () => {
    const scores: MetricScores = { memorization: 0, quality: 0, diversity: 0 };
    const result = computeFeedback(scores, { ...defaultOpts, wordCount: 0 });
    expect(result.level).toBe('untrained');
  });

  it('returns random when quality < 0.2', () => {
    const scores: MetricScores = { memorization: 0, quality: 0.1, diversity: 0.5 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.level).toBe('random');
  });

  it('returns overfitting when memorization > 0.5', () => {
    const scores: MetricScores = { memorization: 0.6, quality: 0.7, diversity: 0.5 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.level).toBe('overfitting');
  });

  it('returns overfitting when trend detected', () => {
    const scores: MetricScores = { memorization: 0.3, quality: 0.5, diversity: 0.5 };
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.8 },
      { memorization: 0.2, quality: 0.5, diversity: 0.6 },
      { memorization: 0.3, quality: 0.5, diversity: 0.4 },
    ];
    const result = computeFeedback(scores, { ...defaultOpts, trendHistory: trend });
    expect(result.level).toBe('overfitting');
  });

  it('returns low-diversity when diversity < 0.3', () => {
    const scores: MetricScores = { memorization: 0.2, quality: 0.5, diversity: 0.2 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.level).toBe('low-diversity');
  });

  it('returns underpowered when capacity ratio is low', () => {
    const scores: MetricScores = { memorization: 0.05, quality: 0.2, diversity: 0.5 };
    // Small model (4192 params) vs 1000 names → ratio ~4.2 < 10
    const opts = { ...defaultOpts, totalSteps: 1500, datasetSize: 1000 };
    const result = computeFeedback(scores, opts);
    expect(result.level).toBe('underpowered');
    expect(result.message).toContain('4192');
    expect(result.message).toContain('1000');
  });

  it('does not return underpowered when capacity ratio is high', () => {
    const scores: MetricScores = { memorization: 0.05, quality: 0.2, diversity: 0.5 };
    // Small model vs 50 names → ratio ~83 > 10
    const opts = { ...defaultOpts, totalSteps: 1500, datasetSize: 50 };
    const result = computeFeedback(scores, opts);
    expect(result.level).not.toBe('underpowered');
  });

  it('does not return underpowered with big model even on large dataset', () => {
    const scores: MetricScores = { memorization: 0.05, quality: 0.2, diversity: 0.5 };
    // Big model (51k+ params) vs 1000 names → ratio ~51 > 10
    const opts = { ...defaultOpts, totalSteps: 1500, datasetSize: 1000, modelConfig: bigModel };
    const result = computeFeedback(scores, opts);
    expect(result.level).not.toBe('underpowered');
  });

  it('returns sweet-spot when balanced', () => {
    const scores: MetricScores = { memorization: 0.3, quality: 0.6, diversity: 0.6 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.level).toBe('sweet-spot');
  });

  it('returns learning as default', () => {
    const scores: MetricScores = { memorization: 0.05, quality: 0.35, diversity: 0.5 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.level).toBe('learning');
  });

  // Temperature normalization tests
  it('adjusts memorization threshold for low temperature', () => {
    // At t=0.5, raw memorization 0.6 is normalized down to ~0.4 (below 0.5 threshold)
    const scores: MetricScores = { memorization: 0.6, quality: 0.6, diversity: 0.6 };
    const result = computeFeedback(scores, { ...defaultOpts, temperature: 0.5 });
    // 0.6 / 1.5 = 0.4 -> not overfitting, could be sweet-spot
    expect(result.level).not.toBe('overfitting');
  });

  it('adjusts memorization threshold for high temperature', () => {
    // At t=1.2, raw memorization 0.3 is normalized UP to ~0.5 (hits threshold)
    const scores: MetricScores = { memorization: 0.3, quality: 0.5, diversity: 0.5 };
    const result = computeFeedback(scores, { ...defaultOpts, temperature: 1.2 });
    // 0.3 / 0.6 = 0.5 -> borderline overfitting
    expect(result.level).toBe('overfitting');
  });

  it('returns all three scores in result', () => {
    const scores: MetricScores = { memorization: 0.1, quality: 0.5, diversity: 0.7 };
    const result = computeFeedback(scores, defaultOpts);
    expect(result.memorization).toBe(0.1);
    expect(result.quality).toBe(0.5);
    expect(result.diversity).toBe(0.7);
  });
});

describe('detectOverfittingTrend', () => {
  it('returns false with < 3 entries', () => {
    expect(detectOverfittingTrend([])).toBe(false);
    expect(detectOverfittingTrend([{ memorization: 0.1, quality: 0.5, diversity: 0.8 }])).toBe(
      false,
    );
  });

  it('detects memorization up + diversity down over 3 entries', () => {
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.8 },
      { memorization: 0.3, quality: 0.5, diversity: 0.6 },
      { memorization: 0.5, quality: 0.5, diversity: 0.4 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(true);
  });

  it('handles noisy but upward memorization trend via regression', () => {
    // 0.1 -> 0.3 -> 0.29 -> 0.5: not strictly monotonic, but slope > 0.02
    // Uses last 3: [0.3, 0.29, 0.5] => slope ~ +0.1
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.9 },
      { memorization: 0.3, quality: 0.5, diversity: 0.7 },
      { memorization: 0.29, quality: 0.5, diversity: 0.55 },
      { memorization: 0.5, quality: 0.5, diversity: 0.3 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(true);
  });

  it('returns false if memorization slope is flat', () => {
    const trend = [
      { memorization: 0.3, quality: 0.5, diversity: 0.8 },
      { memorization: 0.3, quality: 0.5, diversity: 0.6 },
      { memorization: 0.3, quality: 0.5, diversity: 0.4 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(false);
  });

  it('returns false if diversity slope is flat', () => {
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.5 },
      { memorization: 0.3, quality: 0.5, diversity: 0.5 },
      { memorization: 0.5, quality: 0.5, diversity: 0.5 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(false);
  });

  it('only checks last 3 entries of longer history', () => {
    const trend = [
      { memorization: 0.8, quality: 0.5, diversity: 0.1 }, // old, ignored
      { memorization: 0.1, quality: 0.5, diversity: 0.8 },
      { memorization: 0.3, quality: 0.5, diversity: 0.6 },
      { memorization: 0.5, quality: 0.5, diversity: 0.4 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(true);
  });
});

describe('computeParamCount', () => {
  it('returns 4192 for default config (matches known value)', () => {
    expect(computeParamCount(smallModel, 27)).toBe(4192);
  });

  it('scales with n_embd and n_layer', () => {
    const small = computeParamCount(smallModel, 27);
    const big = computeParamCount(bigModel, 27);
    expect(big).toBeGreaterThan(small * 10);
  });
});

describe('computeCapacityRatio', () => {
  it('returns high ratio for small dataset', () => {
    const ratio = computeCapacityRatio(smallModel, 27, 50);
    expect(ratio).toBeCloseTo(83.84, 0);
  });

  it('returns low ratio for large dataset', () => {
    const ratio = computeCapacityRatio(smallModel, 27, 1000);
    expect(ratio).toBeCloseTo(4.192, 1);
  });

  it('returns Infinity for empty dataset', () => {
    expect(computeCapacityRatio(smallModel, 27, 0)).toBe(Infinity);
  });
});

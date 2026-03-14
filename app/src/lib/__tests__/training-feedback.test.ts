import { describe, it, expect } from 'vitest';
import { computeFeedback, detectOverfittingTrend } from '../training-feedback';
import type { MetricScores } from '../training-feedback';

const defaultOpts = {
  temperature: 0.8,
  lossEma: 1.5,
  totalSteps: 500,
  wordCount: 10,
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

  it('returns underpowered for stale large model', () => {
    const scores: MetricScores = { memorization: 0.05, quality: 0.2, diversity: 0.5 };
    const opts = { ...defaultOpts, totalSteps: 1500, lossEma: 2.5 };
    const result = computeFeedback(scores, opts);
    expect(result.level).toBe('underpowered');
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
    // 0.6 / 1.5 = 0.4 → not overfitting, could be sweet-spot
    expect(result.level).not.toBe('overfitting');
  });

  it('adjusts memorization threshold for high temperature', () => {
    // At t=1.2, raw memorization 0.3 is normalized UP to ~0.5 (hits threshold)
    const scores: MetricScores = { memorization: 0.3, quality: 0.5, diversity: 0.5 };
    const result = computeFeedback(scores, { ...defaultOpts, temperature: 1.2 });
    // 0.3 / 0.6 = 0.5 → borderline overfitting
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

  it('detects memorization↑ + diversity↓ over 3 entries', () => {
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.8 },
      { memorization: 0.3, quality: 0.5, diversity: 0.6 },
      { memorization: 0.5, quality: 0.5, diversity: 0.4 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(true);
  });

  it('returns false if memorization is not monotonically increasing', () => {
    const trend = [
      { memorization: 0.3, quality: 0.5, diversity: 0.8 },
      { memorization: 0.2, quality: 0.5, diversity: 0.6 },
      { memorization: 0.5, quality: 0.5, diversity: 0.4 },
    ];
    expect(detectOverfittingTrend(trend)).toBe(false);
  });

  it('returns false if diversity is not monotonically decreasing', () => {
    const trend = [
      { memorization: 0.1, quality: 0.5, diversity: 0.5 },
      { memorization: 0.3, quality: 0.5, diversity: 0.7 },
      { memorization: 0.5, quality: 0.5, diversity: 0.4 },
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

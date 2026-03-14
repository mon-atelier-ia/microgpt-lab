import { describe, it, expect } from 'vitest';
import { computeMemorization, computeQuality, computeDiversity } from '../training-metrics';

describe('computeMemorization', () => {
  const dataset = ['alice', 'bob', 'charlie', 'david', 'emma'];

  it('returns 0 for empty words', () => {
    expect(computeMemorization([], dataset)).toBe(0);
  });

  it('returns ~0.5 for all exact matches (0.5 weight)', () => {
    const words = ['alice', 'bob', 'charlie'];
    const score = computeMemorization(words, dataset);
    expect(score).toBeCloseTo(0.5, 1); // 0.5 × 1.0 exact
  });

  it('returns 0 for completely novel words', () => {
    const words = ['xyzzy', 'qwert', 'plugh'];
    const score = computeMemorization(words, dataset);
    expect(score).toBe(0);
  });

  it('detects fuzzy matches (1-edit distance)', () => {
    // "alicee" is 1 edit from "alice" (5 chars → dist=1/6≈0.17 ≤ 0.2)
    const words = ['alicee', 'xyzzy'];
    const score = computeMemorization(words, dataset);
    expect(score).toBeGreaterThan(0); // fuzzy match detected
    expect(score).toBeLessThan(0.5); // not full exact match
  });

  it('detects prefix matches', () => {
    // "aliceland" starts with "alice" (≥3 chars)
    const words = ['aliceland', 'xyzzy'];
    const score = computeMemorization(words, dataset);
    expect(score).toBeGreaterThan(0); // prefix match detected
  });

  it('does not double-count exact as fuzzy', () => {
    const words = ['alice'];
    const score = computeMemorization(words, dataset);
    // Only exact match (0.5), not also fuzzy (0.3)
    expect(score).toBeCloseTo(0.5, 1);
  });

  it('handles mixed matches', () => {
    const words = ['alice', 'alicee', 'bobland', 'xyzzy', 'qwert'];
    const score = computeMemorization(words, dataset);
    // 1 exact (alice), 1 fuzzy (alicee), 1 prefix (bobland), 2 novel
    expect(score).toBeGreaterThan(0.1);
    expect(score).toBeLessThan(0.5);
  });
});

describe('computeQuality', () => {
  const dataset = ['alice', 'emma', 'anne', 'pierre', 'marie'];

  it('returns 0 for empty inputs', () => {
    expect(computeQuality([], dataset)).toBe(0);
    expect(computeQuality(['test'], [])).toBe(0);
  });

  it('scores high for words with similar bigram patterns', () => {
    // Words with French-name-like bigrams
    const plausible = ['arine', 'emile', 'annic', 'pilar', 'maris'];
    const score = computeQuality(plausible, dataset);
    expect(score).toBeGreaterThan(0.4);
  });

  it('scores low for random character soup', () => {
    const garbage = ['xzqwk', 'bvnml', 'tghjp', 'yrfds', 'wqxcv'];
    const score = computeQuality(garbage, dataset);
    expect(score).toBeLessThan(0.3);
  });

  it('penalizes wrong average length', () => {
    // Dataset avg ≈ 5 chars. Single-char words = length ratio 0.2
    const tooShort = ['a', 'b', 'c', 'd', 'e'];
    const okLength = ['alice', 'emmas', 'annes', 'perry', 'maris'];
    const shortScore = computeQuality(tooShort, dataset);
    const okScore = computeQuality(okLength, dataset);
    expect(okScore).toBeGreaterThan(shortScore);
  });
});

describe('computeDiversity', () => {
  it('returns 0 for empty words', () => {
    expect(computeDiversity([], 27)).toBe(0);
  });

  it('scores high for diverse words', () => {
    const diverse = [
      'alice',
      'bob',
      'charlie',
      'david',
      'emma',
      'frank',
      'grace',
      'henry',
      'iris',
      'jack',
    ];
    const score = computeDiversity(diverse, 27);
    expect(score).toBeGreaterThan(0.6);
  });

  it('scores low for repeated words (mode collapse)', () => {
    const collapsed = ['alice', 'alice', 'alice', 'alice', 'alice'];
    const score = computeDiversity(collapsed, 27);
    expect(score).toBeLessThan(0.5);
  });

  it('scores low for single-character words', () => {
    // Single chars have 0 bigrams → distinct-2 = 0
    const singleChars = ['a', 'b', 'c', 'd', 'e'];
    const score = computeDiversity(singleChars, 27);
    expect(score).toBeLessThan(0.5);
  });

  it('handles vocab size normalization', () => {
    const words = ['abc', 'def', 'ghi'];
    const smallVocab = computeDiversity(words, 10);
    const largeVocab = computeDiversity(words, 100);
    // Same words but higher vocab → lower distinct-1
    expect(smallVocab).toBeGreaterThan(largeVocab);
  });
});

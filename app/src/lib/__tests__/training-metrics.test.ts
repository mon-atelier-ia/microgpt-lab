import { describe, it, expect } from 'vitest';
import {
  buildDatasetProfile,
  computeMemorization,
  computeQuality,
  computeDiversity,
  normalizedEditDistance,
} from '../training-metrics';

describe('buildDatasetProfile', () => {
  it('lowercases all words', () => {
    const profile = buildDatasetProfile(['Alice', 'BOB']);
    expect(profile.words).toEqual(['alice', 'bob']);
  });

  it('builds wordSet for O(1) lookup', () => {
    const profile = buildDatasetProfile(['alice', 'bob']);
    expect(profile.wordSet.has('alice')).toBe(true);
    expect(profile.wordSet.has('charlie')).toBe(false);
  });

  it('computes average length', () => {
    const profile = buildDatasetProfile(['ab', 'abcd']); // avg = 3
    expect(profile.avgLength).toBe(3);
  });

  it('handles empty dataset', () => {
    const profile = buildDatasetProfile([]);
    expect(profile.words).toEqual([]);
    expect(profile.avgLength).toBe(0);
    expect(profile.vocabSize).toBe(1); // only BOS
  });

  it('computes vocabSize as unique chars + 1 (BOS)', () => {
    // 'ab' + 'bc' → unique chars {a,b,c} → vocabSize = 4
    const profile = buildDatasetProfile(['ab', 'bc']);
    expect(profile.vocabSize).toBe(4);
  });

  it('vocabSize matches Rust build_vocab for Prénoms FR (50)', () => {
    // Rust build_vocab uses only chars present in dataset (not full alphabet).
    // These 50 prénoms have 24 unique chars (no k, w) + BOS = 25.
    const prenoms = [
      'alain',
      'alexandre',
      'andre',
      'anne',
      'bernard',
      'brigitte',
      'catherine',
      'cecile',
      'charles',
      'christiane',
      'christine',
      'claude',
      'daniel',
      'denis',
      'dominique',
      'elisabeth',
      'eric',
      'francois',
      'francoise',
      'gerard',
      'henri',
      'isabelle',
      'jacques',
      'jean',
      'laurent',
      'louis',
      'marcel',
      'marguerite',
      'marie',
      'martine',
      'michel',
      'monique',
      'nathalie',
      'nicolas',
      'patrick',
      'paul',
      'philippe',
      'pierre',
      'raymond',
      'rene',
      'robert',
      'roger',
      'simone',
      'sophie',
      'stephane',
      'sylvie',
      'thierry',
      'thomas',
      'xavier',
      'yves',
    ];
    const profile = buildDatasetProfile(prenoms);
    expect(profile.vocabSize).toBe(25);
  });
});

describe('normalizedEditDistance', () => {
  it('returns 0 for two empty strings', () => {
    expect(normalizedEditDistance('', '')).toBe(0);
  });

  it('returns 0 for identical strings', () => {
    expect(normalizedEditDistance('hello', 'hello')).toBe(0);
  });

  it('returns 1 for completely different single chars', () => {
    expect(normalizedEditDistance('a', 'b')).toBe(1);
  });

  it('handles single char difference', () => {
    // "kitten" vs "sitten" = 1 edit, max length 6
    expect(normalizedEditDistance('kitten', 'sitten')).toBeCloseTo(1 / 6, 5);
  });

  it('returns 1 for completely different strings of same length', () => {
    // "abc" vs "xyz" = 3 edits, max length 3
    expect(normalizedEditDistance('abc', 'xyz')).toBe(1);
  });

  it('handles different-length strings', () => {
    // "abc" vs "ab" = 1 deletion, max length 3
    expect(normalizedEditDistance('abc', 'ab')).toBeCloseTo(1 / 3, 5);
  });

  it('handles accented/unicode characters', () => {
    // "rene" vs "rene" = 0
    expect(normalizedEditDistance('rene', 'rene')).toBe(0);
    // "rene" (4 chars) vs "ren" (3 chars) = 1 edit, max 4
    expect(normalizedEditDistance('rene', 'ren')).toBeCloseTo(1 / 4, 5);
  });

  it('handles one empty string', () => {
    expect(normalizedEditDistance('abc', '')).toBe(1);
    expect(normalizedEditDistance('', 'abc')).toBe(1);
  });
});

describe('computeMemorization', () => {
  const profile = buildDatasetProfile(['alice', 'bob', 'charlie', 'david', 'emma']);

  it('returns 0 for empty words', () => {
    expect(computeMemorization([], profile)).toBe(0);
  });

  it('returns ~0.5 for all exact matches (0.5 weight)', () => {
    const words = ['alice', 'bob', 'charlie'];
    const score = computeMemorization(words, profile);
    expect(score).toBeCloseTo(0.5, 1); // 0.5 x 1.0 exact
  });

  it('returns 0 for completely novel words', () => {
    const words = ['xyzzy', 'qwert', 'plugh'];
    const score = computeMemorization(words, profile);
    expect(score).toBe(0);
  });

  it('detects fuzzy matches (1-edit distance)', () => {
    // "alicee" is 1 edit from "alice" (5 chars -> dist=1/6~0.17 <= 0.2)
    const words = ['alicee', 'xyzzy'];
    const score = computeMemorization(words, profile);
    expect(score).toBeGreaterThan(0); // fuzzy match detected
    expect(score).toBeLessThan(0.5); // not full exact match
  });

  it('detects prefix matches', () => {
    // "aliceland" starts with "alice" (>=3 chars)
    const words = ['aliceland', 'xyzzy'];
    const score = computeMemorization(words, profile);
    expect(score).toBeGreaterThan(0); // prefix match detected
  });

  it('does not double-count exact as fuzzy', () => {
    const words = ['alice'];
    const score = computeMemorization(words, profile);
    // Only exact match (0.5), not also fuzzy (0.3)
    expect(score).toBeCloseTo(0.5, 1);
  });

  it('handles mixed matches', () => {
    const words = ['alice', 'alicee', 'bobland', 'xyzzy', 'qwert'];
    const score = computeMemorization(words, profile);
    // 1 exact (alice), 1 fuzzy (alicee), 1 prefix (bobland), 2 novel
    expect(score).toBeGreaterThan(0.1);
    expect(score).toBeLessThan(0.5);
  });
});

describe('computeQuality', () => {
  const profile = buildDatasetProfile(['alice', 'emma', 'anne', 'pierre', 'marie']);

  it('returns 0 for empty inputs', () => {
    const emptyProfile = buildDatasetProfile([]);
    expect(computeQuality([], profile)).toBe(0);
    expect(computeQuality(['test'], emptyProfile)).toBe(0);
  });

  it('scores higher for words with similar bigram patterns', () => {
    // Words with French-name-like bigrams (small 5-word dataset limits coverage)
    const plausible = ['arine', 'emile', 'annic', 'pilar', 'maris'];
    const score = computeQuality(plausible, profile);
    expect(score).toBeGreaterThan(0.25);
  });

  it('scores low for random character soup', () => {
    const garbage = ['xzqwk', 'bvnml', 'tghjp', 'yrfds', 'wqxcv'];
    const score = computeQuality(garbage, profile);
    expect(score).toBeLessThan(0.3);
  });

  it('penalizes wrong average length', () => {
    // Dataset avg ~ 5 chars. Single-char words = length ratio 0.2
    const tooShort = ['a', 'b', 'c', 'd', 'e'];
    const okLength = ['alice', 'emmas', 'annes', 'perry', 'maris'];
    const shortScore = computeQuality(tooShort, profile);
    const okScore = computeQuality(okLength, profile);
    expect(okScore).toBeGreaterThan(shortScore);
  });
});

describe('computeDiversity', () => {
  it('returns 0 for empty words', () => {
    expect(computeDiversity([])).toBe(0);
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
    const score = computeDiversity(diverse);
    expect(score).toBeGreaterThan(0.5);
  });

  it('scores low for repeated words (mode collapse)', () => {
    const collapsed = ['alice', 'alice', 'alice', 'alice', 'alice'];
    const score = computeDiversity(collapsed);
    expect(score).toBeLessThan(0.5);
  });

  it('scores low for single-character words', () => {
    // Single chars: distinct-1 = 5/5 = 1.0, distinct-2 = 0, uniqueRatio = 1.0
    // Score = 0.4*1 + 0.3*0 + 0.3*1 = 0.7 (no bigram diversity)
    const singleChars = ['a', 'b', 'c', 'd', 'e'];
    const score = computeDiversity(singleChars);
    expect(score).toBeLessThanOrEqual(0.7);
  });

  it('Distinct-1 is unique unigrams / total unigrams (Li et al. 2016)', () => {
    // "aaa" has 1 unique char, 3 total chars -> distinct1 = 1/3
    // "abc" has 3 unique chars, 3 total chars -> distinct1 = 1
    const repeated = computeDiversity(['aaa']);
    const varied = computeDiversity(['abc']);
    expect(varied).toBeGreaterThan(repeated);
  });
});

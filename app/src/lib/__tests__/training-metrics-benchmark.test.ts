import { describe, it, expect } from 'vitest';
import { buildDatasetProfile, computeQuality, computeMemorization } from '../training-metrics';

/**
 * Benchmark validation tests using REAL data from Prenoms FR (50) training runs.
 *
 * These tests validate that our metric weights produce sensible scores that
 * separate garbage output (early training) from plausible output (later training).
 *
 * Dataset: Prenoms FR — 50 most common French first names.
 * Source: actual microgpt-lab training runs at default hyperparameters.
 */

const PRENOMS_FR_50 = [
  'alain',
  'alexandre',
  'andre',
  'anne',
  'catherine',
  'charles',
  'christian',
  'claude',
  'daniel',
  'david',
  'dominique',
  'francois',
  'francoise',
  'georges',
  'gerard',
  'isabelle',
  'jacques',
  'jean',
  'jeanne',
  'joseph',
  'laurent',
  'louis',
  'marie',
  'martine',
  'michel',
  'monique',
  'nicole',
  'patrick',
  'paul',
  'philippe',
  'pierre',
  'rene',
  'robert',
  'stephane',
  'suzanne',
  'sylvie',
  'thierry',
  'thomas',
  'alain',
  'andre',
  'anne',
  'catherine',
  'claude',
  'daniel',
  'david',
  'dominique',
  'francois',
  'jean',
  'joseph',
  'marie',
];

const profile = buildDatasetProfile(PRENOMS_FR_50);

describe('Prenoms FR benchmark — quality separation', () => {
  // Step 200: model outputs garbage (random character sequences)
  const garbageWords = ['arccstics', 'n', 'geand', 'maauelice', 'kanid'];
  // Step 1000: model outputs plausible names
  const goodWords = ['pare', 'rene', 're', 'jedereinnne', 'david'];

  it('garbage words (step 200) score quality < 0.3', () => {
    const score = computeQuality(garbageWords, profile);
    expect(score).toBeLessThan(0.3);
  });

  it('plausible words (step 1000) score quality > 0.4', () => {
    const score = computeQuality(goodWords, profile);
    expect(score).toBeGreaterThan(0.4);
  });

  it('quality improves from garbage to plausible', () => {
    const garbageScore = computeQuality(garbageWords, profile);
    const goodScore = computeQuality(goodWords, profile);
    expect(goodScore).toBeGreaterThan(garbageScore);
  });
});

describe('Prenoms FR benchmark — memorization detection', () => {
  it('exact dataset words produce high memorization', () => {
    const exactCopies = ['jean', 'marie', 'pierre', 'paul', 'anne'];
    const score = computeMemorization(exactCopies, profile);
    expect(score).toBeCloseTo(0.5, 1); // 0.5 weight for exact
  });

  it('plausible but novel words produce low memorization', () => {
    const novel = ['pare', 'jedereinnne', 'maron', 'claud', 'philipe'];
    const score = computeMemorization(novel, profile);
    expect(score).toBeLessThan(0.3);
  });

  it('mix of exact and novel produces mid-range memorization', () => {
    // "rene" and "david" are exact; "pare", "jedereinnne", "re" are not
    const mixed = ['pare', 'rene', 're', 'jedereinnne', 'david'];
    const score = computeMemorization(mixed, profile);
    expect(score).toBeGreaterThan(0.1);
    expect(score).toBeLessThan(0.4);
  });
});

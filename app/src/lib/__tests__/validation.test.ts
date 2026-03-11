import { describe, it, expect } from 'vitest';
import { validHeadCounts } from '../validation';

describe('validHeadCounts', () => {
  it('returns divisors of n_embd from options', () => {
    expect(validHeadCounts(16, [1, 2, 4, 8, 16])).toEqual([1, 2, 4, 8, 16]);
  });

  it('filters out non-divisors', () => {
    expect(validHeadCounts(8, [1, 2, 3, 4, 8])).toEqual([1, 2, 4, 8]);
  });

  it('returns empty for no valid options', () => {
    expect(validHeadCounts(7, [2, 4, 8])).toEqual([]);
  });

  it('handles n_embd=1', () => {
    expect(validHeadCounts(1, [1, 2, 4])).toEqual([1]);
  });
});

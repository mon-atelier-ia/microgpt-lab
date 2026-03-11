import { describe, it, expect } from 'vitest';
import { posToLr, lrToPos, formatLr } from '../params-utils';

describe('posToLr / lrToPos roundtrip', () => {
  it('posToLr(0) ≈ 0.001', () => {
    expect(posToLr(0)).toBeCloseTo(0.001, 6);
  });

  it('posToLr(1) ≈ 0.5', () => {
    expect(posToLr(1)).toBeCloseTo(0.5, 6);
  });

  it('roundtrip is identity for pos=0.25', () => {
    expect(lrToPos(posToLr(0.25))).toBeCloseTo(0.25, 10);
  });

  it('roundtrip is identity for pos=0.5', () => {
    expect(lrToPos(posToLr(0.5))).toBeCloseTo(0.5, 10);
  });

  it('roundtrip is identity for pos=0.75', () => {
    expect(lrToPos(posToLr(0.75))).toBeCloseTo(0.75, 10);
  });

  it('roundtrip is identity for pos=1', () => {
    expect(lrToPos(posToLr(1))).toBeCloseTo(1, 10);
  });
});

describe('formatLr', () => {
  it('uses exponential notation for lr < 0.01', () => {
    expect(formatLr(0.001)).toBe('1.0e-3');
  });

  it('uses exponential notation for lr = 0.005', () => {
    expect(formatLr(0.005)).toBe('5.0e-3');
  });

  it('uses fixed notation for lr >= 0.01', () => {
    expect(formatLr(0.01)).toBe('0.010');
  });

  it('uses fixed notation for lr = 0.5', () => {
    expect(formatLr(0.5)).toBe('0.500');
  });

  it('uses fixed notation for lr = 0.123', () => {
    expect(formatLr(0.123)).toBe('0.123');
  });
});

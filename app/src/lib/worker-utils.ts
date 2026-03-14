export function isPositiveFinite(v: number): boolean {
  return Number.isFinite(v) && v > 0;
}

export function hasKey<K extends string>(o: object, k: K): o is Record<K, unknown> {
  return k in o;
}

export function isStepResult(
  v: unknown,
): v is { step: number; loss: number; word: string; lr: number } {
  if (typeof v !== 'object' || v === null) return false;
  return (
    hasKey(v, 'step') &&
    typeof v.step === 'number' &&
    hasKey(v, 'loss') &&
    typeof v.loss === 'number' &&
    hasKey(v, 'word') &&
    typeof v.word === 'string' &&
    hasKey(v, 'lr') &&
    typeof v.lr === 'number'
  );
}

export function validateConfig(c: {
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
}): string | null {
  if (
    !isPositiveFinite(c.n_embd) ||
    !isPositiveFinite(c.n_head) ||
    !isPositiveFinite(c.n_layer) ||
    !isPositiveFinite(c.block_size)
  ) {
    return `Invalid config: n_embd=${c.n_embd} n_head=${c.n_head} n_layer=${c.n_layer} block_size=${c.block_size}`;
  }
  if (c.n_embd % c.n_head !== 0) {
    return `n_embd (${c.n_embd}) must be divisible by n_head (${c.n_head})`;
  }
  return null;
}

export function sampleFromProbs(probs: Float64Array): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

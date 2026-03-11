import type { ModelParams } from './types';

export function validHeadCounts(n_embd: number, options: number[]): number[] {
  return options.filter((h) => n_embd % h === 0);
}

export function isArchChange(a: ModelParams, b: ModelParams): boolean {
  return (
    a.datasetId !== b.datasetId ||
    a.n_embd !== b.n_embd ||
    a.n_head !== b.n_head ||
    a.n_layer !== b.n_layer ||
    a.block_size !== b.block_size
  );
}

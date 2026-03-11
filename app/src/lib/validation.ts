export function validHeadCounts(n_embd: number, options: number[]): number[] {
  return options.filter((h) => n_embd % h === 0);
}

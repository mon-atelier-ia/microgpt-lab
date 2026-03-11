const LR_LOG_MIN = Math.log(0.001);
const LR_LOG_MAX = Math.log(0.5);

export function posToLr(pos: number): number {
  return Math.exp(LR_LOG_MIN + pos * (LR_LOG_MAX - LR_LOG_MIN));
}

export function lrToPos(lr: number): number {
  return (Math.log(lr) - LR_LOG_MIN) / (LR_LOG_MAX - LR_LOG_MIN);
}

export function formatLr(lr: number): string {
  return lr < 0.01 ? lr.toExponential(1) : lr.toFixed(3);
}

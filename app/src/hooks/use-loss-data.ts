import { useMemo } from 'react';
import type { ColorVar, StepResult } from '../lib/types';
import type { ChartData } from 'chart.js';
import { resolveVar } from '../lib/utils';

function computeEma(values: number[], alpha: number): number[] {
  if (values.length === 0) return [];
  const ema: number[] = [values[0]];
  for (let i = 1; i < values.length; i++) {
    ema.push(alpha * values[i] + (1 - alpha) * ema[i - 1]);
  }
  return ema;
}

const EMA_ALPHA = 0.1;

export function useLossData(
  steps: StepResult[],
  colorVar: ColorVar,
): ChartData<'line', number[], number> {
  return useMemo(() => {
    const rawColor = resolveVar(`--model-${colorVar}`);
    const emaColor = resolveVar(`--model-${colorVar}-accent`);
    const losses = steps.map((s) => s.loss);
    const ema = computeEma(losses, EMA_ALPHA);

    return {
      labels: steps.map((s) => s.step),
      datasets: [
        {
          label: 'Loss brute',
          data: losses,
          borderColor: rawColor,
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.1,
          borderDash: [],
          order: 1,
        },
        {
          label: 'EMA',
          data: ema,
          borderColor: emaColor,
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3,
          borderDash: [],
          order: 0,
        },
      ],
    };
  }, [steps, colorVar]);
}

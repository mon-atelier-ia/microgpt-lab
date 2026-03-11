import { useMemo } from 'react';
import type { StepResult } from '../lib/types';
import type { ChartData } from 'chart.js';

function resolveVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

export function useLossData(
  steps: StepResult[],
  colorVar: 'a' | 'b',
): ChartData<'line', number[], number> {
  return useMemo(
    () => ({
      labels: steps.map((s) => s.step),
      datasets: [
        {
          label: 'Train Loss',
          data: steps.map((s) => s.loss),
          borderColor: resolveVar(`--model-${colorVar}`),
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    }),
    [steps, colorVar],
  );
}

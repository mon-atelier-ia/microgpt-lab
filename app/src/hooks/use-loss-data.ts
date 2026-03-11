import { useMemo } from 'react';
import type { ColorVar, StepResult } from '../lib/types';
import type { ChartData } from 'chart.js';
import { resolveVar } from '../lib/utils';

export function useLossData(
  steps: StepResult[],
  colorVar: ColorVar,
): ChartData<'line', number[], number> {
  const borderColor = resolveVar(`--model-${colorVar}`);

  return useMemo(
    () => ({
      labels: steps.map((s) => s.step),
      datasets: [
        {
          label: 'Train Loss',
          data: steps.map((s) => s.loss),
          borderColor,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    }),
    [steps, borderColor],
  );
}

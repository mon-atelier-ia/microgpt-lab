import { useMemo } from 'react';
import type { StepResult } from '../lib/types';
import type { ChartData } from 'chart.js';
import { resolveVar } from '../lib/utils';

export function useLossData(
  steps: StepResult[],
  colorVar: 'a' | 'b',
): ChartData<'line', number[], number> {
  const lossValues = useMemo(() => steps.map((s) => s.loss), [steps]);
  const labels = useMemo(() => steps.map((s) => s.step), [steps]);

  // Resolve color every render so theme changes are picked up immediately.
  const borderColor = resolveVar(`--model-${colorVar}`);

  return {
    labels,
    datasets: [
      {
        label: 'Train Loss',
        data: lossValues,
        borderColor,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
      },
    ],
  };
}

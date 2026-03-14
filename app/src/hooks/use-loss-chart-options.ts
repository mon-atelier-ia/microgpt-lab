import { useMemo, useState } from 'react';
import type { ChartOptions, TooltipItem } from 'chart.js';
import { resolveVar } from '../lib/model-colors';

function buildChartOptions(textMuted: string, gridColor: string): ChartOptions<'line'> {
  return {
    animation: false as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: TooltipItem<'line'>) =>
            `loss: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(4) : '—'}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: textMuted, maxTicksLimit: 6 },
        grid: { color: gridColor },
      },
      y: {
        ticks: { color: textMuted },
        grid: { color: gridColor },
      },
    },
  };
}

export type LossChartOptionsResult = {
  chartOptions: ChartOptions<'line'>;
  hasAnimated: boolean;
  setHasAnimated: (v: boolean) => void;
};

export function useLossChartOptions(hasData: boolean): LossChartOptionsResult {
  const [hasAnimated, setHasAnimated] = useState(false);
  // Resolve CSS vars outside useMemo for theme reactivity
  const textMuted = resolveVar('--text-muted');
  const gridColor = resolveVar('--border-subtle');
  const chartOptions = useMemo(() => {
    const opts = buildChartOptions(textMuted, gridColor);
    // Animate draw on first data appearance only, then disable for real-time perf
    if (hasData && !hasAnimated) {
      opts.animation = {
        duration: 800,
        easing: 'easeOutQuart',
        onComplete: () => setHasAnimated(true),
      };
    }
    return opts;
  }, [hasData, hasAnimated, textMuted, gridColor]);

  return { chartOptions, hasAnimated, setHasAnimated };
}

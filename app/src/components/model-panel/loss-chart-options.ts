import type { ChartOptions, TooltipItem } from 'chart.js';
import { resolveVar } from '../../lib/utils';

const GRID_COLOR = 'rgba(255,255,255,0.05)';

export function getChartOptions(): ChartOptions<'line'> {
  const textMuted = resolveVar('--text-muted');
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
        grid: { color: GRID_COLOR },
      },
      y: {
        ticks: { color: textMuted },
        grid: { color: GRID_COLOR },
      },
    },
  };
}

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type TooltipItem,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import type { StepResult } from '../../lib/types';
import { useLossData } from '../../hooks/use-loss-data';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

/* resolved at runtime for chart.js which needs actual color strings */

function resolveVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function getChartOptions() {
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
        grid: { color: 'rgba(255,255,255,0.05)' },
      },
      y: {
        ticks: { color: textMuted },
        grid: { color: 'rgba(255,255,255,0.05)' },
      },
    },
  };
}

type LossPanelProps = {
  steps: StepResult[];
  colorVar: 'a' | 'b';
};

export function LossPanel({ steps, colorVar }: LossPanelProps) {
  const data = useLossData(steps, colorVar);
  const lastStep = steps[steps.length - 1];
  const chartOptions = getChartOptions();

  return (
    <div aria-label="Loss curve" className="flex flex-col gap-2 rounded-lg bg-surface-1 p-4">
      <div role="status" className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-secondary">Loss curve</span>
        <div className="flex items-center gap-3">
          {lastStep && (
            <>
              <span className="font-mono text-xs text-text-muted">step {lastStep.step}</span>
              <span
                className="font-mono text-xs font-semibold"
                style={{ color: `var(--model-${colorVar})` }}
              >
                {lastStep.loss.toFixed(4)}
              </span>
            </>
          )}
          {!lastStep && <span className="text-xs text-text-muted">— pas encore entraîné</span>}
        </div>
      </div>
      <div className="relative h-40">
        {steps.length > 0 ? (
          <Line data={data} options={chartOptions} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-text-muted">
            Entraînez le modèle pour voir la courbe de perte
          </div>
        )}
      </div>
    </div>
  );
}

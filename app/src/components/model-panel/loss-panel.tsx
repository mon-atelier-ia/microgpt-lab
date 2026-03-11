import { useMemo } from 'react';
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
import type { ColorVar, StepResult } from '../../lib/types';
import { useLossData } from '../../hooks/use-loss-data';
import { modelColor, resolveVar } from '../../lib/utils';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const GRID_COLOR = 'rgba(255,255,255,0.05)';

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
        grid: { color: GRID_COLOR },
      },
      y: {
        ticks: { color: textMuted },
        grid: { color: GRID_COLOR },
      },
    },
  };
}

type LossPanelProps = {
  steps: StepResult[];
  colorVar: ColorVar;
};

export function LossPanel({ steps, colorVar }: LossPanelProps) {
  const data = useLossData(steps, colorVar);
  const lastStep = steps[steps.length - 1];
  const chartOptions = useMemo(() => getChartOptions(), []);

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
                style={{ color: modelColor(colorVar) }}
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

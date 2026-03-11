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

const TEXT_MUTED = 'var(--text-muted)';
const TEXT_SECONDARY = 'var(--text-secondary)';

const CHART_OPTIONS = {
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
      ticks: { color: TEXT_MUTED, maxTicksLimit: 6 },
      grid: { color: 'rgba(255,255,255,0.05)' },
    },
    y: {
      ticks: { color: TEXT_MUTED },
      grid: { color: 'rgba(255,255,255,0.05)' },
    },
  },
};

type LossPanelProps = {
  steps: StepResult[];
  colorVar: 'a' | 'b';
};

export function LossPanel({ steps, colorVar }: LossPanelProps) {
  const data = useLossData(steps, colorVar);
  const lastStep = steps[steps.length - 1];

  return (
    <div
      className="flex flex-col gap-2 p-4"
      style={{ background: 'var(--surface-1)', borderRadius: '0.5rem' }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium" style={{ color: TEXT_SECONDARY }}>
          Loss curve
        </span>
        <div className="flex items-center gap-3">
          {lastStep && (
            <>
              <span className="font-mono text-xs" style={{ color: TEXT_MUTED }}>
                step {lastStep.step}
              </span>
              <span
                className="font-mono text-xs font-semibold"
                style={{ color: `var(--model-${colorVar})` }}
              >
                {lastStep.loss.toFixed(4)}
              </span>
            </>
          )}
          {!lastStep && (
            <span className="text-xs" style={{ color: TEXT_MUTED }}>
              — pas encore entraîné
            </span>
          )}
        </div>
      </div>
      <div className="relative h-40">
        {steps.length > 0 ? (
          <Line data={data} options={CHART_OPTIONS} />
        ) : (
          <div
            className="flex h-full items-center justify-center text-xs"
            style={{ color: TEXT_MUTED }}
          >
            Entraînez le modèle pour voir la courbe de perte
          </div>
        )}
      </div>
    </div>
  );
}

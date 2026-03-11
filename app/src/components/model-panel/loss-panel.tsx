import { lazy, Suspense, useMemo, useState } from 'react';
import type { ChartOptions, TooltipItem } from 'chart.js';
import type { ColorVar, StepResult } from '../../lib/types';
import { useLossData } from '../../hooks/use-loss-data';
import { cn, modelAccent, resolveVar } from '../../lib/utils';

const LossChart = lazy(() => import('./loss-chart').then((m) => ({ default: m.LossChart })));

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

type LossPanelProps = {
  steps: StepResult[];
  colorVar: ColorVar;
  glowClass?: string;
};

export function LossPanel({ steps, colorVar, glowClass }: LossPanelProps) {
  const data = useLossData(steps, colorVar);
  const lastStep = steps[steps.length - 1];
  const hasData = steps.length > 0;
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

  return (
    <div
      aria-label="Courbe de loss"
      className={cn('flex flex-col gap-2 rounded-lg p-4 panel-surface', glowClass)}
    >
      <div role="status" className="flex items-center justify-between">
        <span className="instrument-header text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Courbe de loss
        </span>
        <div className="flex items-center gap-3">
          {lastStep && (
            <>
              <span className="font-mono text-xs text-text-muted">step {lastStep.step}</span>
              <span
                className="font-mono text-xs font-bold"
                style={{ color: modelAccent(colorVar) }}
              >
                {lastStep.loss.toFixed(4)}
              </span>
            </>
          )}
          {!lastStep && (
            <span className="flex items-center gap-1.5 text-xs text-text-muted">
              <span className="status-dot bg-text-muted" />
              en attente
            </span>
          )}
        </div>
      </div>
      <div className="relative h-40">
        {hasData ? (
          <Suspense
            fallback={
              <div className="flex h-full items-center justify-center text-xs text-text-muted">
                Chargement du graphique…
              </div>
            }
          >
            <LossChart data={data} chartOptions={chartOptions} />
          </Suspense>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-text-muted">
            Entraînez le modèle pour voir la courbe
          </div>
        )}
      </div>
    </div>
  );
}

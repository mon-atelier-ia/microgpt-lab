import { lazy, Suspense } from 'react';
import type { ColorVar, StepResult } from '../../lib/types';
import { useLossData } from '../../hooks/use-loss-data';
import { useLossChartOptions } from '../../hooks/use-loss-chart-options';
import { cn } from '../../lib/utils';
import { modelAccent } from '../../lib/model-colors';

const LossChart = lazy(() => import('./loss-chart').then((m) => ({ default: m.LossChart })));

type LossPanelProps = {
  steps: StepResult[];
  colorVar: ColorVar;
  glowClass?: string;
};

export function LossPanel({ steps, colorVar, glowClass }: LossPanelProps) {
  const { chartData, lastEma } = useLossData(steps, colorVar);
  const lastStep = steps[steps.length - 1];
  const hasData = steps.length > 0;
  const { chartOptions } = useLossChartOptions(hasData);

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
                title="EMA (α=0.1)"
              >
                {lastEma !== null ? lastEma.toFixed(4) : lastStep.loss.toFixed(4)}
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
            <LossChart data={chartData} chartOptions={chartOptions} />
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

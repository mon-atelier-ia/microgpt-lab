import type { ColorVar } from '../../lib/types';
import { cn, modelColor, modelMuted } from '../../lib/utils';

type InferencePanelProps = {
  words: string[];
  colorVar: ColorVar;
  temperature?: number;
  glowClass?: string;
};

export function InferencePanel({ words, colorVar, temperature, glowClass }: InferencePanelProps) {
  const primary = modelColor(colorVar);
  const muted = modelMuted(colorVar);

  return (
    <div
      aria-label="Generated words"
      className={cn('flex flex-col gap-3 rounded-lg p-4 panel-surface', glowClass)}
    >
      <span className="instrument-header text-xs font-semibold uppercase tracking-wider text-text-secondary">
        Mots générés
      </span>

      {words.length === 0 ? (
        <div className="flex flex-1 items-center justify-center py-8 text-xs text-text-muted">
          Générez des mots après l&apos;entraînement
        </div>
      ) : (
        <div
          role="list"
          className="grid gap-2"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(5rem, 1fr))' }}
        >
          {words.map((word, i) => (
            <div
              key={i}
              role="listitem"
              className="word-item rounded-md px-2 py-1.5 text-center font-mono text-sm font-semibold transition-all duration-200 hover:scale-105 hover:brightness-125"
              style={{
                color: primary,
                backgroundColor: muted,
                borderLeft: `2px solid ${primary}`,
                animationDelay: `${i * 60}ms`,
              }}
            >
              {word}
            </div>
          ))}
        </div>
      )}

      <div role="status" className="flex justify-between font-mono text-xs text-text-muted">
        {temperature !== undefined && <span>t={temperature.toFixed(2)}</span>}
        <span className="ml-auto">
          {words.length} mot{words.length !== 1 ? 's' : ''}
        </span>
      </div>
    </div>
  );
}

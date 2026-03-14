import type { ColorVar } from '../../lib/types';
import { cn } from '../../lib/utils';
import { modelColor, modelMuted } from '../../lib/model-colors';

type InferencePanelProps = {
  words: string[];
  colorVar: ColorVar;
  temperature?: number;
  glowClass?: string;
};

export function InferencePanel({ words, colorVar, temperature, glowClass }: InferencePanelProps) {
  const primary = modelColor(colorVar);
  const muted = modelMuted(colorVar);
  // Unique key per word set to force grid remount → re-trigger CSS stagger animations
  const genKey = words.join('\0');

  return (
    <div
      role="region"
      aria-label="Mots générés"
      className={cn('flex flex-col gap-3 rounded-lg p-4 panel-surface', glowClass)}
    >
      <span className="instrument-header text-xs font-semibold uppercase tracking-wider text-text-secondary">
        Mots générés
      </span>

      {words.length === 0 ? (
        <div className="py-8 text-center text-xs text-text-muted">
          Générez des mots après l&apos;entraînement
        </div>
      ) : (
        <div key={genKey} role="list" className="flex flex-wrap gap-2">
          {words.map((word, i) => (
            <div
              key={`${i}-${word}`}
              role="listitem"
              className="word-item rounded-full px-3 py-1 font-mono text-sm font-semibold"
              style={{
                color: primary,
                backgroundColor: muted,
                border: `1px solid ${primary}`,
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

type InferencePanelProps = {
  words: string[];
  colorVar: 'a' | 'b';
  temperature?: number;
};

export function InferencePanel({ words, colorVar, temperature }: InferencePanelProps) {
  const accentColor = `var(--model-${colorVar})`;

  return (
    <div
      aria-label="Generated words"
      className="flex flex-col gap-3 p-4"
      style={{ background: 'var(--surface-1)', borderRadius: '0.5rem' }}
    >
      <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
        Mots générés
      </span>

      {words.length === 0 ? (
        <div
          className="flex flex-1 items-center justify-center py-8 text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
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
              className="rounded px-2 py-1 text-center text-sm font-medium"
              style={{ background: 'var(--surface-2)', color: accentColor }}
            >
              {word}
            </div>
          ))}
        </div>
      )}

      <div
        role="status"
        className="flex justify-between text-xs"
        style={{ color: 'var(--text-muted)' }}
      >
        {temperature !== undefined && <span>t={temperature.toFixed(2)}</span>}
        <span className="ml-auto">
          {words.length} mot{words.length !== 1 ? 's' : ''}
        </span>
      </div>
    </div>
  );
}

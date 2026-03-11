type InferencePanelProps = {
  words: string[];
  colorVar: 'a' | 'b';
};

export function InferencePanel({ words, colorVar }: InferencePanelProps) {
  const accentColor = `var(--model-${colorVar})`;

  return (
    <div
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
          className="grid gap-2"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(5rem, 1fr))' }}
        >
          {words.map((word, i) => (
            <div
              key={`${word}-${i}`}
              className="rounded px-2 py-1 text-center text-sm font-medium"
              style={{ background: 'var(--surface-2)', color: accentColor }}
            >
              {word}
            </div>
          ))}
        </div>
      )}

      <div className="text-right text-xs" style={{ color: 'var(--text-muted)' }}>
        {words.length} mot{words.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
}

import { Button } from './ui/button';

export type Mode = 'solo' | 'compare';

export type TopBarProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
};

export function TopBar({ mode, onModeChange }: TopBarProps) {
  return (
    <header
      style={{
        background: 'var(--surface-1)',
        color: 'var(--text-primary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.75rem 1rem',
        borderBottom: '1px solid oklch(0.25 0.01 260)',
      }}
    >
      <span style={{ fontWeight: 700, fontSize: '1.1rem', letterSpacing: '-0.01em' }}>
        microgpt-lab
      </span>
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <Button
          size="sm"
          variant={mode === 'solo' ? 'default' : 'ghost'}
          onClick={() => onModeChange('solo')}
        >
          Solo
        </Button>
        <Button
          size="sm"
          variant={mode === 'compare' ? 'default' : 'ghost'}
          onClick={() => onModeChange('compare')}
        >
          Compare
        </Button>
      </div>
    </header>
  );
}

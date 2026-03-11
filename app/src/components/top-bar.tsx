import { Button } from './ui/button';
import type { Mode } from '../lib/types';

export type TopBarProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
};

export function TopBar({ mode, onModeChange }: TopBarProps) {
  return (
    <header
      aria-label="microgpt-lab"
      className="flex items-center justify-between bg-surface-1 text-text-primary px-4 py-3"
      style={{ borderBottom: '1px solid var(--border-subtle)' }}
    >
      <span className="text-lg font-bold" style={{ letterSpacing: '-0.01em' }}>
        microgpt-lab
      </span>
      <div role="tablist" className="flex gap-2">
        <Button
          size="sm"
          role="tab"
          id="tab-solo"
          aria-selected={mode === 'solo'}
          aria-controls="main-tabpanel"
          variant={mode === 'solo' ? 'default' : 'ghost'}
          onClick={() => onModeChange('solo')}
        >
          Solo
        </Button>
        <Button
          size="sm"
          role="tab"
          id="tab-compare"
          aria-selected={mode === 'compare'}
          aria-controls="main-tabpanel"
          variant={mode === 'compare' ? 'default' : 'ghost'}
          onClick={() => onModeChange('compare')}
        >
          Compare
        </Button>
      </div>
    </header>
  );
}

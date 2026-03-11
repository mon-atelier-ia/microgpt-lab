import type { KeyboardEvent } from 'react';
import { useCallback, useRef } from 'react';
import { cn } from '../lib/utils';
import type { Mode } from '../lib/types';

export type TopBarProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
};

const TABS: { key: Mode; label: string }[] = [
  { key: 'solo', label: 'Solo' },
  { key: 'compare', label: 'Compare' },
];

export function TopBar({ mode, onModeChange }: TopBarProps) {
  const tablistRef = useRef<HTMLDivElement>(null);

  const handleTabKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const currentIdx = TABS.findIndex((t) => t.key === mode);
      const nextIdx =
        e.key === 'ArrowRight'
          ? (currentIdx + 1) % TABS.length
          : (currentIdx - 1 + TABS.length) % TABS.length;
      const nextTab = TABS[nextIdx];
      onModeChange(nextTab.key);
      const btn = tablistRef.current?.querySelector<HTMLButtonElement>(`#tab-${nextTab.key}`);
      btn?.focus();
    },
    [mode, onModeChange],
  );

  return (
    <header
      aria-label="microgpt-lab"
      className="flex items-center justify-between px-5 py-3"
      style={{
        backgroundColor: 'var(--surface-1)',
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <div className="flex items-baseline gap-0.5 select-none">
        <span className="text-lg font-bold tracking-tight" style={{ color: 'var(--model-a)' }}>
          micro
        </span>
        <span className="text-lg font-bold tracking-tight text-text-primary">gpt</span>
        <span className="text-lg font-extralight tracking-tight text-text-muted">-lab</span>
        <span
          className="ml-2 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-widest text-text-muted"
          style={{ backgroundColor: 'var(--surface-2)' }}
        >
          v1
        </span>
      </div>

      <div
        role="tablist"
        aria-label="View mode"
        className="flex gap-1"
        onKeyDown={handleTabKeyDown}
        ref={tablistRef}
      >
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            role="tab"
            id={`tab-${key}`}
            aria-selected={mode === key}
            aria-controls="main-tabpanel"
            tabIndex={mode === key ? 0 : -1}
            onClick={() => onModeChange(key)}
            className={cn(
              'relative rounded-md px-3 py-1.5 font-mono text-xs font-medium uppercase tracking-wider transition-colors duration-200',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              mode === key ? 'text-text-primary' : 'text-text-muted hover:text-text-secondary',
            )}
          >
            {label}
            {mode === key && (
              <span className="tab-indicator" style={{ background: 'var(--model-a)' }} />
            )}
          </button>
        ))}
      </div>
    </header>
  );
}

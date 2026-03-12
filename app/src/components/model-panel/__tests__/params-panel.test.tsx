import { render, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { ParamsPanel, type ParamsPanelProps } from '../params-panel';
import type { ModelParams } from '../../../lib/types';

afterEach(cleanup);

const defaultParams: ModelParams = {
  datasetId: 'prenoms-simple',
  n_embd: 32,
  n_head: 4,
  n_layer: 2,
  block_size: 16,
  lr: 0.001,
  temperature: 1.0,
  trainSteps: 500,
};

const noop = vi.fn();

function renderPanel(overrides: Partial<ParamsPanelProps> = {}) {
  const props: ParamsPanelProps = {
    params: defaultParams,
    onParamsChange: noop,
    onTrain: noop,
    onGenerate: noop,
    onReset: noop,
    trainState: 'idle',
    colorVar: 'a',
    ...overrides,
  };
  return render(<ParamsPanel {...props} />);
}

function getButton(queries: ReturnType<typeof render>, name: RegExp) {
  return queries.getByRole('button', { name }) as HTMLButtonElement;
}

describe('ParamsPanel', () => {
  it('renders "Entraîner" enabled when idle', () => {
    const q = renderPanel();
    const btn = getButton(q, /entraîn/i);
    expect(btn.textContent).toBe('Entraîner');
    expect(btn.disabled).toBe(false);
  });

  it('renders "Entraînement…" disabled when training', () => {
    const q = renderPanel({ trainState: 'training' });
    const btn = getButton(q, /entraîn/i);
    expect(btn.textContent).toBe('Entraînement…');
    expect(btn.disabled).toBe(true);
  });

  it('renders generate button enabled when idle', () => {
    const q = renderPanel({ colorVar: 'b' });
    const btn = getButton(q, /générer/i);
    expect(btn.disabled).toBe(false);
  });

  it('generate button is disabled when training', () => {
    const q = renderPanel({ trainState: 'training' });
    const btn = getButton(q, /générer/i);
    expect(btn.disabled).toBe(true);
  });

  it('renders reset button enabled when idle', () => {
    const q = renderPanel();
    const btn = getButton(q, /réinitialiser/i);
    expect(btn.disabled).toBe(false);
  });

  it('reset button is disabled when training', () => {
    const q = renderPanel({ trainState: 'training' });
    const btn = getButton(q, /réinitialiser/i);
    expect(btn.disabled).toBe(true);
  });

  it('calls onReset when reset button is clicked', () => {
    const onReset = vi.fn();
    const q = renderPanel({ onReset });
    getButton(q, /réinitialiser/i).click();
    expect(onReset).toHaveBeenCalledOnce();
  });
});

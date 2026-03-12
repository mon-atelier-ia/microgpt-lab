import { render, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { ParamsPanel } from '../params-panel';
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

describe('ParamsPanel', () => {
  it('renders train button with "Entraîner" when idle', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="idle"
        colorVar="a"
      />,
    );
    const trainBtn = getByRole('button', { name: /entraîn/i });
    expect(trainBtn.textContent).toBe('Entraîner');
    expect((trainBtn as HTMLButtonElement).disabled).toBe(false);
  });

  it('renders train button with "Entraînement…" and disabled when training', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="training"
        colorVar="a"
      />,
    );
    const trainBtn = getByRole('button', { name: /entraîn/i });
    expect(trainBtn.textContent).toBe('Entraînement…');
    expect((trainBtn as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders generate button', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="idle"
        colorVar="b"
      />,
    );
    const genBtn = getByRole('button', { name: /Générer/i });
    expect(genBtn).toBeDefined();
    expect((genBtn as HTMLButtonElement).disabled).toBe(false);
  });

  it('generate button is disabled when training', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="training"
        colorVar="a"
      />,
    );
    const genBtn = getByRole('button', { name: /Générer/i });
    expect((genBtn as HTMLButtonElement).disabled).toBe(true);
  });
});

describe('ParamsPanel — reset button', () => {
  it('renders reset button', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="idle"
        colorVar="a"
      />,
    );
    const resetBtn = getByRole('button', { name: /réinitialiser/i });
    expect(resetBtn).toBeDefined();
    expect((resetBtn as HTMLButtonElement).disabled).toBe(false);
  });

  it('reset button is disabled when training', () => {
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={noop}
        trainState="training"
        colorVar="a"
      />,
    );
    const resetBtn = getByRole('button', { name: /réinitialiser/i });
    expect((resetBtn as HTMLButtonElement).disabled).toBe(true);
  });

  it('calls onReset when reset button is clicked', () => {
    const onReset = vi.fn();
    const { getByRole } = render(
      <ParamsPanel
        params={defaultParams}
        onParamsChange={noop}
        onTrain={noop}
        onGenerate={noop}
        onReset={onReset}
        trainState="idle"
        colorVar="a"
      />,
    );
    getByRole('button', { name: /réinitialiser/i }).click();
    expect(onReset).toHaveBeenCalledOnce();
  });
});

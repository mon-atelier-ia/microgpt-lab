import type { CSSProperties, ReactNode } from 'react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Slider } from '../ui/slider';
import type { ModelParams, TrainState } from '../../lib/types';
import { PRESETS } from '../../data/presets';
import { validHeadCounts } from '../../lib/validation';
import { posToLr, lrToPos, formatLr } from './params-utils';

export type ParamsPanelProps = {
  params: ModelParams;
  onParamsChange: (params: ModelParams) => void;
  onTrain: () => void;
  onGenerate: () => void;
  trainState: TrainState;
  colorVar: 'a' | 'b';
};

const N_EMBD_OPTIONS = [8, 16, 32] as const;
const N_LAYER_OPTIONS = [1, 2, 4] as const;
const BLOCK_SIZE_OPTIONS = [8, 16, 32, 64] as const;
const TEXT_SECONDARY = 'var(--text-secondary)';

type FieldProps = { label: string; children: ReactNode };

function Field({ label, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs" style={{ color: TEXT_SECONDARY }}>
        {label}
      </label>
      {children}
    </div>
  );
}

type NumSelectProps = {
  label: string;
  value: number;
  options: readonly number[];
  onChange: (v: number) => void;
  disabled: boolean;
};

function NumSelect({ label, value, options, onChange, disabled }: NumSelectProps) {
  return (
    <Field label={label}>
      <Select value={String(value)} onValueChange={(v) => onChange(Number(v))} disabled={disabled}>
        <SelectTrigger className="h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o} value={String(o)} className="text-xs">
              {o}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}

type LabeledSliderProps = {
  label: string;
  display: string;
  accentColor: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onValueChange: (v: number) => void;
  disabled?: boolean;
};

function LabeledSlider({
  label,
  display,
  accentColor,
  min,
  max,
  step,
  value,
  onValueChange,
  disabled,
}: LabeledSliderProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs" style={{ color: TEXT_SECONDARY }}>
          {label}
        </span>
        <span className="font-mono text-xs" style={{ color: accentColor }}>
          {display}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onValueChange(v ?? value)}
        disabled={disabled}
      />
    </div>
  );
}

type ArchGridProps = {
  params: ModelParams;
  disabled: boolean;
  onChange: (patch: Partial<ModelParams>) => void;
};

function ArchGrid({ params, disabled, onChange }: ArchGridProps) {
  const heads = validHeadCounts(params.n_embd, [1, 2, 4, 8]);
  return (
    <div className="grid grid-cols-2 gap-2">
      <NumSelect
        label="n_embd"
        value={params.n_embd}
        options={N_EMBD_OPTIONS}
        onChange={(v) => onChange({ n_embd: v })}
        disabled={disabled}
      />
      <NumSelect
        label="n_head"
        value={params.n_head}
        options={heads}
        onChange={(v) => onChange({ n_head: v })}
        disabled={disabled}
      />
      <NumSelect
        label="n_layer"
        value={params.n_layer}
        options={N_LAYER_OPTIONS}
        onChange={(v) => onChange({ n_layer: v })}
        disabled={disabled}
      />
      <NumSelect
        label="block_size"
        value={params.block_size}
        options={BLOCK_SIZE_OPTIONS}
        onChange={(v) => onChange({ block_size: v })}
        disabled={disabled}
      />
    </div>
  );
}

export function ParamsPanel({
  params,
  onParamsChange,
  onTrain,
  onGenerate,
  trainState,
  colorVar,
}: ParamsPanelProps) {
  const accentColor = `var(--model-${colorVar})`;
  const isTraining = trainState === 'training';

  function update(patch: Partial<ModelParams>) {
    const next = { ...params, ...patch };
    if (patch.n_embd !== undefined) {
      const nextHeads = validHeadCounts(patch.n_embd, [1, 2, 4, 8]);
      if (!nextHeads.includes(next.n_head)) {
        next.n_head = nextHeads[nextHeads.length - 1] ?? 1;
      }
    }
    onParamsChange(next);
  }

  const trainBtnStyle: CSSProperties = { backgroundColor: accentColor, color: 'var(--surface-0)' };

  return (
    <div className="flex flex-col gap-4 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: accentColor }}>
        Modèle {colorVar.toUpperCase()}
      </h2>

      <Field label="Dataset">
        <Select
          value={params.datasetId}
          onValueChange={(v) => update({ datasetId: v })}
          disabled={isTraining}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PRESETS.map((p) => (
              <SelectItem key={p.id} value={p.id} className="text-xs">
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <ArchGrid params={params} disabled={isTraining} onChange={update} />

      <LabeledSlider
        label="Learning rate"
        display={formatLr(params.lr)}
        accentColor={accentColor}
        min={0}
        max={1}
        step={0.01}
        value={lrToPos(params.lr)}
        onValueChange={(pos) => update({ lr: posToLr(pos) })}
        disabled={isTraining}
      />

      <LabeledSlider
        label="Température"
        display={params.temperature.toFixed(2)}
        accentColor={accentColor}
        min={0.1}
        max={2.0}
        step={0.05}
        value={params.temperature}
        onValueChange={(t) => update({ temperature: t })}
      />

      <div className="flex gap-2">
        <Button
          size="sm"
          className="flex-1"
          onClick={onTrain}
          disabled={isTraining}
          style={trainBtnStyle}
        >
          {isTraining ? 'Entraînement…' : 'Entraîner'}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="flex-1"
          onClick={onGenerate}
          disabled={isTraining}
        >
          Générer
        </Button>
      </div>
    </div>
  );
}

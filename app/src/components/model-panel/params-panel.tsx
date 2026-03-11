import type { CSSProperties } from 'react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import type { ColorVar, ModelParams, TrainState } from '../../lib/types';
import { PRESETS } from '../../data/presets';
import {
  N_EMBD_OPTIONS,
  N_LAYER_OPTIONS,
  BLOCK_SIZE_OPTIONS,
  TRAIN_STEPS_OPTIONS,
  HEAD_OPTIONS,
} from '../../lib/constants';
import { validHeadCounts } from '../../lib/validation';
import { cn, modelColor, modelAccent, modelMuted } from '../../lib/utils';
import { posToLr, lrToPos, formatLr } from './params-utils';
import { Field, NumSelect, LabeledSlider } from './param-controls';

export type ParamsPanelProps = {
  params: ModelParams;
  onParamsChange: (params: ModelParams) => void;
  onTrain: () => void;
  onGenerate: () => void;
  trainState: TrainState;
  colorVar: ColorVar;
  glowClass?: string;
};

type ArchGridProps = {
  params: ModelParams;
  disabled: boolean;
  onChange: (patch: Partial<ModelParams>) => void;
};

type ActionButtonsProps = {
  onTrain: () => void;
  onGenerate: () => void;
  isTraining: boolean;
  trainBtnStyle: CSSProperties;
  genBtnStyle: CSSProperties;
};

function ActionButtons({
  onTrain,
  onGenerate,
  isTraining,
  trainBtnStyle,
  genBtnStyle,
}: ActionButtonsProps) {
  return (
    <div className="flex gap-2 pt-2">
      <Button
        size="sm"
        className={cn(
          'flex-1 font-semibold uppercase tracking-wider transition-shadow',
          isTraining && 'training-pulse',
        )}
        onClick={onTrain}
        disabled={isTraining}
        style={trainBtnStyle}
      >
        {isTraining ? 'Entraînement…' : 'Entraîner'}
      </Button>
      <span role="status" aria-live="polite" className="sr-only">
        {isTraining ? 'Entraînement en cours' : ''}
      </span>
      <Button
        size="sm"
        variant="outline"
        className="flex-1 font-semibold uppercase tracking-wider"
        onClick={onGenerate}
        disabled={isTraining}
        style={genBtnStyle}
      >
        Générer
      </Button>
    </div>
  );
}

function ArchGrid({ params, disabled, onChange }: ArchGridProps) {
  const heads = validHeadCounts(params.n_embd, [...HEAD_OPTIONS]);
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
  glowClass,
}: ParamsPanelProps) {
  const primary = modelColor(colorVar);
  const accent = modelAccent(colorVar);
  const muted = modelMuted(colorVar);
  const isTraining = trainState === 'training';

  function update(patch: Partial<ModelParams>) {
    const next = { ...params, ...patch };
    if (patch.n_embd !== undefined) {
      const nextHeads = validHeadCounts(patch.n_embd, [...HEAD_OPTIONS]);
      if (!nextHeads.includes(next.n_head)) {
        next.n_head = nextHeads[nextHeads.length - 1] ?? 1;
      }
    }
    onParamsChange(next);
  }

  const trainBtnStyle: CSSProperties = { backgroundColor: accent, color: 'var(--surface-0)' };
  const genBtnStyle: CSSProperties = { borderColor: muted, color: primary };

  return (
    <div className={cn('flex flex-col gap-4 rounded-lg p-4 panel-surface', glowClass)}>
      <h2
        className="instrument-header text-sm font-bold uppercase tracking-widest"
        style={{ color: primary }}
      >
        Modèle {colorVar.toUpperCase()}
      </h2>

      <Field label="Dataset">
        <Select
          value={params.datasetId}
          onValueChange={(v) => update({ datasetId: v })}
          disabled={isTraining}
        >
          <SelectTrigger className="h-8 font-mono text-xs">
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
        accentColor={accent}
        mutedColor={muted}
        min={0}
        max={1}
        step={0.01}
        value={lrToPos(params.lr)}
        onValueChange={(pos) => update({ lr: posToLr(pos) })}
      />

      <NumSelect
        label="Training steps"
        value={params.trainSteps}
        options={TRAIN_STEPS_OPTIONS}
        onChange={(v) => update({ trainSteps: v })}
        disabled={isTraining}
      />

      <LabeledSlider
        label="Température"
        display={params.temperature.toFixed(2)}
        accentColor={accent}
        mutedColor={muted}
        min={0.1}
        max={2.0}
        step={0.05}
        value={params.temperature}
        onValueChange={(t) => update({ temperature: t })}
      />

      <ActionButtons
        onTrain={onTrain}
        onGenerate={onGenerate}
        isTraining={isTraining}
        trainBtnStyle={trainBtnStyle}
        genBtnStyle={genBtnStyle}
      />
    </div>
  );
}

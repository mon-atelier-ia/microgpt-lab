import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Slider } from '../ui/slider';
import type { ReactNode } from 'react';

type FieldProps = { label: string; children: ReactNode };

export function Field({ label, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-text-secondary">{label}</label>
      {children}
    </div>
  );
}

export type NumSelectProps = {
  label: string;
  value: number;
  options: readonly number[];
  onChange: (v: number) => void;
  disabled: boolean;
};

export function NumSelect({ label, value, options, onChange, disabled }: NumSelectProps) {
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

export type LabeledSliderProps = {
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

export function LabeledSlider({
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
        <span className="text-xs text-text-secondary">{label}</span>
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

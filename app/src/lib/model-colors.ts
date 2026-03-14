import type { ColorVar } from './types';

/** Resolve a CSS custom property to its computed value. */
export function resolveVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

/** Return CSS var reference for a model color slot. */
export function modelColor(colorVar: ColorVar): string {
  return `var(--model-${colorVar})`;
}

/** Return CSS var reference for model accent (hue +15°). */
export function modelAccent(colorVar: ColorVar): string {
  return `var(--model-${colorVar}-accent)`;
}

/** Return CSS var reference for model muted (low-lightness background tint). */
export function modelMuted(colorVar: ColorVar): string {
  return `var(--model-${colorVar}-muted)`;
}

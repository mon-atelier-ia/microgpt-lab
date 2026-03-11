import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { ColorVar } from './types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Resolve a CSS custom property to its computed value. */
export function resolveVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

/** Return CSS var reference for a model color slot. */
export function modelColor(colorVar: ColorVar): string {
  return `var(--model-${colorVar})`;
}

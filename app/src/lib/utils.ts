import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Resolve a CSS custom property to its computed value. */
export function resolveVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

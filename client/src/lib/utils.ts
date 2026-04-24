/**
 * Utility functions for the UI component library
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Combines class names using clsx and merges Tailwind classes
 * Following shadcn/ui patterns for class composition
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

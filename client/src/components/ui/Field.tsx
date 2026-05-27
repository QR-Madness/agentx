/**
 * Input / Textarea — thin wrappers over the globally-styled native elements
 * (see `styles/base.css`) that add a consistent `error` state. Keeps form
 * fields visually uniform without each feature re-styling raw inputs.
 */

import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';
import './Field.css';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Render the error (invalid) visual state. */
  error?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, error, ...props }, ref) => (
    <input
      ref={ref}
      className={cn('ax-field', error && 'ax-field--error', className)}
      aria-invalid={error || undefined}
      {...props}
    />
  )
);

Input.displayName = 'Input';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Render the error (invalid) visual state. */
  error?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn('ax-field', error && 'ax-field--error', className)}
      aria-invalid={error || undefined}
      {...props}
    />
  )
);

Textarea.displayName = 'Textarea';

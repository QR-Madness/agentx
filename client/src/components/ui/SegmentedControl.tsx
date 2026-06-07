/**
 * SegmentedControl — a pill/segment selector (radiogroup). Replaces ad-hoc
 * `<select>`s and one-off pill rows with one accessible, animated primitive.
 *
 * A sliding highlight (framer `layoutId`) follows the active segment; it's gated
 * behind `prefers-reduced-motion`. Arrow keys move the selection.
 */

import { useId } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';
import './SegmentedControl.css';

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  title?: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel?: string;
  size?: 'sm' | 'md';
  className?: string;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  size = 'md',
  className,
}: SegmentedControlProps<T>) {
  const groupId = useId();
  const reduce = useReducedMotion();
  const enabledValues = options.filter(o => !o.disabled).map(o => o.value);

  const move = (dir: 1 | -1) => {
    const i = enabledValues.indexOf(value);
    const next = enabledValues[(i + dir + enabledValues.length) % enabledValues.length];
    if (next !== undefined) onChange(next);
  };

  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn('ax-segmented', size === 'sm' && 'ax-segmented--sm', className)}
      onKeyDown={(e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); move(1); }
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      }}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            disabled={opt.disabled}
            title={opt.title}
            className={cn('ax-segment', active && 'ax-segment--active', opt.disabled && 'ax-segment--disabled')}
            onClick={() => !opt.disabled && onChange(opt.value)}
          >
            {active && !reduce && (
              <motion.span
                layoutId={`seg-${groupId}`}
                className="ax-segment__pill"
                transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              />
            )}
            {active && reduce && <span className="ax-segment__pill" />}
            <span className="ax-segment__content">
              {opt.icon}
              {opt.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

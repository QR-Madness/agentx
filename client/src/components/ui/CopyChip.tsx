/**
 * CopyChip — a compact click-to-copy chip with a transient ✓. Used for the agent_id
 * (and reusable for run ids, channels, keys).
 */

import { useState, useRef, useEffect } from 'react';
import { Copy, Check } from 'lucide-react';
import { cn } from '../../lib/utils';
import './CopyChip.css';

interface CopyChipProps {
  value: string;
  /** Optional display label (defaults to the value). */
  label?: string;
  title?: string;
  className?: string;
}

export function CopyChip({ value, label, title, className }: CopyChipProps) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked — no-op */
    }
  };

  return (
    <button
      type="button"
      className={cn('ax-copychip', copied && 'ax-copychip--copied', className)}
      onClick={copy}
      title={title ?? `Copy ${value}`}
      aria-label={`Copy ${value}`}
    >
      <span className="ax-copychip__text">{label ?? value}</span>
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  );
}

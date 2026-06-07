/**
 * ControlCard — a static settings card for the dashboard-style editors: an icon +
 * title, an optional at-a-glance summary chip in the header, and a body. Animates in
 * (gated by reduced-motion). The building block of the agent-profile card grid;
 * reusable for other settings panels.
 */

import type { ReactNode } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';
import './ControlCard.css';

interface ControlCardProps {
  icon?: ReactNode;
  title: ReactNode;
  /** At-a-glance value chip shown on the header (e.g. "opus · 200k"). */
  summary?: ReactNode;
  children: ReactNode;
  /** Span the full width of the grid (System Prompt, Tools…). */
  full?: boolean;
  className?: string;
}

export function ControlCard({ icon, title, summary, children, full, className }: ControlCardProps) {
  const reduce = useReducedMotion();
  return (
    <motion.section
      className={cn('control-card', full && 'control-card--full', className)}
      variants={reduce ? undefined : { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
    >
      <header className="control-card__head">
        <span className="control-card__title">
          {icon && <span className="control-card__icon">{icon}</span>}
          {title}
        </span>
        {summary != null && <span className="control-card__summary">{summary}</span>}
      </header>
      <div className="control-card__body">{children}</div>
    </motion.section>
  );
}

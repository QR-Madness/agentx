/**
 * SectionHeader — title + optional icon/description block. Consolidates the
 * `.section-title` + description markup repeated across settings/memory panels.
 */

import type { ReactNode } from 'react';
import { cn } from '../../lib/utils';
import './SectionHeader.css';

export interface SectionHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  /** Leading icon (e.g. a Lucide element). */
  icon?: ReactNode;
  /** Trailing content aligned to the right (actions, badges). */
  actions?: ReactNode;
  className?: string;
}

export function SectionHeader({ title, description, icon, actions, className }: SectionHeaderProps) {
  return (
    <div className={cn('ax-section-header', className)}>
      <div className="ax-section-header__main">
        <h3 className="section-title">
          {icon && <span className="section-title-icon">{icon}</span>}
          {title}
        </h3>
        {description && <p className="ax-section-header__desc">{description}</p>}
      </div>
      {actions && <div className="ax-section-header__actions">{actions}</div>}
    </div>
  );
}

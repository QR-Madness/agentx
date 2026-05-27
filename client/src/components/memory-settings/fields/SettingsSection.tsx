/**
 * SettingsSection — titled block wrapper for a settings form. Keeps the
 * `.settings-section` / `.settings-section-title` layout (styled in
 * styles/MemoryPanel.css) while standardizing the title + description markup.
 */

import type { ReactNode } from 'react';

interface SettingsSectionProps {
  title: string;
  icon?: ReactNode;
  description?: ReactNode;
  children: ReactNode;
}

export function SettingsSection({ title, icon, description, children }: SettingsSectionProps) {
  return (
    <div className="settings-section">
      <h3 className="settings-section-title">
        {icon}
        {title}
      </h3>
      {description && <p className="settings-description">{description}</p>}
      {children}
    </div>
  );
}

/**
 * SettingsSection — titled block wrapper for a settings form. Keeps the
 * `.settings-section` / `.settings-section-title` layout (styled in
 * styles/MemoryPanel.css) while standardizing the title + description markup.
 *
 * `variant="disclosure"` collapses the block behind its title, for the
 * cross-cutting knobs most people never open. Sections used to express that
 * three different ways — a plain block titled "Advanced", a bare
 * `<div className="settings-section experimental">`, and an Experimental badge
 * — so nothing about "this is the deep end" was consistent or reusable.
 *
 * Those are gone. A setting's depth is now declared as its `tier` in
 * `settings_registry.py` and rendered from the manifest, so the UI and the
 * generated reference agree on what counts as advanced. The last hand-rolled
 * block said "Experimental" over two switches that ship on and are declared
 * essential — which is the failure mode of writing a judgement into markup.
 */

import { useId, useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';

interface SettingsSectionProps {
  title: string;
  icon?: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  /** Anchor id — a stable target for linking to a section. */
  id?: string;
  /** Right-aligned slot in the title row (status chip, action button). */
  actions?: ReactNode;
  variant?: 'default' | 'disclosure';
  /** Start a disclosure open. Ignored for the default variant. */
  defaultOpen?: boolean;
}

export function SettingsSection({
  title, icon, description, children, id, actions,
  variant = 'default', defaultOpen = false,
}: SettingsSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();

  if (variant === 'disclosure') {
    return (
      <div className="settings-section settings-section--disclosure" id={id}>
        <h3 className="settings-section-title">
          <button
            type="button"
            className="settings-disclosure-trigger"
            aria-expanded={open}
            aria-controls={bodyId}
            onClick={() => setOpen(o => !o)}
          >
            <ChevronRight
              size={14}
              className={open ? 'settings-disclosure-chevron is-open' : 'settings-disclosure-chevron'}
              aria-hidden="true"
            />
            {icon}
            {title}
          </button>
          {actions && <span className="settings-section-actions">{actions}</span>}
        </h3>
        <div id={bodyId} hidden={!open}>
          {description && <p className="settings-description">{description}</p>}
          {children}
        </div>
      </div>
    );
  }

  return (
    <div className="settings-section" id={id}>
      <h3 className="settings-section-title">
        {icon}
        {title}
        {actions && <span className="settings-section-actions">{actions}</span>}
      </h3>
      {description && <p className="settings-description">{description}</p>}
      {children}
    </div>
  );
}

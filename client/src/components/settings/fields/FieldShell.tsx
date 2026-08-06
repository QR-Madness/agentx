/**
 * FieldShell — the frame every settings control shares: a label row, the
 * control, and a hint line.
 *
 * It exists to fix two things at once.
 *
 * **Accessibility.** Five of the eight field primitives rendered a `<Label>`
 * with no `htmlFor` and a control with no `id`, so the label announced nothing;
 * NumberField's accessible name resolved to its `title` — the hint sentence,
 * not the setting's name. FieldShell owns the id, hands it to the control via a
 * render prop, and wires the hint through `aria-describedby` so screen readers
 * get the label *and* the explanation.
 *
 * **Manifest chrome.** When a control is bound to its manifest entry, the label
 * row also carries the "changed from default" dot, the reset control, and the
 * help popover. All of it is optional: pass no `binding` and you get the same
 * markup the fields rendered before.
 */

import { useId, type ReactNode } from 'react';
import { RotateCcw } from 'lucide-react';
import { Label } from '../../ui';
import { SettingHelp } from '../SettingHelp';
import { SettingBadge, type SettingBadgeKind } from '../SettingBadge';
import type { SettingBinding } from '../../unified-settings/SettingsManifestContext';

export interface FieldChromeProps {
  /** Manifest entry for this key — supplies default, bounds, help, tier. */
  binding?: SettingBinding | null;
  /** Reset to the shipped default. Rendered only when the value differs. */
  onReset?: () => void;
  /** Extra badge beside the label (e.g. "Recommended"). */
  badge?: SettingBadgeKind;
}

interface FieldShellProps extends FieldChromeProps {
  label: ReactNode;
  hint?: ReactNode;
  /** Receives the ids to attach to the control. */
  children: (ids: { id: string; describedBy?: string }) => ReactNode;
  /** Label text for the help popover heading + reset button announcement. */
  labelText?: string;
  className?: string;
  /**
   * False when the control can't be reached by `htmlFor` — Radix Slider puts
   * `role="slider"` on the thumb, not on anything the id lands on, and it names
   * itself via `aria-label`. Rendering a `<label for>` at a non-existent id
   * would just recreate the orphaned-label defect in a new place.
   */
  labelledControl?: boolean;
}

export function FieldShell({
  label,
  hint,
  binding,
  onReset,
  badge,
  children,
  labelText,
  className,
  labelledControl = true,
}: FieldShellProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const name = labelText ?? (typeof label === 'string' ? label : 'this setting');
  const modified = Boolean(binding?.isModified);

  return (
    <div className={className ?? 'setting-row'}>
      <div className="setting-label-row">
        {labelledControl ? (
          <Label htmlFor={id} className="setting-label">
            {label}
          </Label>
        ) : (
          <span className="setting-label" aria-hidden="true">
            {label}
          </span>
        )}
        {modified && (
          <span
            className="setting-modified-dot"
            title="Changed from the default"
            aria-label="Changed from the default"
          />
        )}
        {badge && <SettingBadge kind={badge} />}
        {binding?.tier === 'experimental' && <SettingBadge kind="experimental" />}
        <SettingHelp help={binding?.help} label={name} />
        {modified && onReset && (
          <button
            type="button"
            className="setting-reset-btn"
            onClick={onReset}
            title="Reset to default"
            aria-label={`Reset ${name} to default`}
          >
            <RotateCcw size={12} aria-hidden="true" />
          </button>
        )}
      </div>
      {children({ id, describedBy: hint ? hintId : undefined })}
      {hint && (
        <span className="setting-hint" id={hintId}>
          {hint}
        </span>
      )}
    </div>
  );
}

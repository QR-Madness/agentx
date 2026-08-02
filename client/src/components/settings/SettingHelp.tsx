/**
 * SettingHelp — the "?" beside a control, and the panel behind it.
 *
 * The prose comes from the settings manifest, which reads it from
 * `api/agentx_ai/settings_help.yaml`. The same file generates the docs-site
 * reference, so what you read here and what you read in the documentation are
 * the same sentences — they cannot drift.
 *
 * Renders nothing at all when a setting has no authored help yet, so sections
 * can adopt this before every key has been written up.
 */

import { HelpCircle } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../ui';
import type { SettingHelp as SettingHelpContent } from '../../lib/api';

/** Long-form fields, in reading order. `summary` renders inline, not here. */
const DETAIL_FIELDS: { key: keyof SettingHelpContent; label: string }[] = [
  { key: 'what', label: 'What it is' },
  { key: 'how', label: 'How it works' },
  { key: 'why', label: 'When to change it' },
  { key: 'manage', label: 'Managing it' },
];

export function hasHelp(help?: SettingHelpContent): boolean {
  if (!help) return false;
  return Boolean(
    help.summary || help.what || help.how || help.why || help.manage
  );
}

interface SettingHelpProps {
  help?: SettingHelpContent;
  /** The setting's label — announced so the trigger isn't a bare "help". */
  label: string;
}

export function SettingHelp({ help, label }: SettingHelpProps) {
  const details = DETAIL_FIELDS.filter(f => (help?.[f.key] || '').trim());
  if (!help || (!help.summary && details.length === 0)) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="setting-help-trigger"
          aria-label={`About ${label}`}
        >
          <HelpCircle size={13} aria-hidden="true" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="setting-help-panel w-80">
        <h4 className="setting-help-title">{label}</h4>
        {help.summary && <p className="setting-help-summary">{help.summary}</p>}
        {details.map(({ key, label: heading }) => (
          <section key={key} className="setting-help-block">
            <h5 className="setting-help-heading">{heading}</h5>
            <p className="setting-help-body">{help[key]}</p>
          </section>
        ))}
      </PopoverContent>
    </Popover>
  );
}

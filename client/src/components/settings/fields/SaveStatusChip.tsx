/**
 * SaveStatusChip — tiny autosave indicator for settings sections using
 * useSettingsAutosave (Saving… / Saved ✓ / error). Renders nothing when idle.
 */

import { Check, CloudUpload, TriangleAlert } from 'lucide-react';
import type { SettingsSaveStatus } from '../../../lib/hooks';

export function SaveStatusChip({ status }: { status: SettingsSaveStatus }) {
  if (status === 'idle') return null;
  if (status === 'saving') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-fg-muted" role="status">
        <CloudUpload size={13} className="animate-pulse" /> Saving…
      </span>
    );
  }
  if (status === 'saved') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-success" role="status">
        <Check size={13} /> Saved
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-error" role="status">
      <TriangleAlert size={13} /> Not saved
    </span>
  );
}

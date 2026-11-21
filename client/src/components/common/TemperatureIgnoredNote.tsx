import { AlertCircle } from 'lucide-react';
import { temperatureIgnoredBy } from '../../lib/modelSampling';

interface TemperatureIgnoredNoteProps {
  /** The profile's selected model id (`provider:model`); '' inherits the system default. */
  model: string;
  /** Element id, so the slider can point at the note with `aria-describedby`. */
  id?: string;
}

/**
 * Inline note under a Temperature slider, shown when the selected model's provider
 * ignores the value (see `temperatureIgnoredBy`); renders nothing otherwise.
 * The slider stays live: the value is saved and applies again on another provider.
 */
export function TemperatureIgnoredNote({ model, id }: TemperatureIgnoredNoteProps) {
  const provider = temperatureIgnoredBy(model);
  if (!provider) return null;
  return (
    <p id={id} className="m-0 flex items-start gap-1.5 text-xs text-fg-muted">
      <AlertCircle size={12} aria-hidden="true" className="mt-0.5 shrink-0 text-info" />
      <span>No effect on {provider} models — they use their default sampling.</span>
    </p>
  );
}

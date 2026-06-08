/**
 * VoicePicker — a per-speech-model voice selector. Renders a dropdown of the
 * model's known voices (from `lib/voiceCatalog`) plus a **Custom…** option that
 * reveals a free-text input; when the model has no known voices it degrades to a
 * pure free-text field. Style-agnostic (native `select`/`input`) so it inherits
 * the surrounding form's CSS — used in the profile-editor Voice card and the
 * in-mode voice settings.
 */

import { useEffect, useMemo, useState } from 'react';
import { voicesFor } from '../../lib/voiceCatalog';

const CUSTOM = '__custom__';

interface VoicePickerProps {
  /** Speech model id (provider:model or bare) used to look up known voices. */
  model?: string | null;
  /** Current voice id ('' = the model's default). */
  value: string;
  onChange: (voice: string) => void;
  placeholder?: string;
  className?: string;
  selectClassName?: string;
  inputClassName?: string;
}

export function VoicePicker({
  model,
  value,
  onChange,
  placeholder = 'Custom voice id',
  className,
  selectClassName,
  inputClassName,
}: VoicePickerProps) {
  const options = useMemo(() => voicesFor(model), [model]);
  const isKnown = options.some((o) => o.id === value);
  const [custom, setCustom] = useState(!!value && !isKnown);

  // A non-empty value that isn't a known voice (e.g. after switching models) keeps
  // the picker in custom mode so the typed id stays editable.
  useEffect(() => {
    if (value && !options.some((o) => o.id === value)) setCustom(true);
  }, [options, value]);

  if (options.length === 0) {
    return (
      <input
        type="text"
        className={inputClassName}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  const selectValue = custom ? CUSTOM : isKnown ? value : '';
  return (
    <div className={className} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <select
        className={selectClassName}
        value={selectValue}
        onChange={(e) => {
          const val = e.target.value;
          if (val === CUSTOM) {
            setCustom(true);
          } else {
            setCustom(false);
            onChange(val); // '' = model default
          }
        }}
      >
        <option value="">Model default</option>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.label}
          </option>
        ))}
        <option value={CUSTOM}>Custom…</option>
      </select>
      {custom && (
        <input
          type="text"
          className={inputClassName}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}

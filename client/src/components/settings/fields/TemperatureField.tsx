/**
 * TemperatureField — the SliderField for a sampling temperature that belongs
 * to a model setting.
 *
 * When that model's provider ignores the temperature (see
 * `temperatureIgnoredBy`), the slider carries the same note the profile editor
 * shows, announced through `aria-describedby`. The slider stays live: the value
 * is saved and applies again on another provider.
 *
 * Pass the model setting's stored value, not a display fallback. An empty,
 * `inherit` or `role:` value is resolved server-side, so it shows no note.
 */

import type { ComponentProps } from 'react';
import { TemperatureIgnoredNote } from '../../common/TemperatureIgnoredNote';
import { temperatureIgnoredBy } from '../../../lib/modelSampling';
import { SliderField } from './SliderField';

type TemperatureFieldProps = Omit<ComponentProps<typeof SliderField>, 'label' | 'note'> & {
  /** The stored model id (`provider:model`) this temperature is sent with. */
  model: string;
  /** Defaults to "Temperature". */
  label?: string;
};

export function TemperatureField({ model, label = 'Temperature', ...props }: TemperatureFieldProps) {
  return (
    <SliderField
      {...props}
      label={label}
      note={temperatureIgnoredBy(model) ? <TemperatureIgnoredNote model={model} /> : undefined}
    />
  );
}

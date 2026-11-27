/**
 * streamErrorFixes — where in Settings a failed chat turn can be fixed.
 *
 * The chat stream's `error` event carries a stable `code` for the few errors a
 * client can act on (server side: `AgentXError.code`). Each maps here to the
 * Settings screen, and the control when there is one, that fixes it, so the
 * error toast can open it in one click. Without this the user read "set the
 * Global Default Model in Settings → Model Roles" and had to find it by hand.
 *
 * A code missing from this table, or no code at all, keeps the plain error
 * toast: the client acts only on reasons it knows.
 */

import type { StreamErrorCode } from '../../lib/api/streaming';
import { DEFAULT_MODEL_SECTION, DEFAULT_MODEL_SETTING } from '../unified-settings/defaultModel';

export interface StreamErrorFix {
  /** Toast title: what went wrong, in a few words. */
  title: string;
  /** Text of the toast's action button. */
  label: string;
  /** Settings section to open. */
  section: string;
  /** `store:key` anchor to land on, when a single control is the fix. */
  setting?: string;
}

const STREAM_ERROR_FIXES: Record<StreamErrorCode, StreamErrorFix> = {
  no_model_chosen: {
    title: 'No model chosen',
    label: 'Open Model Roles',
    section: DEFAULT_MODEL_SECTION,
    setting: DEFAULT_MODEL_SETTING,
  },
  // No setting to land on: the first step is connecting a provider, which is
  // a card on that screen rather than one control.
  no_provider_connected: {
    title: 'No model provider connected',
    label: 'Open Model Providers',
    section: 'providers',
  },
};

/** The fix for a stream error `code`, or undefined when there is none to offer. */
export function streamErrorFix(code: string | undefined): StreamErrorFix | undefined {
  if (!code || !Object.prototype.hasOwnProperty.call(STREAM_ERROR_FIXES, code)) return undefined;
  return STREAM_ERROR_FIXES[code as StreamErrorCode];
}

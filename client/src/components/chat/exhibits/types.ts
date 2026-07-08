/**
 * Shared prop contract for exhibit element renderers.
 *
 * Every renderer in `elementRegistry` receives the full typed element plus
 * interaction context and narrows on `element.type`. Lives in its own module so
 * the registry and the renderers can both import it without a cycle.
 */

import type { ExhibitElement } from '../../../lib/exhibits';

export interface ElementRenderProps {
  element: ExhibitElement;
  /** The owning exhibit message's id — passed back when submitting a choice. */
  messageId: string;
  /** The user's prior selection (choice elements) — renders the choice resolved. */
  answeredValue?: string;
  /** A turn is in flight — interactive elements render inert. */
  busy?: boolean;
  /** Submit a choice selection (sent as the next user turn). */
  onSubmitChoice?: (value: string, messageId: string) => void;
  /** Owning exhibit's title — passed for card-less renderers (e.g. citation-only). */
  containerTitle?: string;
}

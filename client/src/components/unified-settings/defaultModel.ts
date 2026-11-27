/**
 * The Global Default Model (`preferences.default_model`), as the rest of
 * Settings points at it.
 *
 * An agent whose profile names no model runs on this. Left unset, that agent
 * falls to the LM Studio floor, and on an install without LM Studio its
 * conversation stops with "No model chosen": the server never substitutes a
 * cloud model nobody chose, because that would spend money without consent.
 *
 * So the setting is worth reaching *before* a conversation fails. The Overview's
 * setup list and the Model Providers nudge both send people to the control on
 * Model Roles, which renders the anchor they land on. Declaring the anchor once
 * keeps the three from drifting apart; a stale anchor fails silently (the
 * section opens, but nothing is scrolled to or flashed).
 */

/** Settings section that renders the control. */
export const DEFAULT_MODEL_SECTION = 'model-roles';

/** Config key behind the control. */
export const DEFAULT_MODEL_KEY = 'preferences.default_model';

/** `store:key` of the control's `data-setting` anchor (see `entryId`). */
export const DEFAULT_MODEL_SETTING = `config:${DEFAULT_MODEL_KEY}`;

/** Why it matters, for the places that raise it before it is needed. */
export const DEFAULT_MODEL_WHY =
  "Agents without a model of their own run on it. Until it's set, they can't start a conversation unless LM Studio is connected.";

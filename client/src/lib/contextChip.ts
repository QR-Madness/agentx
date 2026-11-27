/**
 * contextChipState — pure display logic for the context-usage indicator that
 * sits beside the composer's draft-message token estimate.
 *
 * Shown **whenever the model's window is known** — including a brand-new
 * conversation at 0% — so the user always has a read on how full the context
 * is. It only needs the window: `used` defaults to 0 (a fresh chat that hasn't
 * completed a turn yet). Near the ceiling it switches to a warn state with a
 * hint that older turns are summarized automatically (the server compresses
 * just before the context limit — see context.verbatim_budget_ratio).
 *
 * The seed helpers below choose the window the chip reads before a turn's
 * `done` event reports the server's own.
 */

import type { ModelInfo } from './api/types';
import type { ConversationMessage } from './messages';

export interface ChipContextInfo {
  window: number;
  used?: number;
  summarized?: boolean;
  droppedTurns?: number;
}

export interface ContextChipState {
  /** e.g. "62% context" */
  label: string;
  /** Warn styling (composer-chip warn) at high usage. */
  warn: boolean;
  /** Tooltip text. */
  title: string;
}

const WARN_AT = 0.75;

export function contextChipState(info: ChipContextInfo | null | undefined): ContextChipState | null {
  // The window is the only hard requirement — without it there is no ratio to
  // show. `used` is optional (0 on a fresh conversation).
  if (!info || !info.window || info.window <= 0) return null;
  const used = Math.max(0, info.used ?? 0);
  const ratio = used / info.window;
  if (!Number.isFinite(ratio)) return null;

  const pct = Math.min(Math.round(ratio * 100), 100);
  const tokens = `${used.toLocaleString()} / ${info.window.toLocaleString()} tokens`;
  if (ratio < WARN_AT) {
    return { label: `${pct}% context`, warn: false, title: `Context: ${tokens}` };
  }
  let title = `Context ${pct}% full — older turns are summarized automatically (${tokens})`;
  if (info.droppedTurns && info.droppedTurns > 0) {
    title += ` · ${info.droppedTurns} older turn${info.droppedTurns === 1 ? '' : 's'} compressed`;
  } else if (info.summarized) {
    title += ' · summary active';
  }
  return { label: `${pct}% context`, warn: true, title };
}

// ---------------------------------------------------------------------------
// Seeding: the chip before the first `done`
// ---------------------------------------------------------------------------

/** The model-catalog fields the seed reads (`/api/providers/models`). */
export type SeedCatalogEntry = Pick<ModelInfo, 'id' | 'context_window' | 'context_length'>;

/** Model Limits overrides (`/api/config/context-limits`), keyed by model id. */
export interface SeedLimits {
  models?: Record<string, { context_window?: number } | null | undefined>;
}

/**
 * The model a tab's chip measures against before its first `done`: the model
 * recorded on the latest assistant turn, else the composer's effective model
 * (conversation override, then profile model), else the Global Default Model.
 *
 * A conversation on "Default model" names no model of its own. Without the
 * last step it had nothing to look up, so a fresh one showed no chip while a
 * conversation with an explicit model showed "0% context".
 */
export function seedModelId(
  messages: readonly ConversationMessage[],
  effectiveModel?: string | null,
  globalDefaultModel?: string | null,
): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.type === 'assistant' && m.model) return m.model;
  }
  return effectiveModel || globalDefaultModel || undefined;
}

/**
 * Chip state for a tab with no live `done` value yet, or null when the
 * model's window is unknown. Resolves the window like the server does: a
 * Model Limits override wins over the catalog. An id the catalog doesn't list
 * seeds nothing; a percentage of a guessed window would be a false reading.
 * `used` is the latest recorded turn's tokens, else ~chars/4, else 0.
 */
export function seedContextInfo(input: {
  messages: readonly ConversationMessage[];
  modelId: string | undefined;
  models: readonly SeedCatalogEntry[];
  limits?: SeedLimits | null;
}): { window: number; used: number } | null {
  const { messages, modelId, models, limits } = input;
  if (!modelId) return null;
  const info = models.find((m) => m.id === modelId);
  const window =
    limits?.models?.[modelId]?.context_window || info?.context_window || info?.context_length;
  if (!window || window <= 0) return null;
  return { window, used: seedUsedTokens(messages) };
}

function seedUsedTokens(messages: readonly ConversationMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.type === 'assistant' && m.tokensInput !== undefined) {
      return m.tokensInput + (m.tokensOutput ?? 0);
    }
  }
  const chars = messages.reduce(
    (n, m) => n + (m.type === 'user' || m.type === 'assistant' ? m.content.length : 0),
    0,
  );
  return Math.ceil(chars / 4);
}

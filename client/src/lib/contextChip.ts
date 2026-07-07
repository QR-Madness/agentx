/**
 * contextChipState — pure display logic for the composer context chip.
 *
 * The chip is the single context-usage indicator (it replaced the header
 * usage bar). It is shown **whenever the model's window is known** — including
 * a brand-new conversation at 0% — so the user always has a read on how full
 * the context is. It only needs the window: `used` defaults to 0 (a fresh chat
 * that hasn't completed a turn yet). Near the ceiling it switches to a warn
 * state with a hint that older turns are summarized automatically (the server
 * compresses just before the context limit — see context.verbatim_budget_ratio).
 */

export interface ChipContextInfo {
  window: number;
  used?: number;
  summarized?: boolean;
  droppedTurns?: number;
}

export interface ContextChipState {
  /** e.g. "62% ctx" */
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
    return { label: `${pct}% ctx`, warn: false, title: `Context: ${tokens}` };
  }
  let title = `Context ${pct}% full — older turns are summarized automatically (${tokens})`;
  if (info.droppedTurns && info.droppedTurns > 0) {
    title += ` · ${info.droppedTurns} older turn${info.droppedTurns === 1 ? '' : 's'} compressed`;
  } else if (info.summarized) {
    title += ' · summary active';
  }
  return { label: `${pct}% ctx`, warn: true, title };
}

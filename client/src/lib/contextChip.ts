/**
 * contextChipState — pure display logic for the composer context chip.
 *
 * The chip is the single context-usage indicator (it replaced the header
 * usage bar): hidden while usage is low, a quiet percentage once the
 * conversation is meaningfully long, and a warn state near the ceiling with
 * a hint that older turns are summarized automatically (the server compresses
 * just before the context limit — see context.verbatim_budget_ratio).
 */

export interface ChipContextInfo {
  window: number;
  used: number;
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

const SHOW_AT = 0.5;
const WARN_AT = 0.75;

export function contextChipState(info: ChipContextInfo | null | undefined): ContextChipState | null {
  if (!info || !info.window || !info.used) return null;
  const ratio = info.used / info.window;
  if (!Number.isFinite(ratio) || ratio < SHOW_AT) return null;

  const pct = Math.min(Math.round(ratio * 100), 100);
  const tokens = `${info.used.toLocaleString()} / ${info.window.toLocaleString()} tokens`;
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

/**
 * thinkingModes — the unified per-conversation "how should the agent think"
 * selection: the chat thinking patterns PLUS Research Mode as one choice.
 *
 * Research and thinking patterns are mutually exclusive by design — the server
 * short-circuits pattern resolution to an empty plan on research turns
 * (`resolve_thinking_plan`: research keeps its own rigorous prompt, no pattern
 * stacking). Presenting them as one mode makes that rule visible instead of
 * letting a picked pattern silently no-op under Research.
 *
 * The wire contract is unchanged: the single mode DERIVES the two request
 * fields (`research_mode`, `thinking_pattern`). Legacy per-tab fields
 * (`researchMode`, `thinkingPattern`) are still written on every change so a
 * downgraded client keeps working; `thinkingMode` wins on read.
 */

import type { ConversationTab } from './storage';

/** Sentinel mode value for Research Mode (not a thinking pattern). */
export const RESEARCH_MODE = 'research';

export interface ThinkingModeOption {
  value: string;
  label: string;
  hint: string;
}

/** '' = Auto (profile/auto chain). Pattern values mirror the server's chat patterns. */
export const THINKING_MODE_OPTIONS: ThinkingModeOption[] = [
  { value: '', label: 'Auto', hint: 'Pick the best pattern per message' },
  { value: 'native', label: 'Native', hint: 'The model thinks freely — no scaffold' },
  { value: 'cot', label: 'Step-by-step', hint: 'Explicit numbered reasoning steps' },
  { value: 'step_back', label: 'Step-back', hint: 'Distill governing principles first' },
  { value: 'reflection', label: 'Reflect', hint: 'Draft, self-critique, improve — one pass' },
  { value: 'deep_reflection', label: 'Reflect deeply', hint: 'Live draft → critique → final (extra calls)' },
  { value: 'self_consistency', label: 'Consensus', hint: 'Sample several solutions, keep the agreement (extra calls)' },
  { value: RESEARCH_MODE, label: 'Research', hint: 'Deep, cited research engagement — replaces thinking patterns' },
];

export const THINKING_MODE_LABELS: Record<string, string> = Object.fromEntries(
  THINKING_MODE_OPTIONS.filter(o => o.value).map(o => [o.value, o.label]),
);

/** Per-pattern availability, mirrored from Settings → Intelligence → Thinking
 *  (`reasoning.*`) + Research (`research.enabled`). Everything defaults ON. */
export interface ThinkingModeGates {
  /** reasoning.chat_patterns_enabled — the patterns kill-switch. */
  patternsEnabled: boolean;
  cot: boolean;
  stepBack: boolean;
  reflection: boolean;
  selfConsistency: boolean;
  /** research.enabled AND not inside a workflow (team runs are their own mode). */
  research: boolean;
}

export const DEFAULT_MODE_GATES: ThinkingModeGates = {
  patternsEnabled: true,
  cot: true,
  stepBack: true,
  reflection: true,
  selfConsistency: true,
  research: true,
};

/** The options actually offered given the settings gates. Auto/Native always
 *  remain (with patterns killed they mean "no scaffold" — honest). */
export function availableThinkingModes(gates: ThinkingModeGates): ThinkingModeOption[] {
  return THINKING_MODE_OPTIONS.filter(o => {
    switch (o.value) {
      case '':
      case 'native':
        return true;
      case 'cot':
        return gates.patternsEnabled && gates.cot;
      case 'step_back':
        return gates.patternsEnabled && gates.stepBack;
      case 'reflection':
      case 'deep_reflection':
        return gates.patternsEnabled && gates.reflection;
      case 'self_consistency':
        return gates.patternsEnabled && gates.selfConsistency;
      case RESEARCH_MODE:
        return gates.research;
      default:
        return true;
    }
  });
}

/** Effective mode of a tab — `thinkingMode` wins, legacy fields back-fill. */
export function thinkingModeOf(
  tab: Pick<ConversationTab, 'thinkingMode' | 'researchMode' | 'thinkingPattern'> | null | undefined,
): string {
  if (!tab) return '';
  if (tab.thinkingMode !== undefined) return tab.thinkingMode;
  if (tab.researchMode) return RESEARCH_MODE;
  return tab.thinkingPattern ?? '';
}

/** Tab patch for a mode change — keeps the legacy fields in lockstep. */
export function thinkingModeTabPatch(mode: string): Partial<ConversationTab> {
  return {
    thinkingMode: mode,
    researchMode: mode === RESEARCH_MODE,
    thinkingPattern: mode && mode !== RESEARCH_MODE ? mode : null,
  };
}

/** The stream-request fields the mode stands for (wire contract unchanged). */
export function thinkingModeWireFields(
  mode: string,
): { research_mode?: boolean; thinking_pattern?: string } {
  if (mode === RESEARCH_MODE) return { research_mode: true };
  if (mode) return { thinking_pattern: mode };
  return {};
}

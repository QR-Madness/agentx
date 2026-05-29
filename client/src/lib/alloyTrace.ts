/**
 * Alloy run tracing — group the flat conversation message list into discrete
 * multi-agent "runs" for the run-trace UI.
 *
 * Because specialists do not re-delegate (only the supervisor holds the
 * `delegate_to` tool), every delegation is a depth-1 fan-out from one
 * supervisor turn. A "run" is therefore a block of *consecutive* delegation
 * messages, attributed to the assistant (supervisor) turn that precedes it.
 *
 * Pure + read-only: works identically for live and restored conversations,
 * since both produce `DelegationMessage[]` in the same array.
 */

import type { ConversationMessage, DelegationMessage } from './messages';
import { isAssistantMessage, isDelegationMessage } from './messages';

export interface AlloyRunTotals {
  /** Number of delegations in the run. */
  count: number;
  tokensInput: number;
  tokensOutput: number;
  /** Summed cost across delegations sharing a single currency, else null. */
  costEstimate: number | null;
  /** The shared currency, or null when costs are mixed/missing. */
  costCurrency: string | null;
  /** True when delegations reported more than one currency (cost is partial). */
  costPartial: boolean;
  /** Wall-clock span of the run in ms (best-effort). */
  wallClockMs: number | null;
}

export interface AlloyRun {
  /** Stable key: the supervisor message id, else the first delegation id. */
  id: string;
  /** The assistant (supervisor) turn that owns this run, if found. */
  supervisorMessageId?: string;
  supervisorAgentName?: string;
  delegations: DelegationMessage[];
  startedAt: string;
  endedAt?: string;
  totals: AlloyRunTotals;
}

function computeTotals(delegations: DelegationMessage[]): AlloyRunTotals {
  let tokensInput = 0;
  let tokensOutput = 0;

  const currencies = new Set<string>();
  let costSum = 0;
  let sawCost = false;

  for (const d of delegations) {
    tokensInput += d.tokensInput ?? 0;
    tokensOutput += d.tokensOutput ?? 0;
    if (typeof d.costEstimate === 'number') {
      sawCost = true;
      costSum += d.costEstimate;
      if (d.costCurrency) currencies.add(d.costCurrency);
    }
  }

  const costPartial = currencies.size > 1;
  const costCurrency = currencies.size === 1 ? [...currencies][0] : null;
  const costEstimate = sawCost && !costPartial ? costSum : costPartial ? costSum : null;

  return {
    count: delegations.length,
    tokensInput,
    tokensOutput,
    costEstimate: sawCost ? costEstimate : null,
    costCurrency,
    costPartial,
    wallClockMs: computeWallClock(delegations),
  };
}

function computeWallClock(delegations: DelegationMessage[]): number | null {
  if (delegations.length === 0) return null;
  const first = delegations[0];
  const last = delegations[delegations.length - 1];
  // Prefer the real span between the first start and the last completion.
  if (last.completedAt && first.timestamp) {
    const span = Date.parse(last.completedAt) - Date.parse(first.timestamp);
    if (Number.isFinite(span) && span >= 0) return span;
  }
  // Restored runs lack completedAt — fall back to summed per-delegation durations.
  const summed = delegations.reduce((acc, d) => acc + (d.durationMs ?? 0), 0);
  return summed > 0 ? summed : null;
}

function finalizeRun(
  delegations: DelegationMessage[],
  supervisor: { id: string; agentName?: string } | null,
): AlloyRun {
  const first = delegations[0];
  const last = delegations[delegations.length - 1];
  return {
    id: supervisor?.id ?? first.delegationId,
    supervisorMessageId: supervisor?.id,
    supervisorAgentName: supervisor?.agentName,
    delegations,
    startedAt: first.timestamp,
    endedAt: last.completedAt ?? last.timestamp,
    totals: computeTotals(delegations),
  };
}

/**
 * Split a conversation transcript into Alloy runs (oldest first). Returns an
 * empty array when the conversation contains no delegations.
 */
export function groupRunsFromMessages(messages: ConversationMessage[]): AlloyRun[] {
  const runs: AlloyRun[] = [];
  let current: DelegationMessage[] = [];
  let lastAssistant: { id: string; agentName?: string } | null = null;

  const flush = () => {
    if (current.length > 0) {
      runs.push(finalizeRun(current, lastAssistant));
      current = [];
    }
  };

  for (const msg of messages) {
    if (isDelegationMessage(msg)) {
      current.push(msg);
      continue;
    }
    // A non-delegation message ends any run in progress.
    flush();
    if (isAssistantMessage(msg)) {
      lastAssistant = { id: msg.id, agentName: msg.agentName };
    }
  }
  flush();

  return runs;
}

/** The most recent run, or null when the transcript has no delegations. */
export function latestRun(messages: ConversationMessage[]): AlloyRun | null {
  const runs = groupRunsFromMessages(messages);
  return runs.length > 0 ? runs[runs.length - 1] : null;
}

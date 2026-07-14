/**
 * Alloy run tracing — group the flat conversation message list into discrete
 * multi-agent "runs" for the run-trace UI.
 *
 * A "run" is all the delegations of one user turn: background work orders
 * (`delegate_start`) interleave assistant prose, receipts, and report markers
 * between dispatch and completion by design, so only a *real user message*
 * (not a steer, not a folded report) closes a run. The supervisor is the
 * assistant turn nearest the run's first delegation.
 *
 * Pure + read-only: works identically for live and restored conversations,
 * since both produce `DelegationMessage[]` in the same array.
 */

import type { ConversationMessage, DelegationMessage } from './messages';
import { isAssistantMessage, isDelegationMessage, isUserMessage } from './messages';

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
  // The supervisor is snapshotted when the run's FIRST delegation is pushed —
  // assistant prose streamed after a background dispatch must not steal
  // attribution from the turn that issued the work orders.
  let supervisor: { id: string; agentName?: string } | null = null;

  const flush = () => {
    if (current.length > 0) {
      runs.push(finalizeRun(current, supervisor));
      current = [];
      supervisor = null;
    }
  };

  for (const msg of messages) {
    if (isDelegationMessage(msg)) {
      if (current.length === 0) supervisor = lastAssistant;
      current.push(msg);
      continue;
    }
    if (isAssistantMessage(msg)) {
      lastAssistant = { id: msg.id, agentName: msg.agentName };
      continue;
    }
    // Only a real user message closes a run — a steer or a folded work-order
    // report is mid-turn context, and tool cards/exhibits interleave freely.
    if (isUserMessage(msg) && !msg.steered) {
      flush();
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

/** One node of the work-order tree (the "filesystem" view of delegated work). */
export interface DelegationNode {
  delegation: DelegationMessage;
  children: DelegationNode[];
}

/**
 * Arrange a run's delegations into a tree by `parentDelegationId`. Everything
 * is a root today (nesting ships with the chain of command); orphaned parents
 * degrade to roots so a partial transcript still renders.
 */
export function buildDelegationTree(delegations: DelegationMessage[]): DelegationNode[] {
  const nodes = new Map<string, DelegationNode>();
  for (const d of delegations) {
    nodes.set(d.delegationId, { delegation: d, children: [] });
  }
  const roots: DelegationNode[] = [];
  for (const d of delegations) {
    const node = nodes.get(d.delegationId)!;
    const parent = d.parentDelegationId ? nodes.get(d.parentDelegationId) : undefined;
    if (parent && parent !== node) {
      parent.children.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}

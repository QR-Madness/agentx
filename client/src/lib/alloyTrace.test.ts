import { describe, it, expect } from 'vitest';
import { buildDelegationTree, groupRunsFromMessages, latestRun } from './alloyTrace';
import type {
  ConversationMessage,
  DelegationMessage,
  AssistantMessage,
  UserMessage,
} from './messages';

let seq = 0;
function delegation(p: Partial<DelegationMessage> = {}): DelegationMessage {
  seq += 1;
  return {
    id: `del_${seq}`,
    type: 'delegation',
    timestamp: '2026-05-28T00:00:00.000Z',
    delegationId: `dg_${seq}`,
    targetAgentId: 'spec',
    task: 'do work',
    depth: 1,
    status: 'completed',
    content: 'result',
    ...p,
  };
}

function assistant(id: string, p: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    id,
    type: 'assistant',
    timestamp: '2026-05-28T00:00:00.000Z',
    content: 'supervisor turn',
    ...p,
  };
}

function user(id: string, p: Partial<UserMessage> = {}): UserMessage {
  return { id, type: 'user', timestamp: '2026-05-28T00:00:00.000Z', content: 'hi', ...p };
}

describe('groupRunsFromMessages', () => {
  it('returns no runs when there are no delegations', () => {
    expect(groupRunsFromMessages([user('u1'), assistant('a1')])).toEqual([]);
  });

  it('groups consecutive delegations into one run under the preceding assistant', () => {
    const msgs: ConversationMessage[] = [
      user('u1'),
      assistant('a1', { agentName: 'Boss' }),
      delegation(),
      delegation(),
    ];
    const runs = groupRunsFromMessages(msgs);
    expect(runs).toHaveLength(1);
    expect(runs[0].delegations).toHaveLength(2);
    expect(runs[0].supervisorMessageId).toBe('a1');
    expect(runs[0].supervisorAgentName).toBe('Boss');
    expect(runs[0].id).toBe('a1');
  });

  it('splits runs on real user turns only (a run = one user turn of work)', () => {
    const msgs: ConversationMessage[] = [
      assistant('a1'),
      delegation(),
      user('u2'), // ends run 1
      assistant('a2'),
      delegation(),
    ];
    const runs = groupRunsFromMessages(msgs);
    expect(runs).toHaveLength(2);
    expect(runs[0].supervisorMessageId).toBe('a1');
    expect(runs[1].supervisorMessageId).toBe('a2');
  });

  it('does NOT split on interleaved assistant prose (background work orders)', () => {
    // delegate_start: assistant keeps talking while the order runs — one run.
    const msgs: ConversationMessage[] = [
      assistant('a1'),
      delegation(),
      assistant('a2'), // prose streamed while the work order runs
      delegation(),
    ];
    const runs = groupRunsFromMessages(msgs);
    expect(runs).toHaveLength(1);
    expect(runs[0].delegations).toHaveLength(2);
    // Supervisor snapshots at the FIRST delegation — later prose can't steal it.
    expect(runs[0].supervisorMessageId).toBe('a1');
  });

  it('does not split on steered user turns or report markers', () => {
    const msgs: ConversationMessage[] = [
      assistant('a1'),
      delegation(),
      user('u-steer', { steered: true }),
      {
        id: 'm1',
        type: 'work_order_report',
        timestamp: '2026-05-28T00:00:01.000Z',
        delegationId: 'dg_x',
        targetAgentId: 'spec',
        status: 'completed',
      },
      delegation(),
    ];
    const runs = groupRunsFromMessages(msgs);
    expect(runs).toHaveLength(1);
    expect(runs[0].delegations).toHaveLength(2);
  });

  it('sums tokens and cost across same-currency delegations', () => {
    const runs = groupRunsFromMessages([
      assistant('a1'),
      delegation({ tokensInput: 100, tokensOutput: 50, costEstimate: 0.01, costCurrency: 'USD' }),
      delegation({ tokensInput: 30, tokensOutput: 20, costEstimate: 0.02, costCurrency: 'USD' }),
    ]);
    const t = runs[0].totals;
    expect(t.count).toBe(2);
    expect(t.tokensInput).toBe(130);
    expect(t.tokensOutput).toBe(70);
    expect(t.costEstimate).toBeCloseTo(0.03, 6);
    expect(t.costCurrency).toBe('USD');
    expect(t.costPartial).toBe(false);
  });

  it('marks cost partial when currencies are mixed', () => {
    const runs = groupRunsFromMessages([
      assistant('a1'),
      delegation({ costEstimate: 0.01, costCurrency: 'USD' }),
      delegation({ costEstimate: 0.02, costCurrency: 'EUR' }),
    ]);
    expect(runs[0].totals.costPartial).toBe(true);
    expect(runs[0].totals.costCurrency).toBeNull();
  });

  it('falls back to summed durationMs for wall-clock when completedAt is absent', () => {
    const runs = groupRunsFromMessages([
      assistant('a1'),
      delegation({ durationMs: 1000 }),
      delegation({ durationMs: 1500 }),
    ]);
    expect(runs[0].totals.wallClockMs).toBe(2500);
  });

  it('prefers the real span between first start and last completion', () => {
    const runs = groupRunsFromMessages([
      assistant('a1'),
      delegation({ timestamp: '2026-05-28T00:00:00.000Z', durationMs: 100 }),
      delegation({ completedAt: '2026-05-28T00:00:10.000Z', durationMs: 100 }),
    ]);
    expect(runs[0].totals.wallClockMs).toBe(10_000);
  });
});

describe('latestRun', () => {
  it('returns the most recent run', () => {
    const run = latestRun([
      assistant('a1'),
      delegation(),
      user('u2'),
      assistant('a2'),
      delegation(),
    ]);
    expect(run?.supervisorMessageId).toBe('a2');
  });

  it('returns null with no delegations', () => {
    expect(latestRun([assistant('a1')])).toBeNull();
  });
});

describe('buildDelegationTree', () => {
  it('renders a flat list as all roots', () => {
    const tree = buildDelegationTree([
      delegation({ delegationId: 'a' }),
      delegation({ delegationId: 'b' }),
    ]);
    expect(tree.map(n => n.delegation.delegationId)).toEqual(['a', 'b']);
    expect(tree.every(n => n.children.length === 0)).toBe(true);
  });

  it('nests children under their parentDelegationId', () => {
    const tree = buildDelegationTree([
      delegation({ delegationId: 'root' }),
      delegation({ delegationId: 'child', parentDelegationId: 'root' }),
      delegation({ delegationId: 'grandchild', parentDelegationId: 'child' }),
    ]);
    expect(tree).toHaveLength(1);
    expect(tree[0].children[0].delegation.delegationId).toBe('child');
    expect(tree[0].children[0].children[0].delegation.delegationId).toBe('grandchild');
  });

  it('degrades orphaned parents to roots', () => {
    const tree = buildDelegationTree([
      delegation({ delegationId: 'x', parentDelegationId: 'missing' }),
    ]);
    expect(tree).toHaveLength(1);
    expect(tree[0].delegation.delegationId).toBe('x');
  });
});

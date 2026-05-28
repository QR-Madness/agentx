import { describe, it, expect } from 'vitest';
import { orphanedRuns } from './orphanedRuns';
import type { ActiveChatRun } from '../../lib/api';
import type { ConversationTab } from '../../lib/storage';

const run = (over: Partial<ActiveChatRun>): ActiveChatRun => ({
  run_id: 'r1',
  status: 'running',
  message: 'm',
  session_id: null,
  created_at: 't',
  updated_at: 't',
  ...over,
});

const tab = (over: Partial<ConversationTab>): ConversationTab => ({
  id: 't1',
  title: 'T',
  sessionId: null,
  profileId: null,
  workflowId: null,
  messages: [],
  isStreaming: false,
  createdAt: 't',
  lastMessageAt: 't',
  ...over,
});

describe('orphanedRuns', () => {
  it('keeps running runs not owned by an open tab', () => {
    const runs = [run({ run_id: 'a' }), run({ run_id: 'b' })];
    const tabs = [tab({ id: 'x', activeRun: { runId: 'a' } })];
    expect(orphanedRuns(runs, tabs).map(r => r.run_id)).toEqual(['b']);
  });

  it('drops non-running runs', () => {
    const runs = [run({ run_id: 'a', status: 'done' }), run({ run_id: 'b', status: 'running' })];
    expect(orphanedRuns(runs, []).map(r => r.run_id)).toEqual(['b']);
  });

  it('returns empty when every run is owned', () => {
    const runs = [run({ run_id: 'a' })];
    const tabs = [tab({ activeRun: { runId: 'a' } })];
    expect(orphanedRuns(runs, tabs)).toEqual([]);
  });
});

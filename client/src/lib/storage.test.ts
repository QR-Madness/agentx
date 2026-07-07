import { describe, it, expect, beforeEach } from 'vitest';
import { getConversationTabs } from './storage';

describe('getConversationTabs backfill', () => {
  beforeEach(() => localStorage.clear());

  it('defaults flags added after a tab was persisted (legacy tabs)', () => {
    // A tab persisted before noMemorization/noDelegation/modelOverride existed.
    localStorage.setItem(
      'agentx:server:s1:tabs',
      JSON.stringify([{
        id: 'tab_legacy', title: 'Old', sessionId: null, profileId: null,
        messages: [], isStreaming: false,
        createdAt: '2026-01-01T00:00:00Z', lastMessageAt: '2026-01-01T00:00:00Z',
      }]),
    );
    const [tab] = getConversationTabs('s1');
    expect(tab.workflowId).toBeNull();
    expect(tab.noMemorization).toBe(false);
    expect(tab.noDelegation).toBe(false);
    expect(tab.modelOverride).toBeNull();
  });
});

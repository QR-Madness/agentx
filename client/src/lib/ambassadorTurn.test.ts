import { describe, it, expect } from 'vitest';
import { gatherTurnContext, resolveTurnAgentName } from './ambassadorTurn';
import type { AssistantMessage, ConversationMessage } from './messages';

const ts = '2026-06-05T00:00:00Z';

function user(id: string, content: string): ConversationMessage {
  return { id, type: 'user', timestamp: ts, content };
}
function assistant(id: string, extra: Partial<AssistantMessage> = {}): ConversationMessage {
  return { id, type: 'assistant', timestamp: ts, content: 'reply', ...extra };
}
function toolCall(
  id: string,
  toolName: string,
  args: Record<string, unknown>,
  result?: string,
): ConversationMessage {
  return {
    id,
    type: 'tool_call',
    timestamp: ts,
    toolName,
    toolCallId: `${id}_call`,
    arguments: args,
    status: 'completed',
    result: result ? { content: result, success: true } : undefined,
  };
}
function citationExhibit(id: string, sources: Array<{ label: string; url?: string }>): ConversationMessage {
  return {
    id,
    type: 'exhibit',
    timestamp: ts,
    exhibit: {
      schemaVersion: 1,
      id: `ex_${id}`,
      layout: 'stack',
      elements: [{ type: 'citation', sources: sources.map((s) => ({ ...s, kind: 'passive' as const })) }],
    },
  };
}

describe('resolveTurnAgentName', () => {
  it('prefers the stamped agentName', () => {
    expect(resolveTurnAgentName({ agentName: 'Atlas', profileId: 'p1' }, { fallback: 'X' })).toBe('Atlas');
  });

  it('falls back to the profile name by profileId (restored turns lack agentName)', () => {
    const name = resolveTurnAgentName(
      { agentName: undefined, profileId: 'p1' },
      { nameByProfileId: (id) => (id === 'p1' ? 'Mobius' : undefined), fallback: 'X' },
    );
    expect(name).toBe('Mobius');
  });

  it('falls back to the conversation name when nothing else resolves', () => {
    expect(resolveTurnAgentName({ agentName: '  ', profileId: undefined }, { fallback: 'AgentX' })).toBe('AgentX');
  });
});

describe('gatherTurnContext', () => {
  it('gathers the turn prompt + tools + sources between the user msg and the reply', () => {
    const messages: ConversationMessage[] = [
      user('u1', 'find the town business registry'),
      toolCall('t1', 'web_search', { query: 'town business registry' }, 'raw results blob'),
      citationExhibit('e1', [{ label: 'County Index', url: 'https://county.example/biz' }]),
      assistant('a1'),
    ];
    const { userText, artifacts } = gatherTurnContext(messages, 'a1');
    expect(userText).toBe('find the town business registry');
    expect(artifacts?.tools?.[0]).toMatchObject({ name: 'web_search', detail: 'town business registry', ok: true });
    expect(artifacts?.sources?.[0]).toEqual({ label: 'County Index', url: 'https://county.example/biz' });
  });

  it('returns no artifacts for a pure-prose turn', () => {
    const messages: ConversationMessage[] = [user('u1', 'hi'), assistant('a1')];
    expect(gatherTurnContext(messages, 'a1').artifacts).toBeUndefined();
  });

  it("does not bleed the previous turn's artifacts into this one", () => {
    const messages: ConversationMessage[] = [
      user('u1', 'first'),
      toolCall('t1', 'web_search', { query: 'first search' }),
      assistant('a1'),
      user('u2', 'second'),
      assistant('a2'),
    ];
    expect(gatherTurnContext(messages, 'a2').artifacts).toBeUndefined();
  });
});

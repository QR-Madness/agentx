import { describe, it, expect } from 'vitest';
import { mapServerMessages } from './mapServerMessages';
import type { ServerMessage } from '../../lib/api';
import type {
  AssistantMessage,
  ToolCallMessage,
  ToolResultMessage,
  DelegationMessage,
  ExhibitMessage,
} from '../../lib/messages';

function msg(partial: Partial<ServerMessage> & { role: ServerMessage['role'] }): ServerMessage {
  return {
    content: '',
    timestamp: '2026-05-27T00:00:00.000Z',
    turn_index: 0,
    ...partial,
  };
}

describe('mapServerMessages', () => {
  it('maps a user + assistant pair and carries assistant metadata', () => {
    const out = mapServerMessages([
      msg({ role: 'user', content: 'hello' }),
      msg({
        role: 'assistant',
        content: 'hi there',
        metadata: { model: 'claude-opus-4-7', tokens_input: 10, tokens_output: 5 },
      }),
    ]);

    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({ type: 'user', content: 'hello' });
    const assistant = out[1] as AssistantMessage;
    expect(assistant.type).toBe('assistant');
    expect(assistant.content).toBe('hi there');
    expect(assistant.model).toBe('claude-opus-4-7');
    expect(assistant.tokensInput).toBe(10);
    expect(assistant.tokensOutput).toBe(5);
  });

  it('restores a steered user turn and an interrupted assistant turn', () => {
    const out = mapServerMessages([
      msg({ role: 'user', content: 'also check Y', metadata: { steered: true } }),
      msg({ role: 'assistant', content: 'partial…', metadata: { model: 'm', interrupted: true } }),
    ]);
    expect(out[0]).toMatchObject({ type: 'user', content: 'also check Y', steered: true });
    expect((out[1] as AssistantMessage).interrupted).toBe(true);
  });

  it('leaves steered/interrupted falsy when metadata is absent', () => {
    const out = mapServerMessages([
      msg({ role: 'user', content: 'hi' }),
      msg({ role: 'assistant', content: 'yo', metadata: { model: 'm' } }),
    ]);
    expect((out[0] as { steered?: boolean }).steered).toBe(false);
    expect((out[1] as AssistantMessage).interrupted).toBe(false);
  });

  it('carries agent attribution (agent_name) onto the assistant message', () => {
    const out = mapServerMessages([
      msg({
        role: 'assistant',
        content: 'delegated answer',
        metadata: { model: 'm', agent_id: 'bold-cosmic-falcon', agent_name: 'Researcher' },
      }),
    ]);
    expect((out[0] as AssistantMessage).agentName).toBe('Researcher');
  });

  it('leaves agentName undefined when no attribution metadata is present', () => {
    const out = mapServerMessages([
      msg({ role: 'assistant', content: 'plain', metadata: { model: 'm' } }),
    ]);
    expect((out[0] as AssistantMessage).agentName).toBeUndefined();
  });

  it('collapses a tool_call + matching tool_result into one tool_call message', () => {
    const out = mapServerMessages([
      msg({ role: 'tool_call', content: '{"q":"x"}', metadata: { tool: 'search', tool_call_id: 'c1' } }),
      msg({
        role: 'tool_result',
        content: 'found it',
        metadata: { tool_call_id: 'c1', success: true, duration_ms: 42 },
      }),
    ]);

    expect(out).toHaveLength(1);
    const call = out[0] as ToolCallMessage;
    expect(call.type).toBe('tool_call');
    expect(call.toolName).toBe('search');
    expect(call.status).toBe('completed');
    expect(call.result).toMatchObject({ content: 'found it', success: true, durationMs: 42 });
  });

  it('marks a tool_call failed when its result reports success: false', () => {
    const out = mapServerMessages([
      msg({ role: 'tool_call', content: '{}', metadata: { tool: 'search', tool_call_id: 'c2' } }),
      msg({ role: 'tool_result', content: 'boom', metadata: { tool_call_id: 'c2', success: false } }),
    ]);
    expect(out).toHaveLength(1);
    expect((out[0] as ToolCallMessage).status).toBe('failed');
  });

  it('reconstructs a delegation message from a delegate_to pair', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"agent_id":"fallback","task":"fallback task"}',
        metadata: { tool: 'delegate_to', tool_call_id: 'd1' },
      }),
      msg({
        role: 'tool_result',
        content: 'specialist output',
        metadata: {
          tool_call_id: 'd1',
          success: true,
          delegation: { raw_content: 'full content', target_agent_id: 'bold-cosmic-falcon', task: 'do the thing' },
        },
      }),
    ]);

    expect(out).toHaveLength(1);
    const del = out[0] as DelegationMessage;
    expect(del.type).toBe('delegation');
    expect(del.targetAgentId).toBe('bold-cosmic-falcon');
    expect(del.task).toBe('do the thing');
    expect(del.status).toBe('completed');
    expect(del.content).toBe('full content');
  });

  it('restores per-delegation metrics from the delegation metadata blob', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"agent_id":"x","task":"t"}',
        metadata: { tool: 'delegate_to', tool_call_id: 'd9' },
      }),
      msg({
        role: 'tool_result',
        content: 'out',
        metadata: {
          tool_call_id: 'd9',
          success: true,
          delegation: {
            raw_content: 'out',
            target_agent_id: 'bold-cosmic-falcon',
            task: 't',
            tokens_input: 120,
            tokens_output: 60,
            duration_ms: 2500,
            cost_estimate: 0.0042,
            cost_currency: 'USD',
          },
        },
      }),
    ]);

    const del = out[0] as DelegationMessage;
    expect(del.tokensInput).toBe(120);
    expect(del.tokensOutput).toBe(60);
    expect(del.durationMs).toBe(2500);
    expect(del.costEstimate).toBe(0.0042);
    expect(del.costCurrency).toBe('USD');
  });

  it('keeps an orphan tool_result as a standalone card', () => {
    const out = mapServerMessages([
      msg({ role: 'tool_result', content: 'orphan', metadata: { tool: 'search', tool_call_id: 'no-match', success: true } }),
    ]);
    expect(out).toHaveLength(1);
    const res = out[0] as ToolResultMessage;
    expect(res.type).toBe('tool_result');
    expect(res.content).toBe('orphan');
  });

  it('restores a present_exhibit tool call as an exhibit message', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"id":"d1","title":"Login","elements":[{"type":"mermaid","content":"graph TD; A-->B"}]}',
        metadata: { tool: 'present_exhibit', tool_call_id: 'e1' },
      }),
      msg({
        role: 'tool_result',
        content: '{"success":true}',
        metadata: { tool_call_id: 'e1', success: true },
      }),
    ]);

    expect(out).toHaveLength(1); // tool_result is consumed, not shown
    const ex = out[0] as ExhibitMessage;
    expect(ex.type).toBe('exhibit');
    expect(ex.exhibit.id).toBe('d1');
    expect(ex.exhibit.title).toBe('Login');
    expect(ex.exhibit.layout).toBe('stack');
    expect(ex.exhibit.elements[0]).toMatchObject({ type: 'mermaid', content: 'graph TD; A-->B' });
  });

  it('restores a present_exhibit choice element', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"elements":[{"type":"choice","prompt":"Which DB?","options":["PostgreSQL","Neo4j"]}]}',
        metadata: { tool: 'present_exhibit', tool_call_id: 'c1' },
      }),
    ]);
    expect(out).toHaveLength(1);
    const ex = out[0] as ExhibitMessage;
    const el = ex.exhibit.elements[0];
    expect(el.type).toBe('choice');
    expect(el.type === 'choice' && el.options).toEqual(['PostgreSQL', 'Neo4j']);
    expect(el.type === 'choice' && el.prompt).toBe('Which DB?');
  });

  it('restores present_exhibit table and citation elements', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content:
          '{"elements":[{"type":"table","columns":["M","Cost"],"rows":[["opus","0.4"]]},{"type":"citation","sources":[{"label":"NLLB","kind":"active"}]}]}',
        metadata: { tool: 'present_exhibit', tool_call_id: 't1' },
      }),
    ]);
    const ex = out[0] as ExhibitMessage;
    expect(ex.exhibit.elements[0].type).toBe('table');
    expect(ex.exhibit.elements[1].type).toBe('citation');
  });

  it('amends an exhibit in place when the same id is re-presented', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"id":"keep","elements":[{"type":"mermaid","content":"graph TD; A-->B"}]}',
        metadata: { tool: 'present_exhibit', tool_call_id: 'e1' },
      }),
      msg({ role: 'assistant', content: 'let me revise that', metadata: { model: 'm' } }),
      msg({
        role: 'tool_call',
        content: '{"id":"keep","elements":[{"type":"mermaid","content":"graph TD; A-->C"}]}',
        metadata: { tool: 'present_exhibit', tool_call_id: 'e2' },
      }),
    ]);

    // One exhibit (amended in place) + the assistant message between them.
    const exhibits = out.filter((m) => m.type === 'exhibit') as ExhibitMessage[];
    expect(exhibits).toHaveLength(1);
    const el = exhibits[0].exhibit.elements[0];
    expect(el.type === 'mermaid' && el.content).toBe('graph TD; A-->C');
  });

  it('filters out roles it does not render', () => {
    const out = mapServerMessages([
      msg({ role: 'system', content: 'sys' }),
      msg({ role: 'user', content: 'hi' }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe('user');
  });

  it('restores a citation exhibit beneath a successful web_search turn', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"query":"nllb"}',
        metadata: { tool: 'web_search', tool_call_id: 'w1' },
      }),
      msg({
        role: 'tool_result',
        content: JSON.stringify({
          success: true,
          results: [
            { title: 'NLLB paper', url: 'https://a' },
            { title: 'HF NLLB', url: 'https://b' },
          ],
        }),
        metadata: { tool: 'web_search', tool_call_id: 'w1', success: true },
      }),
    ]);
    // The web_search tool card, then the derived citation exhibit.
    const toolCall = out.find((m) => m.type === 'tool_call') as ToolCallMessage;
    expect(toolCall.toolName).toBe('web_search');
    const exhibits = out.filter((m) => m.type === 'exhibit') as ExhibitMessage[];
    expect(exhibits).toHaveLength(1);
    expect(exhibits[0].exhibit.id).toBe('exh_src_w1');
    const el = exhibits[0].exhibit.elements[0];
    expect(el.type === 'citation' && el.sources).toHaveLength(2);
    // ordering: tool card precedes its citation
    expect(out.indexOf(toolCall)).toBeLessThan(out.indexOf(exhibits[0]));
  });

  it('does not synthesize a citation for a failed web_search', () => {
    const out = mapServerMessages([
      msg({
        role: 'tool_call',
        content: '{"query":"x"}',
        metadata: { tool: 'web_search', tool_call_id: 'w2' },
      }),
      msg({
        role: 'tool_result',
        content: JSON.stringify({ success: false, results: [] }),
        metadata: { tool: 'web_search', tool_call_id: 'w2', success: false },
      }),
    ]);
    expect(out.filter((m) => m.type === 'exhibit')).toHaveLength(0);
  });
});

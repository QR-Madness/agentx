import { describe, it, expect } from 'vitest';
import { mapServerMessages } from './mapServerMessages';
import type { ServerMessage } from '../../lib/api';
import type {
  AssistantMessage,
  ToolCallMessage,
  ToolResultMessage,
  DelegationMessage,
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

  it('filters out roles it does not render', () => {
    const out = mapServerMessages([
      msg({ role: 'system', content: 'sys' }),
      msg({ role: 'user', content: 'hi' }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe('user');
  });
});

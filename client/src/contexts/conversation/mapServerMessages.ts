/**
 * Map persisted server messages into frontend ConversationMessage types.
 *
 * Pairs adjacent tool_call + tool_result rows (matched by tool_call_id) into a
 * single ToolCallMessage, and reconstructs a DelegationMessage for delegate_to
 * pairs so workflow conversations restore as the user saw them live (single
 * delegation card with full specialist output).
 */

import type { ConversationMessage } from '../../lib/messages';
import { createMessageId } from '../../lib/messages';
import type { ServerMessage } from '../../lib/api';

function safeParseJson(str: string): Record<string, unknown> {
  try { return JSON.parse(str); } catch { return {}; }
}

export function mapServerMessages(messages: ServerMessage[]): ConversationMessage[] {
  const filtered = messages.filter(m =>
    ['user', 'assistant', 'tool_call', 'tool_result'].includes(m.role)
  );

  // Index tool_result rows by tool_call_id so we can pair them with their
  // tool_call and skip them when iterating.
  const resultsByCallId = new Map<string, ServerMessage>();
  for (const m of filtered) {
    if (m.role === 'tool_result') {
      const id = m.metadata?.tool_call_id as string | undefined;
      if (id) resultsByCallId.set(id, m);
    }
  }
  const consumedResults = new Set<string>();

  const out: ConversationMessage[] = [];
  for (const m of filtered) {
    const base = {
      id: createMessageId(),
      timestamp: m.timestamp || new Date().toISOString(),
    };

    if (m.role === 'user') {
      out.push({ ...base, type: 'user', content: m.content });
      continue;
    }

    if (m.role === 'assistant') {
      // Render whatever was persisted, even if empty — silently dropping
      // legacy rows hides messages from conversations stored before the
      // write-side empty-skip landed. New empty rows are blocked at
      // _store_turns() time.
      out.push({
        ...base,
        type: 'assistant',
        content: m.content || '',
        model: m.metadata?.model as string | undefined,
        thinking: m.metadata?.thinking as string | undefined,
        tokensInput: m.metadata?.tokens_input as number | undefined,
        tokensOutput: m.metadata?.tokens_output as number | undefined,
        costEstimate: m.metadata?.cost_estimate as number | undefined,
        costCurrency: m.metadata?.cost_currency as string | undefined,
        latencyMs: m.metadata?.latency_ms as number | undefined,
      });
      continue;
    }

    if (m.role === 'tool_call') {
      const toolName = (m.metadata?.tool as string) || 'unknown';
      const toolCallId = (m.metadata?.tool_call_id as string) || base.id;
      const args = safeParseJson(m.content);
      const result = resultsByCallId.get(toolCallId);
      if (result) consumedResults.add(toolCallId);

      if (toolName === 'delegate_to') {
        const delegationMeta = (result?.metadata?.delegation ?? {}) as {
          raw_content?: string;
          target_agent_id?: string;
          task?: string;
        };
        const targetAgentId =
          delegationMeta.target_agent_id ||
          (args.agent_id as string | undefined) ||
          'unknown';
        const task = delegationMeta.task || (args.task as string | undefined) || '';
        const success = (result?.metadata?.success as boolean) ?? true;
        out.push({
          ...base,
          type: 'delegation',
          delegationId: toolCallId,
          targetAgentId,
          task,
          depth: 1,
          status: success ? 'completed' : 'failed',
          content: delegationMeta.raw_content || result?.content || '',
          resultPreview: result?.content,
        });
        continue;
      }

      out.push({
        ...base,
        type: 'tool_call',
        toolName,
        toolCallId,
        arguments: args,
        status: result
          ? ((result.metadata?.success as boolean) ?? true ? 'completed' : 'failed')
          : 'completed',
        result: result
          ? {
              content: result.content,
              success: (result.metadata?.success as boolean) ?? true,
              durationMs: result.metadata?.duration_ms as number | undefined,
            }
          : undefined,
      });
      continue;
    }

    if (m.role === 'tool_result') {
      const toolCallId = (m.metadata?.tool_call_id as string) || base.id;
      if (consumedResults.has(toolCallId)) continue;  // already merged into its tool_call
      // Orphan tool_result (no matching tool_call row) — keep as a standalone card.
      out.push({
        ...base,
        type: 'tool_result',
        toolName: (m.metadata?.tool as string) || 'unknown',
        toolCallId,
        content: m.content,
        success: (m.metadata?.success as boolean) ?? true,
        durationMs: m.metadata?.duration_ms as number | undefined,
      });
      continue;
    }
  }

  return out;
}

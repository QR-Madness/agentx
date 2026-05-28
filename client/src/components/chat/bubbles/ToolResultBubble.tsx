import { ToolExecutionBlock } from '../ToolExecutionBlock';
import type { BubbleProps } from './types';

/**
 * Renders a standalone `tool_result` row — an orphan result with no paired
 * `tool_call` in the persisted history (plan-execution turn storage can emit
 * these). Without a registered renderer these fell through to `UnknownBubble`
 * ("Unknown message type") when reopening a conversation. We reuse
 * `ToolExecutionBlock` so it reads identically to a completed tool call.
 */
export function ToolResultBubble({ message }: BubbleProps<'tool_result'>) {
  return (
    <div className="message-bubble tool_result">
      <ToolExecutionBlock
        toolName={message.toolName}
        toolCallId={message.toolCallId}
        arguments={{}}
        status={message.success ? 'completed' : 'failed'}
        result={{
          content: message.content,
          success: message.success,
          durationMs: message.durationMs,
        }}
      />
    </div>
  );
}

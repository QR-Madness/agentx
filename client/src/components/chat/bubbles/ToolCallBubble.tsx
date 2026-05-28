import { ToolExecutionBlock } from '../ToolExecutionBlock';
import type { BubbleProps } from './types';

export function ToolCallBubble({ message }: BubbleProps<'tool_call'>) {
  return (
    <div className="message-bubble tool_call">
      <ToolExecutionBlock
        toolName={message.toolName}
        toolCallId={message.toolCallId}
        arguments={message.arguments}
        status={message.status}
        result={message.result}
      />
    </div>
  );
}

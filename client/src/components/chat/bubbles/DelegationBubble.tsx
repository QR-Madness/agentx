import { AlertTriangle, CheckCircle2, Loader2, Workflow as WorkflowIcon, XCircle } from 'lucide-react';
import { MessageContent } from '../MessageContent';
import { ToolExecutionBlock } from '../ToolExecutionBlock';
import type { BubbleProps } from './types';

export function DelegationBubble({ message }: BubbleProps<'delegation'>) {
  const StatusIcon =
    message.status === 'streaming' ? Loader2 :
    message.status === 'completed' ? CheckCircle2 :
    XCircle;
  const statusClass = `delegation-status ${message.status}`;
  const targetName = message.targetAgentName || message.targetAgentId;

  return (
    <div className="message-bubble delegation">
      <div className="delegation-card">
        <div className="delegation-header">
          <WorkflowIcon size={14} />
          <span className="delegation-title">
            Delegated to <strong>{targetName}</strong>
          </span>
          {message.depth > 1 && (
            <span className="delegation-depth">depth {message.depth}</span>
          )}
          <span className={statusClass}>
            <StatusIcon
              size={12}
              className={message.status === 'streaming' ? 'spin' : ''}
            />
            <span>{message.status}</span>
          </span>
        </div>
        {message.task && <div className="delegation-task">{message.task}</div>}
        {message.toolEvents && message.toolEvents.length > 0 && (
          <div className="delegation-tool-events">
            {message.toolEvents.map(evt => (
              <ToolExecutionBlock
                key={evt.toolCallId}
                toolName={evt.toolName}
                toolCallId={evt.toolCallId}
                arguments={evt.arguments ?? {}}
                status={evt.status}
                result={evt.content !== undefined ? {
                  content: evt.content,
                  success: evt.success ?? true,
                  durationMs: evt.durationMs,
                } : undefined}
              />
            ))}
          </div>
        )}
        {message.content && (
          <div className="delegation-body">
            <MessageContent content={message.content} />
          </div>
        )}
        {message.error && (
          <div className="delegation-error">
            <AlertTriangle size={12} /> {message.error}
          </div>
        )}
      </div>
    </div>
  );
}

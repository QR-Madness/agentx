/**
 * MessageBubble — Unified message renderer for all conversation message types
 */

import { User, AlertTriangle, Info, Edit3, Workflow as WorkflowIcon, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { MessageContent } from './MessageContent';
import { ThinkingBubble } from './ThinkingBubble';
import { MessageActions } from './MessageActions';
import { MetadataBar } from './MetadataBar';
import { ToolExecutionBlock } from './ToolExecutionBlock';
import { MemoryInjectionBlock } from './MemoryInjectionBlock';
import { PlanExecutionBlock } from './PlanExecutionBlock';
import { getAvatarIcon } from '../../lib/avatars';
import {
  type ConversationMessage,
  isUserMessage,
  isAssistantMessage,
  isToolCallMessage,
  isMemoryInjectionMessage,
  isPlanExecutionMessage,
  isSystemMessage,
  isErrorMessage,
  isDelegationMessage,
} from '../../lib/messages';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: ConversationMessage;
  agentName?: string;
  avatarId?: string;
  onRegenerate?: () => void;
  onEdit?: (content: string) => void;
}

export function MessageBubble({ message, agentName, avatarId, onRegenerate, onEdit }: MessageBubbleProps) {
  // Route to specific message type renderer
  if (isUserMessage(message)) {
    return <UserBubble message={message} onEdit={onEdit} />;
  }

  if (isAssistantMessage(message)) {
    return <AssistantBubble message={message} agentName={agentName} avatarId={avatarId} onRegenerate={onRegenerate} />;
  }

  if (isToolCallMessage(message)) {
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

  if (isMemoryInjectionMessage(message)) {
    return (
      <div className="message-bubble memory_injection">
        <MemoryInjectionBlock
          facts={message.facts}
          entities={message.entities}
          relevantTurns={message.relevantTurns}
          queryUsed={message.queryUsed}
        />
      </div>
    );
  }

  if (isPlanExecutionMessage(message)) {
    return (
      <div className="message-bubble plan_execution">
        <PlanExecutionBlock
          planId={message.planId}
          task={message.task}
          complexity={message.complexity}
          subtaskCount={message.subtaskCount}
          status={message.status}
          subtasks={message.subtasks}
          totalTimeMs={message.totalTimeMs}
          completedCount={message.completedCount}
        />
      </div>
    );
  }

  if (isDelegationMessage(message)) {
    return <DelegationBubble message={message} />;
  }

  if (isSystemMessage(message)) {
    return <SystemBubble message={message} />;
  }

  if (isErrorMessage(message)) {
    return <ErrorBubble message={message} />;
  }

  // Fallback for unknown types
  return (
    <div className="message-bubble unknown">
      <div className="message-avatar">
        <Info size={16} />
      </div>
      <div className="message-body">
        <div className="message-text">Unknown message type</div>
      </div>
    </div>
  );
}

// ============================================================================
// User Message
// ============================================================================

interface UserBubbleProps {
  message: Extract<ConversationMessage, { type: 'user' }>;
  onEdit?: (content: string) => void;
}

function UserBubble({ message, onEdit }: UserBubbleProps) {
  return (
    <div className="message-bubble user">
      <div className="message-avatar user-avatar">
        <User size={16} />
      </div>
      <div className="message-body">
        <div className="message-text">{message.content}</div>
        <div className="message-meta">
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {message.editedAt && (
            <span className="message-edited">(edited)</span>
          )}
        </div>
        {onEdit && (
          <button className="edit-button" onClick={() => onEdit(message.content)} title="Edit message">
            <Edit3 size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Assistant Message
// ============================================================================

interface AssistantBubbleProps {
  message: Extract<ConversationMessage, { type: 'assistant' }>;
  agentName?: string;
  avatarId?: string;
  onRegenerate?: () => void;
}

function AssistantBubble({ message, agentName, avatarId, onRegenerate }: AssistantBubbleProps) {
  const displayName = message.agentName || agentName || 'Assistant';
  const AvatarIcon = getAvatarIcon(avatarId);

  return (
    <div className="message-bubble assistant">
      <div className="message-avatar assistant-avatar">
        <AvatarIcon size={16} />
      </div>
      <div className="message-body">
        {/* Agent name header */}
        <div className="assistant-header">
          <span className="assistant-name">{displayName}</span>
        </div>

        {/* Thinking bubble */}
        {message.thinking && (
          <ThinkingBubble thinking={message.thinking} defaultExpanded={true} />
        )}

        {/* Message content */}
        <MessageContent content={message.content} />

        {/* Metadata bar */}
        <MetadataBar
          model={message.model}
          tokensInput={message.tokensInput}
          tokensOutput={message.tokensOutput}
          tokensUsed={message.tokensUsed}
          latencyMs={message.latencyMs}
        />

        {/* Actions */}
        <MessageActions
          content={message.content}
          isAssistant={true}
          timestamp={new Date(message.timestamp)}
          onRegenerate={onRegenerate}
        />
      </div>
    </div>
  );
}

// ============================================================================
// System Message
// ============================================================================

interface SystemBubbleProps {
  message: Extract<ConversationMessage, { type: 'system' }>;
}

function SystemBubble({ message }: SystemBubbleProps) {
  return (
    <div className="message-bubble system">
      <div className="system-content">
        <Info size={14} />
        <span>{message.content}</span>
      </div>
    </div>
  );
}

// ============================================================================
// Error Message
// ============================================================================

interface ErrorBubbleProps {
  message: Extract<ConversationMessage, { type: 'error' }>;
}

function ErrorBubble({ message }: ErrorBubbleProps) {
  return (
    <div className="message-bubble error">
      <div className="error-content">
        <div className="error-header">
          <AlertTriangle size={16} />
          <span>Error</span>
          {message.recoverable && (
            <span className="recoverable-badge">Recoverable</span>
          )}
        </div>
        <div className="error-message">{message.content}</div>
      </div>
    </div>
  );
}

// ============================================================================
// Delegation Message (Agent Alloy specialist run)
// ============================================================================

interface DelegationBubbleProps {
  message: Extract<ConversationMessage, { type: 'delegation' }>;
}

function DelegationBubble({ message }: DelegationBubbleProps) {
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
        {message.task && (
          <div className="delegation-task">{message.task}</div>
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

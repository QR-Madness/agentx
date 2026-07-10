/**
 * StepGroup — collapsible wrapper around the messages produced inside a single
 * plan subtask. Header shows "Step k/n · title"; finished-plan steps fold away
 * by default so the transcript stays scannable when a plan conversation is
 * reopened.
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, ListChecks } from 'lucide-react';
import type { ConversationMessage, PlanStepRef } from '../../lib/messages';
import { MessageBubble } from './MessageBubble';
import { ToolRunGroup } from './ToolRunGroup';
import { foldToolRuns } from './groupMessagesBySteps';
import './StepGroup.css';

interface StepGroupProps {
  step: PlanStepRef;
  messages: ConversationMessage[];
  agentName?: string;
  avatarId?: string;
  /** Collapsed initially (plan no longer running). */
  defaultCollapsed?: boolean;
  onSubmitChoice?: (value: string, messageId: string) => void;
  busy?: boolean;
}

export function StepGroup({
  step,
  messages,
  agentName,
  avatarId,
  defaultCollapsed,
  onSubmitChoice,
  busy,
}: StepGroupProps) {
  const [collapsed, setCollapsed] = useState(!!defaultCollapsed);

  return (
    <div
      className="step-group"
      data-step-anchor={`${step.planId}:${step.subtaskId}`}
    >
      <button
        className="step-group-header"
        onClick={() => setCollapsed(c => !c)}
        aria-expanded={!collapsed}
      >
        {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
        <ListChecks size={13} className="step-group-icon" />
        <span className="step-group-label">
          Step {step.subtaskIndex}/{step.subtaskCount}
        </span>
        <span className="step-group-title" title={step.subtaskTitle}>
          {step.subtaskTitle}
        </span>
        {collapsed && (
          <span className="step-group-count">{messages.length}</span>
        )}
      </button>

      {!collapsed && (
        <div className="step-group-body">
          {foldToolRuns(messages).map(item =>
            item.kind === 'toolRun' ? (
              <div key={item.key} className="message-bubble tool_call">
                <ToolRunGroup messages={item.messages} />
              </div>
            ) : (
              <MessageBubble
                key={item.message.id}
                message={item.message}
                agentName={agentName}
                avatarId={avatarId}
                onSubmitChoice={onSubmitChoice}
                busy={busy}
              />
            ),
          )}
        </div>
      )}
    </div>
  );
}

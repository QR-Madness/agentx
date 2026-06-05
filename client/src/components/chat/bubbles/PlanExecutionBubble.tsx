import { PlanExecutionBlock } from '../PlanExecutionBlock';
import type { BubbleProps } from './types';

export function PlanExecutionBubble({ message, onResumePlan, busy }: BubbleProps<'plan_execution'>) {
  // An interrupted plan (persisted 'running' but no live stream) can be
  // resumed; suppress while another turn is streaming.
  const canResume = message.status === 'running' && !busy;
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
        onResume={onResumePlan}
        canResume={canResume}
      />
    </div>
  );
}

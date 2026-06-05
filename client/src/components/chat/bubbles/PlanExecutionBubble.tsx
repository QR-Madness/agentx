import { PlanExecutionBlock } from '../PlanExecutionBlock';
import type { BubbleProps } from './types';

export function PlanExecutionBubble({ message, busy }: BubbleProps<'plan_execution'>) {
  // Only animate the running spinner while this plan is actually streaming.
  const live = message.status === 'running' && !!busy;
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
        live={live}
      />
    </div>
  );
}

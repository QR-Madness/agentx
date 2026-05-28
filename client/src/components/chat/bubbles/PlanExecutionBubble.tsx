import { PlanExecutionBlock } from '../PlanExecutionBlock';
import type { BubbleProps } from './types';

export function PlanExecutionBubble({ message }: BubbleProps<'plan_execution'>) {
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

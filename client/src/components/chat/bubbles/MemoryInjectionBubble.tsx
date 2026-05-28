import { MemoryInjectionBlock } from '../MemoryInjectionBlock';
import type { BubbleProps } from './types';

export function MemoryInjectionBubble({ message }: BubbleProps<'memory_injection'>) {
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

import { Info } from 'lucide-react';
import type { BubbleProps } from './types';

export function SystemBubble({ message }: BubbleProps<'system'>) {
  return (
    <div className="message-bubble system">
      <div className="system-content">
        <Info size={14} />
        <span>{message.content}</span>
      </div>
    </div>
  );
}

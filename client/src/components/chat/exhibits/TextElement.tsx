/**
 * TextElement — a markdown passage inside an exhibit, rendered through the SAME
 * pipeline as chat messages (`MessageContent`: react-markdown + GFM + math — never
 * raw HTML), so exhibit prose can't do anything a chat message couldn't.
 */

import { MessageContent } from '../MessageContent';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

function TextElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'text') return null;
  const { content, title } = element;
  return (
    <div className="flex flex-col gap-1.5">
      {title && <div className="text-sm font-medium text-fg">{title}</div>}
      <MessageContent content={content} className="text-sm" />
    </div>
  );
}

export const TextElement = memoElement(TextElementImpl);

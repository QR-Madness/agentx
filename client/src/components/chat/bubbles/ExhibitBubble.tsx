/**
 * ExhibitBubble — renders one agent-presented Exhibit in a single bubble.
 *
 * Iterates the exhibit's elements (in `layout` order; `stack` = vertical) and
 * renders each via the element registry. Unknown element types degrade to a
 * safe source-as-code fallback so a newer backend can't break the transcript.
 */

import { elementRegistry } from '../exhibits/elementRegistry';
import type { BubbleProps } from './types';

export function ExhibitBubble({ message }: BubbleProps<'exhibit'>) {
  const { exhibit } = message;
  return (
    <div className="message-bubble exhibit">
      <div className="flex flex-col gap-2 rounded-lg border border-line bg-surface-raised p-3">
        {exhibit.title && (
          <div className="text-sm font-medium text-fg">{exhibit.title}</div>
        )}
        <div className="flex flex-col gap-3">
          {exhibit.elements.map((el, i) => {
            const Renderer = elementRegistry[el.type];
            if (Renderer) {
              return <Renderer key={i} content={el.content} title={el.title} />;
            }
            return (
              <div
                key={i}
                className="flex flex-col gap-1.5 rounded-md border border-line bg-surface-sunken p-3"
              >
                <span className="text-xs font-medium text-fg-muted">
                  Unsupported element: {el.type}
                </span>
                <pre className="overflow-x-auto text-xs text-fg-muted">
                  <code>{el.content}</code>
                </pre>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

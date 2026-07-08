/**
 * ExhibitBubble — renders one agent-presented Exhibit in a single bubble.
 *
 * Iterates the exhibit's elements (in `layout` order; `stack` = vertical) and
 * renders each via the element registry. Unknown element types degrade to a
 * safe source-as-code fallback so a newer backend can't break the transcript.
 */

import { elementRegistry } from '../exhibits/elementRegistry';
import type { BubbleProps } from './types';

export function ExhibitBubble({ message, busy, onSubmitChoice }: BubbleProps<'exhibit'>) {
  const { exhibit } = message;
  // Citation-only exhibits (e.g. auto-captured web_search sources) render as a
  // slim collapsed row directly in the flow — no card chrome — so they don't
  // stack as disruptive full-width cards. Everything else keeps its card.
  const citationOnly =
    exhibit.elements.length > 0 && exhibit.elements.every((el) => el.type === 'citation');
  return (
    <div className="message-bubble exhibit">
      <div
        className={
          citationOnly
            ? 'flex flex-col gap-2'
            : 'flex flex-col gap-2 rounded-lg border border-line bg-surface-raised p-3'
        }
      >
        {!citationOnly && exhibit.title && (
          <div className="text-sm font-medium text-fg">{exhibit.title}</div>
        )}
        <div className="flex flex-col gap-3">
          {exhibit.elements.map((el, i) => {
            const Renderer = elementRegistry[el.type];
            if (Renderer) {
              return (
                <Renderer
                  key={i}
                  element={el}
                  messageId={message.id}
                  answeredValue={message.answeredValue}
                  busy={busy}
                  onSubmitChoice={onSubmitChoice}
                  containerTitle={citationOnly ? exhibit.title : undefined}
                />
              );
            }
            // Unknown element type → safe source-as-code fallback, never raw HTML.
            const raw = 'content' in el ? el.content : JSON.stringify(el, null, 2);
            return (
              <div
                key={i}
                className="flex flex-col gap-1.5 rounded-md border border-line bg-surface-sunken p-3"
              >
                <span className="text-xs font-medium text-fg-muted">
                  Unsupported element: {el.type}
                </span>
                <pre className="overflow-x-auto text-xs text-fg-muted">
                  <code>{raw}</code>
                </pre>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

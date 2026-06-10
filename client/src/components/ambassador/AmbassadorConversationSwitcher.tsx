/**
 * AmbassadorConversationSwitcher — pick which conversation the ambassador is focused
 * on, independent of the chat tab. The ambassador's focus "stays put" (it doesn't
 * follow the active tab); this is how you move it. A "Current conversation" shortcut
 * jumps it to the chat tab you're in. Read-only over `useConversationList().items`.
 */

import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Check, CornerUpLeft, MessagesSquare } from 'lucide-react';

export interface SwitcherItem {
  id: string;
  title: string;
}

interface Props {
  items: SwitcherItem[];
  focusedId?: string;
  activeId?: string;
  onSelect: (id: string) => void;
}

export function AmbassadorConversationSwitcher({ items, focusedId, activeId, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const focused = items.find((it) => it.id === focusedId);
  const label = focused?.title ?? 'this conversation';
  const onActive = !!focusedId && focusedId === activeId;

  const pick = (id: string) => {
    onSelect(id);
    setOpen(false);
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        data-on={open || undefined}
        className="inline-flex max-w-[220px] items-center gap-1 rounded-md border border-line bg-surface-raised px-2 py-1 text-xs text-fg-secondary transition-colors hover:text-fg data-[on=true]:border-line-strong"
        title="Switch which conversation the ambassador is on"
      >
        <MessagesSquare size={12} className="shrink-0 text-fg-muted" />
        <span className="truncate">{label}</span>
        <ChevronDown size={13} className="shrink-0" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 max-h-72 w-64 max-w-[calc(100vw-2rem)] overflow-y-auto rounded-lg border border-line bg-surface-overlay p-1 shadow-lg">
          {!onActive && activeId && (
            <button
              type="button"
              onClick={() => pick(activeId)}
              className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs font-medium text-accent transition-colors hover:bg-surface-hover"
            >
              <CornerUpLeft size={13} className="shrink-0" /> Current conversation
            </button>
          )}
          {items.length === 0 ? (
            <p className="px-2 py-2 text-xs text-fg-muted">No conversations yet.</p>
          ) : (
            items.map((it) => (
              <button
                key={it.id}
                type="button"
                onClick={() => pick(it.id)}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-fg-secondary transition-colors hover:bg-surface-hover"
              >
                <span className="w-3 shrink-0">
                  {it.id === focusedId && <Check size={13} className="text-accent" />}
                </span>
                <span className="truncate">{it.title}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

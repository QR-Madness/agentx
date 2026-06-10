/**
 * AmbassadorConversationSwitcher — pick which conversation the ambassador is focused
 * on, independent of the chat tab. The ambassador's focus "stays put" (it doesn't
 * follow the active tab); this is how you move it. A "Current conversation" shortcut
 * jumps it to the chat tab you're in. Read-only over `useConversationList().items`.
 */

import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Check, CornerUpLeft, MessagesSquare, Pencil } from 'lucide-react';

export interface SwitcherItem {
  id: string;
  title: string;
}

interface Props {
  items: SwitcherItem[];
  focusedId?: string;
  activeId?: string;
  onSelect: (id: string) => void;
  /** The Inquiry's own title (overrides the chat title in the label). */
  title?: string;
  /** Rename the Inquiry (empty clears → falls back to the chat title). */
  onRename?: (title: string) => void;
}

export function AmbassadorConversationSwitcher({
  items,
  focusedId,
  activeId,
  onSelect,
  title,
  onRename,
}: Props) {
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState('');
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  useEffect(() => {
    if (renaming) inputRef.current?.select();
  }, [renaming]);

  const focused = items.find((it) => it.id === focusedId);
  const chatTitle = focused?.title ?? 'this conversation';
  const label = title?.trim() || chatTitle;
  const onActive = !!focusedId && focusedId === activeId;

  const pick = (id: string) => {
    onSelect(id);
    setOpen(false);
  };

  const startRename = () => {
    setDraft(title?.trim() || '');
    setRenaming(true);
    setOpen(false);
  };
  const commitRename = () => {
    onRename?.(draft.trim());
    setRenaming(false);
  };

  if (renaming) {
    return (
      <div ref={ref} className="relative">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitRename();
            else if (e.key === 'Escape') setRenaming(false);
          }}
          onBlur={commitRename}
          placeholder={chatTitle}
          aria-label="Rename this Inquiry"
          className="w-[220px] max-w-[calc(100vw-2rem)] rounded-md border border-line-strong bg-surface-raised px-2 py-1 text-xs text-fg outline-none"
        />
      </div>
    );
  }

  return (
    <div ref={ref} className="relative inline-flex items-center gap-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        data-on={open || undefined}
        className="inline-flex max-w-[200px] items-center gap-1 rounded-md border border-line bg-surface-raised px-2 py-1 text-xs text-fg-secondary transition-colors hover:text-fg data-[on=true]:border-line-strong"
        title="Switch which conversation the ambassador is on"
      >
        <MessagesSquare size={12} className="shrink-0 text-fg-muted" />
        <span className="truncate">{label}</span>
        <ChevronDown size={13} className="shrink-0" />
      </button>
      {onRename && (
        <button
          type="button"
          onClick={startRename}
          className="shrink-0 rounded-md border border-line bg-surface-raised p-1 text-fg-muted transition-colors hover:text-fg"
          title="Rename this Inquiry"
          aria-label="Rename this Inquiry"
        >
          <Pencil size={12} />
        </button>
      )}
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

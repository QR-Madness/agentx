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
  /** `inline` = borderless ghost trigger (sits in the compact header bar). */
  variant?: 'bordered' | 'inline';
}

export function AmbassadorConversationSwitcher({
  items,
  focusedId,
  activeId,
  onSelect,
  title,
  onRename,
  variant = 'bordered',
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

  const triggerClass =
    variant === 'inline'
      ? 'inline-flex max-w-[200px] items-center gap-1 rounded-md px-1.5 py-0.5 text-sm font-medium text-fg transition-colors hover:bg-surface-hover data-[on=true]:bg-surface-hover'
      : 'inline-flex max-w-[200px] items-center gap-1 rounded-md border border-line bg-surface-raised px-2 py-1 text-xs text-fg-secondary transition-colors hover:text-fg data-[on=true]:border-line-strong';

  return (
    <div ref={ref} className="relative inline-flex items-center gap-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        data-on={open || undefined}
        className={triggerClass}
        title="Switch which conversation the ambassador is on"
      >
        {variant === 'bordered' && (
          <MessagesSquare size={12} className="shrink-0 text-fg-muted" />
        )}
        <span className="truncate">{label}</span>
        <ChevronDown
          size={13}
          className={`shrink-0 opacity-70 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
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
        <div className="absolute left-0 top-full z-30 mt-1.5 w-72 max-w-[calc(100vw-2rem)] origin-top-left overflow-hidden rounded-xl border border-line bg-surface-overlay shadow-lg backdrop-blur-sm animate-in fade-in-0 zoom-in-95 slide-in-from-top-1 duration-150">
          <div className="border-b border-line/60 px-3 py-2">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">
              Focus on a conversation
            </p>
          </div>
          <div className="max-h-72 overflow-y-auto p-1.5">
            {!onActive && activeId && (
              <>
                <button
                  type="button"
                  onClick={() => pick(activeId)}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm font-medium text-accent transition-colors hover:bg-accent/10"
                >
                  <CornerUpLeft size={14} className="shrink-0" /> Jump to current conversation
                </button>
                {items.length > 0 && <div className="my-1 h-px bg-line/60" />}
              </>
            )}
            {items.length === 0 ? (
              <p className="px-2.5 py-4 text-center text-xs text-fg-muted">No conversations open yet.</p>
            ) : (
              items.map((it) => (
                <button
                  key={it.id}
                  type="button"
                  onClick={() => pick(it.id)}
                  data-on={it.id === focusedId || undefined}
                  className="group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors hover:bg-surface-hover data-[on=true]:bg-accent/10"
                >
                  <MessagesSquare
                    size={14}
                    className="shrink-0 text-fg-muted transition-colors group-data-[on=true]:text-accent"
                  />
                  <span className="flex-1 truncate text-fg-secondary group-hover:text-fg group-data-[on=true]:font-medium group-data-[on=true]:text-fg">
                    {it.title}
                  </span>
                  {it.id === focusedId && <Check size={14} className="shrink-0 text-accent" />}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

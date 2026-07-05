/**
 * AmbassadorConversationSwitcher — pick which conversation the ambassador is focused
 * on, independent of the chat tab. The ambassador's focus "stays put" (it doesn't
 * follow the active tab); this is how you move it. A "Current conversation" shortcut
 * jumps it to the chat tab you're in.
 *
 * Built on the shared `DropdownMenu` primitive so it inherits the app's themed menu
 * surface/colors and portals out of the panel's `overflow-hidden` (no clipping).
 */

import { ChevronDown, Check, CornerUpLeft, MessagesSquare } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
  FieldTrigger,
} from '../ui';

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
  /** `inline` = borderless ghost trigger (sits in the compact header bar). */
  variant?: 'bordered' | 'inline';
}

export function AmbassadorConversationSwitcher({
  items,
  focusedId,
  activeId,
  onSelect,
  title,
  variant = 'bordered',
}: Props) {
  const focused = items.find((it) => it.id === focusedId);
  const label = title?.trim() || focused?.title || 'this conversation';
  const onActive = !!focusedId && focusedId === activeId;

  // Both variants get real field chrome (`ax-trigger`) so the picker reads as a
  // control at rest — the old ghost/washed-out triggers only appeared on hover.
  const triggerClass =
    variant === 'inline' ? 'group text-sm font-medium' : 'group max-w-[200px] text-xs';

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <FieldTrigger
          className={triggerClass}
          title="Switch which conversation the ambassador is on"
        >
          {variant === 'bordered' && (
            <MessagesSquare size={12} className="shrink-0 text-fg-muted" />
          )}
          <span className="truncate">{label}</span>
          <ChevronDown
            size={13}
            className="shrink-0 opacity-70 transition-transform duration-200 group-data-[state=open]:rotate-180"
          />
        </FieldTrigger>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64 max-w-[calc(100vw-2rem)]">
        <DropdownMenuLabel>Focus on a conversation</DropdownMenuLabel>
        {!onActive && activeId && (
          <DropdownMenuItem onSelect={() => onSelect(activeId)} className="text-accent">
            <CornerUpLeft size={14} className="shrink-0" />
            Jump to current conversation
          </DropdownMenuItem>
        )}
        {items.length === 0 ? (
          <p className="px-3 py-2 text-xs text-fg-muted">No conversations open yet.</p>
        ) : (
          items.map((it) => (
            <DropdownMenuItem
              key={it.id}
              onSelect={() => onSelect(it.id)}
              data-on={it.id === focusedId || undefined}
              className="data-[on=true]:text-accent"
            >
              <MessagesSquare size={14} className="shrink-0 opacity-70" />
              <span className="flex-1 truncate">{it.title}</span>
              {it.id === focusedId && <Check size={14} className="shrink-0 text-accent" />}
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

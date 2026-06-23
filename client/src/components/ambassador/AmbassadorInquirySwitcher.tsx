/**
 * AmbassadorInquirySwitcher — pick / create which standalone Inquiry the command deck is
 * showing. The deck holds several named Inquiries (the home deck thread, pinned, plus minted
 * ones); this is how you switch between them or start a new one. Sibling of
 * AmbassadorConversationSwitcher (same themed `DropdownMenu` primitive), but Inquiry-flavored.
 */

import { ChevronDown, Check, Plus, Radio, Pin } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui';
import { type DeckInquiry, orderInquiries, inquiryLabel } from '../../lib/ambassadorDeck';

interface Props {
  inquiries: DeckInquiry[];
  selectedId: string;
  deckThreadId: string;
  onSelect: (threadId: string) => void;
  onNew: () => void;
}

export function AmbassadorInquirySwitcher({
  inquiries,
  selectedId,
  deckThreadId,
  onSelect,
  onNew,
}: Props) {
  const ordered = orderInquiries(inquiries, deckThreadId);
  const selected = ordered.find((i) => i.thread_id === selectedId);
  const label = selected
    ? inquiryLabel(selected, deckThreadId)
    : inquiryLabel({ thread_id: selectedId, title: '' }, deckThreadId);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="group flex min-w-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-sm font-medium text-fg transition-colors hover:bg-surface-hover data-[state=open]:bg-surface-hover"
          title="Switch or create an Inquiry"
        >
          <span className="truncate">{label}</span>
          <ChevronDown
            size={13}
            className="shrink-0 opacity-70 transition-transform duration-200 group-data-[state=open]:rotate-180"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64 max-w-[calc(100vw-2rem)]">
        <DropdownMenuLabel>Your Inquiries</DropdownMenuLabel>
        {ordered.map((inq) => {
          const isHome = inq.thread_id === deckThreadId;
          return (
            <DropdownMenuItem
              key={inq.thread_id}
              onSelect={() => onSelect(inq.thread_id)}
              data-on={inq.thread_id === selectedId || undefined}
              className="data-[on=true]:text-accent"
            >
              {isHome ? (
                <Pin size={14} className="shrink-0 opacity-70" />
              ) : (
                <Radio size={14} className="shrink-0 opacity-70" />
              )}
              <span className="flex-1 truncate">{inquiryLabel(inq, deckThreadId)}</span>
              {inq.thread_id === selectedId && <Check size={14} className="shrink-0 text-accent" />}
            </DropdownMenuItem>
          );
        })}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onNew} className="text-accent">
          <Plus size={14} className="shrink-0" />
          New Inquiry
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

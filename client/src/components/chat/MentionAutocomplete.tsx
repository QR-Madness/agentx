/**
 * MentionAutocomplete — the @-mention picker shown in the chat composer.
 *
 * Presentational only: the active query, filtered items, and highlight index
 * live in ChatPanel (so the textarea keeps focus and drives keyboard nav).
 * Picking an agent inserts its `@<agent_id>` slug, which the backend (16.5)
 * parses to route the turn.
 */

import { AtSign } from 'lucide-react';
import type { AgentProfile } from '../../lib/api/types';
import { getAvatarIcon } from '../../lib/avatars';
import { DropdownPortal } from '../ui/DropdownPortal';
import './MentionAutocomplete.css';

interface MentionAutocompleteProps {
  isOpen: boolean;
  items: AgentProfile[];
  highlight: number;
  anchorRef: React.RefObject<HTMLElement | null>;
  onHover: (index: number) => void;
  onPick: (profile: AgentProfile) => void;
  onClose: () => void;
}

export function MentionAutocomplete({
  isOpen,
  items,
  highlight,
  anchorRef,
  onHover,
  onPick,
  onClose,
}: MentionAutocompleteProps) {
  if (!isOpen || items.length === 0) return null;

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="top"
      align="start"
      estimatedHeight={260}
    >
      <div className="mention-ac" role="listbox" aria-label="Mention an agent">
        <div className="mention-ac-header">
          <AtSign size={12} />
          <span>Mention an agent</span>
        </div>
        {items.map((profile, i) => {
          const AvatarIcon = getAvatarIcon(profile.avatar);
          return (
            <div
              key={profile.id}
              role="option"
              aria-selected={i === highlight}
              className={`mention-ac-item ${i === highlight ? 'kb-active' : ''}`}
              // onMouseDown (not onClick) so the textarea doesn't blur first.
              onMouseDown={(e) => { e.preventDefault(); onPick(profile); }}
              onMouseEnter={() => onHover(i)}
            >
              <div className="mention-ac-avatar"><AvatarIcon size={14} /></div>
              <div className="mention-ac-info">
                <span className="mention-ac-name">{profile.name}</span>
                <span className="mention-ac-id">{profile.agentId}</span>
              </div>
              {profile.tags && profile.tags.length > 0 && (
                <span className="mention-ac-tags">
                  {profile.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="mention-ac-tag">{tag}</span>
                  ))}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </DropdownPortal>
  );
}

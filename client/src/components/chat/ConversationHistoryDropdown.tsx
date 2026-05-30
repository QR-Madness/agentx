/**
 * ConversationHistoryDropdown — Browse and switch between past conversations
 * Shows open tabs and server-side conversation history.
 * Renders via portal (DropdownPortal) to escape overflow constraints.
 *
 * Thin wrapper: the searchable list body lives in ConversationListContent,
 * shared with the mobile Conversations drawer (ConversationsDrawerContent).
 */

import { Clock, X } from 'lucide-react';
import { DropdownPortal } from '../ui/DropdownPortal';
import { ConversationListContent } from './ConversationListContent';
import './ConversationHistoryDropdown.css';

interface ConversationHistoryDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

export function ConversationHistoryDropdown({ isOpen, onClose, anchorRef }: ConversationHistoryDropdownProps) {
  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="bottom"
      align="end"
      estimatedHeight={420}
    >
      <div className="history-dropdown-portal">
        <div className="history-header">
          <Clock size={16} />
          <span>Conversation History</span>
          <button className="close-button" onClick={onClose}>
            <X size={14} />
          </button>
        </div>

        <ConversationListContent onActivated={onClose} />
      </div>
    </DropdownPortal>
  );
}

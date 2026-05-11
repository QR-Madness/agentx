/**
 * ActiveAgentsDropdown — Shows list of active/streaming conversations
 * Rendered as a portal dropdown anchored to the brain icon in TopBar
 */

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { MessageSquare, Plus, Radio } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import './ActiveAgentsDropdown.css';

interface ActiveAgentsDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

export function ActiveAgentsDropdown({ isOpen, onClose, anchorRef }: ActiveAgentsDropdownProps) {
  const { tabs, activeTabId, switchTab, addTab } = useConversation();
  const { activeProfile } = useAgentProfile();
  const [position, setPosition] = useState<{
    top?: number;
    bottom?: number;
    left: number;
  }>({ top: 0, left: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Calculate position based on anchor element. Flip above the anchor when
  // there isn't enough room below (e.g. mobile, where the TopBar sits at
  // the bottom of the viewport).
  useEffect(() => {
    if (!isOpen || !anchorRef.current) return;

    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (!rect) return;
      const estHeight = dropdownRef.current?.offsetHeight ?? 360;
      const flip = rect.bottom + estHeight + 8 > window.innerHeight;
      if (flip) {
        setPosition({
          bottom: Math.max(8, window.innerHeight - rect.top + 8),
          left: rect.left,
        });
      } else {
        setPosition({
          top: rect.bottom + 8,
          left: rect.left,
        });
      }
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    return () => window.removeEventListener('resize', updatePosition);
  }, [isOpen, anchorRef]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose, anchorRef]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const activeTabs = tabs.filter(t => t.isStreaming);
  const inactiveTabs = tabs.filter(t => !t.isStreaming);

  const handleSelect = (tabId: string) => {
    switchTab(tabId);
    onClose();
  };

  const handleNew = () => {
    addTab();
    onClose();
  };

  const AvatarIcon = getAvatarIcon(activeProfile?.avatar);

  const dropdown = (
    <div
      className="active-agents-dropdown"
      ref={dropdownRef}
      style={{
        top: position.top,
        bottom: position.bottom,
        left: position.left,
      }}
    >
      <button className="agents-new" onClick={handleNew}>
        <Plus size={14} />
        <span>New conversation</span>
      </button>

      {/* Active/streaming conversations */}
      {activeTabs.length > 0 && (
        <>
          <div className="agents-section-header">
            <Radio size={12} className="section-icon active" />
            <span>Active</span>
          </div>
          {activeTabs.map(tab => (
            <button
              key={tab.id}
              className={`agents-item ${tab.id === activeTabId ? 'current' : ''}`}
              onClick={() => handleSelect(tab.id)}
            >
              <div className="agents-item-avatar streaming">
                <AvatarIcon size={14} />
              </div>
              <div className="agents-item-info">
                <span className="agents-item-title">{tab.title}</span>
                <span className="agents-item-status">Streaming...</span>
              </div>
              <span className="agents-item-pulse" />
            </button>
          ))}
        </>
      )}

      {/* Inactive conversations */}
      {inactiveTabs.length > 0 && (
        <>
          <div className="agents-section-header">
            <MessageSquare size={12} className="section-icon" />
            <span>Conversations</span>
          </div>
          {inactiveTabs.map(tab => (
            <button
              key={tab.id}
              className={`agents-item ${tab.id === activeTabId ? 'current' : ''}`}
              onClick={() => handleSelect(tab.id)}
            >
              <div className="agents-item-avatar">
                <AvatarIcon size={14} />
              </div>
              <div className="agents-item-info">
                <span className="agents-item-title">{tab.title}</span>
                {tab.sessionId && (
                  <span className="agents-item-session">{tab.sessionId.slice(0, 8)}</span>
                )}
              </div>
            </button>
          ))}
        </>
      )}

      {tabs.length === 0 && (
        <div className="agents-empty">No conversations yet</div>
      )}
    </div>
  );

  return createPortal(dropdown, document.body);
}

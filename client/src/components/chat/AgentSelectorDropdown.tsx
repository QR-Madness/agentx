/**
 * AgentSelectorDropdown — Select and manage agent profiles
 * Renders via portal for proper z-index handling
 */

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Bot, Settings, Check, Plus, ChevronUp } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useModal } from '../../contexts/ModalContext';
import { getAvatarIcon } from '../../lib/avatars';
import './AgentSelectorDropdown.css';

interface AgentSelectorDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

export function AgentSelectorDropdown({ isOpen, onClose, anchorRef }: AgentSelectorDropdownProps) {
  const { profiles, activeProfile, setActiveProfile } = useAgentProfile();
  const { openModal } = useModal();
  const [position, setPosition] = useState({ bottom: 0, left: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Calculate position based on anchor element (opens upward)
  useEffect(() => {
    if (!isOpen || !anchorRef.current) return;

    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (rect) {
        setPosition({
          bottom: window.innerHeight - rect.top + 8,
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
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSelect = (profileId: string) => {
    setActiveProfile(profileId);
    onClose();
  };

  const handleEdit = (e: React.MouseEvent, profileId: string) => {
    e.stopPropagation();
    openModal({
      id: 'profile-editor',
      type: 'modal',
      component: 'profileEditor',
      size: 'md',
      props: { profileId },
    });
    onClose();
  };

  const handleCreateNew = () => {
    openModal({
      id: 'profile-editor',
      type: 'modal',
      component: 'profileEditor',
      size: 'md',
      props: { isNew: true },
    });
    onClose();
  };

  const dropdown = (
    <div
      className="agent-selector-dropdown"
      ref={dropdownRef}
      style={{ bottom: position.bottom, left: position.left }}
    >
      <div className="agent-selector-header">
        <Bot size={14} />
        <span>Select Agent</span>
        <ChevronUp size={14} className="header-icon" />
      </div>

      <div className="agent-selector-list">
        {profiles.map(profile => {
          const AvatarIcon = getAvatarIcon(profile.avatar);
          return (
            <div
              key={profile.id}
              className={`agent-selector-item ${profile.id === activeProfile?.id ? 'active' : ''}`}
              onClick={() => handleSelect(profile.id)}
            >
              <div className="agent-item-avatar">
                <AvatarIcon size={16} />
              </div>
              <div className="agent-item-info">
              <span className="agent-item-name">{profile.name}</span>
              <span className="agent-item-model">{profile.defaultModel || 'Default model'}</span>
            </div>
            {profile.id === activeProfile?.id && (
              <Check size={14} className="agent-item-check" />
            )}
            <button
              className="agent-item-edit"
              onClick={(e) => handleEdit(e, profile.id)}
              title="Edit profile"
            >
              <Settings size={12} />
            </button>
          </div>
          );
        })}
      </div>

      <div className="agent-selector-footer">
        <button className="agent-create-button" onClick={handleCreateNew}>
          <Plus size={14} />
          <span>Create New Agent</span>
        </button>
      </div>
    </div>
  );

  return createPortal(dropdown, document.body);
}

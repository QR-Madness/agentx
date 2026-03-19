/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 *
 * Brain icon (left) opens the active agents dropdown.
 * Lightning icon (right toolbar) opens the consolidation menu with live SSE progress.
 * Both pulse when their respective operations are active.
 */

import { useState, useRef } from 'react';
import {
  Sparkles,
  Home,
  LayoutDashboard,
  Bot,
  Settings,
  Database,
  Wrench,
  Languages,
  Brain,
  Zap,
} from 'lucide-react';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { useConsolidationStatus } from '../lib/hooks';
import { ActiveAgentsDropdown } from '../components/chat/ActiveAgentsDropdown';
import { ConsolidationMenu } from '../components/chat/ConsolidationMenu';
import { ConversationTabBar } from './ConversationTabBar';
import './TopBar.css';

export type PageId = 'start' | 'dashboard' | 'agentx';

interface TopBarProps {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
}

const NAV_ITEMS: { id: PageId; label: string; icon: React.ReactNode }[] = [
  { id: 'start', label: 'Start', icon: <Home size={16} /> },
  { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={16} /> },
  { id: 'agentx', label: 'AgentX', icon: <Bot size={16} /> },
];

export function TopBar({ activePage, onPageChange }: TopBarProps) {
  const { openModal } = useModal();
  const { tabs } = useConversation();
  const consolidation = useConsolidationStatus();

  const [showAgentsDropdown, setShowAgentsDropdown] = useState(false);
  const [showConsolidationMenu, setShowConsolidationMenu] = useState(false);

  const brainButtonRef = useRef<HTMLButtonElement>(null);
  const lightningButtonRef = useRef<HTMLButtonElement>(null);

  const hasStreamingTabs = tabs.some(t => t.isStreaming);

  const openSettings = () => {
    openModal({
      id: 'settings-drawer',
      type: 'drawer',
      component: 'settings',
      position: 'right',
      size: 'xxl',
    });
  };

  const openMemory = () => {
    openModal({
      id: 'memory-drawer',
      type: 'drawer',
      component: 'memory',
      position: 'right',
      size: 'xxl',
    });
  };

  const openTools = () => {
    openModal({
      id: 'tools-drawer',
      type: 'drawer',
      component: 'tools',
      position: 'right',
      size: 'md',
    });
  };

  const openTranslation = () => {
    openModal({
      id: 'translation-modal',
      type: 'modal',
      component: 'translation',
      size: 'lg',
    });
  };

  const openProfileEditor = () => {
    openModal({
      id: 'profile-editor',
      type: 'modal',
      component: 'profileEditor',
      size: 'md',
    });
  };

  return (
    <header className="top-bar">
      {/* Left: Brain icon (opens active agents) + Logo (edit profile) */}
      <div className="top-bar-left">
        <button
          ref={brainButtonRef}
          className={`toolbar-icon toolbar-icon-brain ${hasStreamingTabs ? 'pulsing' : ''} ${showAgentsDropdown ? 'active' : ''}`}
          onClick={() => setShowAgentsDropdown(prev => !prev)}
          title="Active conversations"
        >
          <Brain size={20} />
        </button>
        <ActiveAgentsDropdown
          isOpen={showAgentsDropdown}
          onClose={() => setShowAgentsDropdown(false)}
          anchorRef={brainButtonRef}
        />
        <button
          className="top-bar-logo"
          onClick={openProfileEditor}
          title="Edit agent profile"
        >
          <div className="logo-icon">
            <Sparkles size={20} />
          </div>
        </button>
      </div>

      {/* Center-left: Navigation pills */}
      <nav className="top-bar-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            className={`nav-pill ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onPageChange(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Center: Conversation tabs (visible only on AgentX page) */}
      <div className="top-bar-center">
        <ConversationTabBar visible={activePage === 'agentx'} />
      </div>

      {/* Right: Toolbar icons */}
      <div className="top-bar-right">
        {/* Consolidation lightning with dropdown menu */}
        <div className="consolidation-trigger-container">
          <button
            ref={lightningButtonRef}
            className={`toolbar-icon toolbar-icon-lightning ${consolidation.isActive ? 'pulsing' : ''}`}
            onClick={() => setShowConsolidationMenu(prev => !prev)}
            title="Memory consolidation"
          >
            <Zap size={18} />
          </button>

          <ConsolidationMenu
            isOpen={showConsolidationMenu}
            onClose={() => setShowConsolidationMenu(false)}
            anchorRef={lightningButtonRef}
            consolidation={consolidation}
          />
        </div>

        <button
          className="toolbar-icon"
          onClick={openTranslation}
          title="Translation"
        >
          <Languages size={18} />
        </button>
        <button
          className="toolbar-icon"
          onClick={openTools}
          title="Tools"
        >
          <Wrench size={18} />
        </button>
        <button
          className="toolbar-icon"
          onClick={openMemory}
          title="Memory"
        >
          <Database size={18} />
        </button>
        <button
          className="toolbar-icon"
          onClick={openSettings}
          title="Settings"
        >
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
}

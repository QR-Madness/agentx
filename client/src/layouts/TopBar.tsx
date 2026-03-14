/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 */

import {
  Sparkles,
  Home,
  LayoutDashboard,
  Bot,
  Settings,
  Database,
  Wrench,
  Languages,
} from 'lucide-react';
import { useModal } from '../contexts/ModalContext';
import { useAgentProfile } from '../contexts/AgentProfileContext';
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
  const { getAgentName } = useAgentProfile();

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
      {/* Left: Logo (clickable to edit profile) */}
      <div className="top-bar-left">
        <button
          className="top-bar-logo"
          onClick={openProfileEditor}
          title="Edit agent profile"
        >
          <div className="logo-icon">
            <Sparkles size={20} />
          </div>
          <span className="logo-text">{getAgentName()}</span>
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

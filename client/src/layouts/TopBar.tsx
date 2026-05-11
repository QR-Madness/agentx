/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 *
 * Brain icon (left) opens the active agents dropdown.
 * Lightning icon (right toolbar) opens the consolidation menu with live SSE progress.
 * Both pulse when their respective operations are active.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Home,
  LayoutDashboard,
  Bot,
  Settings,
  Database,
  Wrench,
  Languages,
  Brain,
  Zap,
  LogOut,
  KeyRound,
  MoreHorizontal,
  BrainCircuit,
} from 'lucide-react';
import { createPortal } from 'react-dom';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { useAuth } from '../contexts/AuthContext';
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
  const { authRequired, isAuthenticated, logout } = useAuth();

  const [showAgentsDropdown, setShowAgentsDropdown] = useState(false);
  const [showConsolidationMenu, setShowConsolidationMenu] = useState(false);
  const [showOverflow, setShowOverflow] = useState(false);
  const [overflowPos, setOverflowPos] = useState<{ top: number; right: number } | null>(null);

  const brainButtonRef = useRef<HTMLButtonElement>(null);
  const lightningButtonRef = useRef<HTMLButtonElement>(null);
  const overflowButtonRef = useRef<HTMLButtonElement>(null);

  const openOverflow = useCallback(() => {
    if (!overflowButtonRef.current) return;
    const rect = overflowButtonRef.current.getBoundingClientRect();
    const itemCount = 3 + (authRequired && isAuthenticated ? 2 : 0);
    const estHeight = itemCount * 44 + 16;
    const wouldOverflow = rect.bottom + estHeight + 8 > window.innerHeight;
    const top = wouldOverflow ? Math.max(8, rect.top - estHeight - 6) : rect.bottom + 4;
    setOverflowPos({ top, right: window.innerWidth - rect.right });
    setShowOverflow(true);
  }, [authRequired, isAuthenticated]);

  const closeOverflow = useCallback(() => {
    setShowOverflow(false);
    setOverflowPos(null);
  }, []);

  useEffect(() => {
    if (!showOverflow) return;
    let lastWidth = window.innerWidth;
    const onScroll = () => closeOverflow();
    const onResize = () => {
      const w = window.innerWidth;
      if (w !== lastWidth) { lastWidth = w; closeOverflow(); }
    };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [showOverflow, closeOverflow]);

  const hasStreamingTabs = tabs.some(t => t.isStreaming);

  const openSettings = () => {
    openModal({
      id: 'unified-settings',
      type: 'modal',
      component: 'unifiedSettings',
      size: 'full',
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
      id: 'toolkit',
      type: 'modal',
      component: 'tools',
      size: 'full',
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
      component: 'unifiedProfileEditor',
      size: 'full',
    });
  };

  const handleLogout = async () => {
    await logout();
    onPageChange('start');
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
          className="top-bar-logo toolbar-secondary"
          onClick={openProfileEditor}
          title="Edit agent profile"
        >
          <div className="logo-icon">
            <BrainCircuit size={20} />
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

        {/* Secondary icons — hidden on small screens, exposed via overflow menu */}
        <button
          className="toolbar-icon toolbar-secondary"
          onClick={openTranslation}
          title="Translation"
        >
          <Languages size={18} />
        </button>
        <button
          className="toolbar-icon toolbar-secondary"
          onClick={openTools}
          title="Tools"
        >
          <Wrench size={18} />
        </button>
        <button
          className="toolbar-icon toolbar-secondary"
          onClick={openMemory}
          title="Memory"
        >
          <Database size={18} />
        </button>

        <button
          className="toolbar-icon toolbar-secondary"
          onClick={openSettings}
          title="Settings"
        >
          <Settings size={18} />
        </button>

        {/* Overflow button — only visible on small screens */}
        <button
          ref={overflowButtonRef}
          className={`toolbar-icon toolbar-overflow-trigger ${showOverflow ? 'active' : ''}`}
          onClick={() => (showOverflow ? closeOverflow() : openOverflow())}
          title="More options"
        >
          <MoreHorizontal size={18} />
        </button>

        {authRequired && isAuthenticated && (
          <>
            <span className="topbar-divider toolbar-secondary" />
            <button
              className="toolbar-icon toolbar-secondary"
              onClick={() => openModal({
                id: 'change-password',
                type: 'modal',
                component: 'changePassword',
                size: 'sm',
              })}
              title="Change password"
            >
              <KeyRound size={18} />
            </button>
            <button
              className="toolbar-icon toolbar-icon--danger toolbar-secondary"
              onClick={handleLogout}
              title="Sign out"
            >
              <LogOut size={18} />
            </button>
          </>
        )}
      </div>

      {/* Overflow dropdown — portal-rendered below the overflow button */}
      {showOverflow && overflowPos && createPortal(
        <>
          <div className="toolbar-overflow-backdrop" onClick={closeOverflow} />
          <div
            className="toolbar-overflow-menu"
            style={{ top: overflowPos.top, right: overflowPos.right }}
          >
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openSettings(); }}>
              <Settings size={16} />
              <span>Settings</span>
            </button>
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openTranslation(); }}>
              <Languages size={16} />
              <span>Translation</span>
            </button>
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openTools(); }}>
              <Wrench size={16} />
              <span>Tools</span>
            </button>
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openMemory(); }}>
              <Database size={16} />
              <span>Memory</span>
            </button>
            {authRequired && isAuthenticated && (
              <>
                <div className="toolbar-overflow-divider" />
                <button
                  className="toolbar-overflow-item"
                  onClick={() => {
                    closeOverflow();
                    openModal({ id: 'change-password', type: 'modal', component: 'changePassword', size: 'sm' });
                  }}
                >
                  <KeyRound size={16} />
                  <span>Change Password</span>
                </button>
                <button className="toolbar-overflow-item toolbar-overflow-item--danger" onClick={() => { closeOverflow(); handleLogout(); }}>
                  <LogOut size={16} />
                  <span>Sign Out</span>
                </button>
              </>
            )}
          </div>
        </>,
        document.body,
      )}
    </header>
  );
}

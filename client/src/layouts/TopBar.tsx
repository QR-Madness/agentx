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
  BookMarked,
  Languages,
  Brain,
  Zap,
  LogOut,
  KeyRound,
  MoreHorizontal,
  BrainCircuit,
  ListChecks,
  Command,
  Eye,
  EyeOff,
  MessagesSquare,
} from 'lucide-react';
import { createPortal } from 'react-dom';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { usePlans } from '../contexts/PlansContext';
import { useAuth } from '../contexts/AuthContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { useConsolidationStatus, useIsMobile } from '../lib/hooks';
import { ActiveAgentsDropdown } from '../components/chat/ActiveAgentsDropdown';
import { ConsolidationMenu } from '../components/chat/ConsolidationMenu';
import { ConversationTabBar } from './ConversationTabBar';
import { WindowControls } from './WindowControls';
import { showWindowControls, isMac } from '../lib/platform';
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
  const { livePlans } = usePlans();
  const consolidation = useConsolidationStatus();

  const activePlanCount = Array.from(livePlans.values()).filter(
    p => p.status === 'running',
  ).length;
  const { authRequired, isAuthenticated, logout } = useAuth();
  const { focusMode, toggleFocusMode } = useUIChrome();
  const isMobile = useIsMobile();

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
    const itemCount = 5 + (authRequired && isAuthenticated ? 2 : 0);
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
      id: 'memory',
      type: 'modal',
      component: 'memory',
      size: 'full',
    });
  };

  const openPlans = () => {
    openModal({
      id: 'plans-drawer',
      type: 'drawer',
      component: 'plans',
      position: 'right',
      size: 'xxl',
    });
  };

  const openSources = () => {
    openModal({
      id: 'sources-drawer',
      type: 'drawer',
      component: 'sources',
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

  const openConversations = () => {
    openModal({
      id: 'conversations-drawer',
      type: 'drawer',
      component: 'conversations',
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
      component: 'unifiedProfileEditor',
      size: 'full',
    });
  };

  const handleLogout = async () => {
    await logout();
    onPageChange('start');
  };

  return (
    <header
      className={`top-bar${isMac ? ' top-bar--mac' : ''}${focusMode ? ' top-bar--focus' : ''}`}
      data-tauri-drag-region
    >
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

      {/* Center: Conversation tabs (visible only on AgentX page) + drag area */}
      <div className="top-bar-center" data-tauri-drag-region>
        <ConversationTabBar visible={activePage === 'agentx'} />
      </div>

      {/* Right: live indicators + ⌘K + Focus + Workspace menu + window controls */}
      <div className="top-bar-right">
        {/* Conversations — mobile-only switcher (the tab bar is hidden on mobile) */}
        {isMobile && (
          <button
            className="toolbar-icon"
            onClick={openConversations}
            title="Conversations"
          >
            <MessagesSquare size={18} />
          </button>
        )}

        {/* Consolidation lightning — pulses as a live indicator when active */}
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

        {/* Plans: live indicator only while plans are running */}
        {activePlanCount > 0 && (
          <button
            className="toolbar-icon toolbar-icon-plans building"
            onClick={openPlans}
            title={`${activePlanCount} plan${activePlanCount > 1 ? 's' : ''} in progress`}
          >
            <ListChecks size={18} />
            <span className="toolbar-icon-badge">{activePlanCount}</span>
          </button>
        )}

        {/* Command palette */}
        <button
          className="toolbar-icon toolbar-cmdk"
          onClick={() => window.dispatchEvent(new CustomEvent('agentx:toggle-command-palette'))}
          title="Command palette (⌘K)"
        >
          <Command size={18} />
        </button>

        {/* Focus / Zen mode */}
        <button
          className={`toolbar-icon toolbar-focus ${focusMode ? 'active' : ''}`}
          onClick={toggleFocusMode}
          title={focusMode ? 'Exit focus mode' : 'Focus mode'}
        >
          {focusMode ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>

        {/* Workspace menu — canonical home for all secondary tools */}
        <button
          ref={overflowButtonRef}
          className={`toolbar-icon toolbar-workspace-trigger ${showOverflow ? 'active' : ''}`}
          onClick={() => (showOverflow ? closeOverflow() : openOverflow())}
          title="Workspace"
        >
          <MoreHorizontal size={18} />
        </button>

        {showWindowControls && <WindowControls />}
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
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openPlans(); }}>
              <ListChecks size={16} />
              <span>Plans{activePlanCount > 0 ? ` (${activePlanCount})` : ''}</span>
            </button>
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openMemory(); }}>
              <Database size={16} />
              <span>Memory</span>
            </button>
            <button className="toolbar-overflow-item" onClick={() => { closeOverflow(); openSources(); }}>
              <BookMarked size={16} />
              <span>Sources</span>
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

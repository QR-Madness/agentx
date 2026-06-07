/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 *
 * Brain icon (left) opens the active agents dropdown.
 * Lightning icon (right toolbar) opens the consolidation menu with live SSE progress.
 * Both pulse when their respective operations are active.
 */

import { useState, useRef } from 'react';
import {
  Home,
  LayoutDashboard,
  Bot,
  Brain,
  Zap,
  BrainCircuit,
  ListChecks,
  Eye,
  EyeOff,
  MessagesSquare,
  Search,
} from 'lucide-react';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { usePlans } from '../contexts/PlansContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { SURFACES } from '../lib/surfaces';
import { useConsolidationStatus, useIsMobile } from '../lib/hooks';
import { ActiveAgentsDropdown } from '../components/chat/ActiveAgentsDropdown';
import { ConsolidationMenu } from '../components/chat/ConsolidationMenu';
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
  const { focusMode, toggleFocusMode } = useUIChrome();
  const isMobile = useIsMobile();

  const [showAgentsDropdown, setShowAgentsDropdown] = useState(false);
  const [showConsolidationMenu, setShowConsolidationMenu] = useState(false);

  const brainButtonRef = useRef<HTMLButtonElement>(null);
  const lightningButtonRef = useRef<HTMLButtonElement>(null);

  const openPalette = () =>
    window.dispatchEvent(new CustomEvent('agentx:toggle-command-palette'));

  const hasStreamingTabs = tabs.some(t => t.isStreaming);

  // The few stateful strip icons open surfaces through SURFACES so they can't
  // drift from the command palette (the single home for every other action).
  const openPlans = () => openModal(SURFACES.plans);
  const openConversations = () => openModal(SURFACES.conversations);
  const openProfileEditor = () => openModal(SURFACES.profileEditor);

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

      {/* Center: drag area (conversations now live in the chat-page sidebar) */}
      <div className="top-bar-center" data-tauri-drag-region />

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

        {/* Command palette — the primary entry to everything (replaces the old
            Workspace overflow). Labeled search pill: discoverable, mobile-first. */}
        <button
          className="toolbar-search-pill"
          onClick={openPalette}
          title="Search commands (⌘K)"
          aria-label="Search commands"
        >
          <Search size={16} />
          <span className="toolbar-search-pill-label">Search…</span>
          {!isMobile && (
            <kbd className="toolbar-search-pill-kbd">{isMac ? '⌘' : 'Ctrl'} K</kbd>
          )}
        </button>

        {/* Focus / Zen mode */}
        <button
          className={`toolbar-icon toolbar-focus ${focusMode ? 'active' : ''}`}
          onClick={toggleFocusMode}
          title={focusMode ? 'Exit focus mode' : 'Focus mode'}
        >
          {focusMode ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>

        {showWindowControls && <WindowControls />}
      </div>
    </header>
  );
}

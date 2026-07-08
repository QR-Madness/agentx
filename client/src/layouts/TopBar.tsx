/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 *
 * Lightning icon (right toolbar) opens the consolidation menu with live SSE
 * progress and pulses while consolidation is active. Conversations are reached
 * from the chat-page sidebar (desktop) / its mobile header toggle + the command
 * palette — the old TopBar conversation switchers were removed.
 */

import {
  Home,
  LayoutDashboard,
  Zap,
  BrainCircuit,
  ListChecks,
  Eye,
  EyeOff,
  Search,
} from 'lucide-react';
import { GalaxyIcon } from '../components/common/GalaxyIcon';
import { useModal } from '../contexts/ModalContext';
import { usePlans } from '../contexts/PlansContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { SURFACES } from '../lib/surfaces';
import { useIsMobile } from '../lib/hooks';
import { useConsolidation } from '../contexts/ConsolidationContext';
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
  { id: 'agentx', label: 'Agents', icon: <GalaxyIcon size={16} /> },
];

export function TopBar({ activePage, onPageChange }: TopBarProps) {
  const { openModal } = useModal();
  const { livePlans } = usePlans();
  const consolidation = useConsolidation();

  const activePlanCount = Array.from(livePlans.values()).filter(
    p => p.status === 'running',
  ).length;
  const { focusMode, toggleFocusMode } = useUIChrome();
  const isMobile = useIsMobile();


  const openPalette = () =>
    window.dispatchEvent(new CustomEvent('agentx:toggle-command-palette'));

  // The few stateful strip icons open surfaces through SURFACES so they can't
  // drift from the command palette (the single home for every other action).
  const openPlans = () => openModal(SURFACES.plans);
  const openProfileEditor = () => openModal(SURFACES.profileEditor);

  return (
    <header
      className={`top-bar${isMac ? ' top-bar--mac' : ''}${focusMode ? ' top-bar--focus' : ''}`}
      data-tauri-drag-region
    >
      {/* Left: Logo (edit profile) */}
      <div className="top-bar-left">
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
        {/* Consolidation lightning — pulses as a live indicator when active; opens the drawer */}
        <button
          className={`toolbar-icon toolbar-icon-lightning ${consolidation.isActive ? 'pulsing' : ''}`}
          onClick={() => openModal(SURFACES.consolidation)}
          title="Memory consolidation"
        >
          <Zap size={18} />
        </button>

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
          aria-label={focusMode ? 'Exit focus mode' : 'Focus mode'}
        >
          {focusMode ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>

        {showWindowControls && <WindowControls />}
      </div>
    </header>
  );
}

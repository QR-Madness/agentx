/**
 * TopBar — Horizontal navigation bar with logo, page pills, and toolbar icons
 *
 * Lightning icon (right toolbar) opens the consolidation menu with live SSE
 * progress and pulses while consolidation is active. Conversations are reached
 * from the chat-page sidebar (desktop) / its mobile header toggle + the command
 * palette — the old TopBar conversation switchers were removed.
 */

import { useEffect, useState } from 'react';
import {
  Home,
  LayoutDashboard,
  Zap,
  BrainCircuit,
  ListChecks,
  Eye,
  EyeOff,
  Search,
  Radar,
  FolderKanban,
} from 'lucide-react';
import { GalaxyIcon } from '../components/common/GalaxyIcon';
import { MemoryIcon } from '../components/common/MemoryIcon';
import { useModal } from '../contexts/ModalContext';
import { usePlans } from '../contexts/PlansContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { SURFACES } from '../lib/surfaces';
import { useIsMobile } from '../lib/hooks';
import { useConsolidation } from '../contexts/ConsolidationContext';
import { WindowControls } from './WindowControls';
import { showWindowControls, isMac } from '../lib/platform';
import { api } from '../lib/api';
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
  const { openModal, closeModal, isOpen } = useModal();
  const { livePlans } = usePlans();
  const consolidation = useConsolidation();

  const activePlanCount = Array.from(livePlans.values()).filter(
    p => p.status === 'running',
  ).length;
  const { focusMode, toggleFocusMode } = useUIChrome();
  const isMobile = useIsMobile();

  // Surface pills (Deck / Memory) — first-class desktop tabs whose selected
  // state DERIVES from the modal stack (the Dock lesson: never duplicate
  // open-state). Clicking a selected pill closes its surface (toggle).
  // Palette-only on mobile — the entries in useCommands stay unconditional.
  const deckId = SURFACES.ambassadorDeck.id ?? 'ambassador-deck';
  const memoryId = SURFACES.memory.id ?? 'memory';
  const projectsId = SURFACES.workspaces.id ?? 'workspaces-drawer';
  const deckOpen = isOpen(deckId);
  const memoryOpen = isOpen(memoryId);
  const projectsOpen = isOpen(projectsId);
  const toggleDeck = () =>
    deckOpen ? closeModal(deckId) : openModal(SURFACES.ambassadorDeck);
  const toggleMemory = () =>
    memoryOpen ? closeModal(memoryId) : openModal(SURFACES.memory);
  const toggleProjects = () =>
    projectsOpen ? closeModal(projectsId) : openModal(SURFACES.workspaces);

  // Deck pill live-pulse: background runs in flight (precedent: the consolidation
  // lightning). Cheap 30s poll, desktop only, quiet once the Deck is open.
  const [runsActive, setRunsActive] = useState(false);
  useEffect(() => {
    if (isMobile) return;
    let alive = true;
    const check = () =>
      api.listChatRuns()
        .then(r => { if (alive) setRunsActive(r.runs.some(x => x.status === 'running')); })
        .catch(() => { if (alive) setRunsActive(false); });
    check();
    const t = window.setInterval(check, 30_000);
    return () => { alive = false; window.clearInterval(t); };
  }, [isMobile]);

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

      {/* Center-left: Navigation pills + desktop surface pills (Deck / Memory) */}
      <nav className="top-bar-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            // Tab semantics: while a tab-surface (Deck/Memory) is open it holds the
            // selection — the underlying page pill goes quiet until it closes.
            className={`nav-pill ${activePage === item.id && !deckOpen && !memoryOpen && !projectsOpen ? 'active' : ''}`}
            onClick={() => {
              onPageChange(item.id);
              // Tab behavior: navigating to a page dismisses an open tab-surface.
              if (deckOpen) closeModal(deckId);
              if (memoryOpen) closeModal(memoryId);
              if (projectsOpen) closeModal(projectsId);
            }}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
        {!isMobile && (
          <>
            <span className="mx-1 h-4 w-px shrink-0 self-center bg-line" aria-hidden />
            <button
              className={`nav-pill relative ${deckOpen ? 'active' : ''}`}
              onClick={toggleDeck}
              aria-pressed={deckOpen}
              title={deckOpen ? 'Close the Command Deck' : 'Open the Command Deck'}
            >
              <Radar size={16} />
              <span>Deck</span>
              {runsActive && !deckOpen && (
                <span
                  className="absolute right-1.5 top-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-accent"
                  aria-hidden
                />
              )}
            </button>
            <button
              className={`nav-pill ${memoryOpen ? 'active' : ''}`}
              onClick={toggleMemory}
              aria-pressed={memoryOpen}
              title={memoryOpen ? 'Close Memory' : 'Open Memory'}
            >
              <MemoryIcon size={16} />
              <span>Memory</span>
            </button>
            <button
              className={`nav-pill ${projectsOpen ? 'active' : ''}`}
              onClick={toggleProjects}
              aria-pressed={projectsOpen}
              title={projectsOpen ? 'Close Projects' : 'Open Projects'}
            >
              <FolderKanban size={16} />
              <span>Projects</span>
            </button>
          </>
        )}
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

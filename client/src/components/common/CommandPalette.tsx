/**
 * CommandPalette — ⌘K / Ctrl+K launcher for navigation + every drawer/modal.
 *
 * Consolidates the controls that used to live as individual TopBar icons into a
 * single searchable list, so the titlebar strip can stay minimal. Actions reuse
 * the same `openModal(...)` calls the strip's Workspace menu uses.
 *
 * Opened/closed by RootLayout (which owns the page-navigation state and the
 * global key listener); the strip's ⌘K button dispatches
 * `agentx:toggle-command-palette` which RootLayout also listens for.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Home,
  LayoutDashboard,
  Bot,
  Plus,
  X,
  Settings,
  Wrench,
  Database,
  ListChecks,
  BookMarked,
  Languages,
  BrainCircuit,
  Eye,
  EyeOff,
  Zap,
  Search,
  KeyRound,
  LogOut,
  Radio,
} from 'lucide-react';
import { useModal } from '../../contexts/ModalContext';
import { useConversation } from '../../contexts/ConversationContext';
import { useUIChrome } from '../../contexts/UIChromeContext';
import { useAuth } from '../../contexts/AuthContext';
import type { PageId } from '../../layouts/TopBar';
import './CommandPalette.css';

interface CommandAction {
  id: string;
  label: string;
  hint?: string;
  icon: React.ReactNode;
  keywords?: string;
  run: () => void;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (page: PageId) => void;
}

export function CommandPalette({ isOpen, onClose, onNavigate }: CommandPaletteProps) {
  const { openModal } = useModal();
  const { addTab, closeTab, activeTabId, activeTab, updateTab } = useConversation();
  const { focusMode, toggleFocusMode } = useUIChrome();
  const { authRequired, isAuthenticated, logout } = useAuth();
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const actions = useMemo<CommandAction[]>(() => {
    const go = (page: PageId) => () => { onNavigate(page); onClose(); };
    const open = (m: Parameters<typeof openModal>[0]) => () => { openModal(m); onClose(); };
    const memLocked = !!(activeTab && (activeTab.sessionId || activeTab.messages.length > 0));

    const list: CommandAction[] = [
      { id: 'nav-chat', label: 'Go to Chat', icon: <Bot size={16} />, keywords: 'agentx conversation', run: go('agentx') },
      { id: 'nav-start', label: 'Go to Start', icon: <Home size={16} />, keywords: 'home', run: go('start') },
      { id: 'nav-dashboard', label: 'Go to Dashboard', icon: <LayoutDashboard size={16} />, keywords: 'metrics status', run: go('dashboard') },
      { id: 'conv-new', label: 'New conversation', hint: '⌘T', icon: <Plus size={16} />, keywords: 'tab create chat', run: () => { addTab(); onNavigate('agentx'); onClose(); } },
      {
        id: 'focus', label: focusMode ? 'Exit focus mode' : 'Enter focus mode', hint: 'Zen',
        icon: focusMode ? <EyeOff size={16} /> : <Eye size={16} />, keywords: 'zen immersive hide chrome',
        run: () => { toggleFocusMode(); onClose(); },
      },
      { id: 'open-settings', label: 'Open Settings', hint: '⌘,', icon: <Settings size={16} />, keywords: 'config preferences providers', run: open({ id: 'unified-settings', type: 'modal', component: 'unifiedSettings', size: 'full' }) },
      { id: 'open-tools', label: 'Open Tools', icon: <Wrench size={16} />, keywords: 'toolkit mcp servers', run: open({ id: 'toolkit', type: 'modal', component: 'tools', size: 'full' }) },
      { id: 'open-memory', label: 'Open Memory', icon: <Database size={16} />, keywords: 'facts entities recall', run: open({ id: 'memory-drawer', type: 'drawer', component: 'memory', position: 'right', size: 'xxl' }) },
      { id: 'open-sources', label: 'Open Sources', icon: <BookMarked size={16} />, keywords: 'citations bibliography references links', run: open({ id: 'sources-drawer', type: 'drawer', component: 'sources', position: 'right', size: 'xxl' }) },
      { id: 'open-ambassador', label: 'Open Ambassador', icon: <Radio size={16} />, keywords: 'ambassador briefing summarize interpret turn parallel', run: open({ id: 'ambassador-drawer', type: 'drawer', component: 'ambassador', position: 'right', size: 'xxl' }) },
      { id: 'open-plans', label: 'Open Plans', icon: <ListChecks size={16} />, keywords: 'tasks subtasks progress', run: open({ id: 'plans-drawer', type: 'drawer', component: 'plans', position: 'right', size: 'xxl' }) },
      { id: 'open-translation', label: 'Open Translation', icon: <Languages size={16} />, keywords: 'translate language nllb', run: open({ id: 'translation-modal', type: 'modal', component: 'translation', size: 'lg' }) },
      { id: 'open-profile', label: 'Edit agent profile', icon: <BrainCircuit size={16} />, keywords: 'agent model temperature prompt', run: open({ id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full' }) },
    ];

    if (activeTabId && !memLocked) {
      list.push({
        id: 'toggle-mem',
        label: activeTab?.noMemorization ? 'Enable memorization for this chat' : 'Disable memorization for this chat',
        icon: <Zap size={16} />, keywords: 'memory private temporary no-memorization',
        run: () => { updateTab(activeTabId, { noMemorization: !activeTab?.noMemorization }); onClose(); },
      });
    }
    if (activeTabId) {
      list.push({ id: 'conv-close', label: 'Close conversation', hint: '⌘W', icon: <X size={16} />, keywords: 'tab remove', run: () => { closeTab(activeTabId); onClose(); } });
    }

    // Account actions live only here (and the desktop Workspace menu); the
    // mobile toolbar drops its overflow trigger, so the palette is their home.
    if (authRequired && isAuthenticated) {
      list.push({ id: 'change-password', label: 'Change Password', icon: <KeyRound size={16} />, keywords: 'account security login', run: open({ id: 'change-password', type: 'modal', component: 'changePassword', size: 'sm' }) });
      list.push({ id: 'sign-out', label: 'Sign Out', icon: <LogOut size={16} />, keywords: 'logout exit session', run: () => { onClose(); logout().then(() => onNavigate('start')); } });
    }
    return list;
  }, [openModal, onNavigate, onClose, addTab, closeTab, activeTabId, activeTab, focusMode, toggleFocusMode, updateTab, authRequired, isAuthenticated, logout]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return actions;
    return actions.filter(a => (a.label + ' ' + (a.keywords ?? '')).toLowerCase().includes(q));
  }, [actions, query]);

  // Reset + focus on open
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelected(0);
      // focus after the portal mounts
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [isOpen]);

  useEffect(() => { setSelected(0); }, [query]);

  if (!isOpen) return null;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelected(s => Math.min(s + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelected(s => Math.max(s - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      filtered[selected]?.run();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return createPortal(
    <div className="cmdk-overlay" onClick={onClose} role="presentation">
      <div className="cmdk-panel" onClick={e => e.stopPropagation()} role="dialog" aria-label="Command palette">
        <div className="cmdk-search">
          <Search size={16} className="cmdk-search-icon" />
          <input
            ref={inputRef}
            className="cmdk-input"
            placeholder="Type a command or search…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="cmdk-list" role="listbox">
          {filtered.length === 0 && <div className="cmdk-empty">No matching commands</div>}
          {filtered.map((a, i) => (
            <button
              key={a.id}
              className={`cmdk-item ${i === selected ? 'selected' : ''}`}
              onMouseEnter={() => setSelected(i)}
              onClick={a.run}
              role="option"
              aria-selected={i === selected}
            >
              <span className="cmdk-item-icon">{a.icon}</span>
              <span className="cmdk-item-label">{a.label}</span>
              {a.hint && <kbd className="cmdk-item-hint">{a.hint}</kbd>}
            </button>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}

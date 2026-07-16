/**
 * StartPage — immersive launchpad: a cosmic backdrop, the agent as a living
 * mark (avatar core + orbiting star + glow halo), a time-aware greeting with
 * an ambient status line, and a hero composer — typing here *is* the first
 * message of a new conversation.
 *
 * The composer delivers through the same seam the Ambassador uses
 * (`relayToConversation` → ChatPanel's registered relay handler). The handler
 * registers when ChatPanel mounts with the new tab active, so delivery
 * retries briefly rather than racing the navigation.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  MessageSquarePlus,
  LayoutDashboard,
  MessageSquare,
  Download,
  ChevronDown,
  ChevronRight,
  Send,
} from 'lucide-react';
import { useAgentProfile } from '../contexts/AgentProfileContext';
import { useConversation } from '../contexts/ConversationContext';
import { useConsolidation } from '../contexts/ConsolidationContext';
import { AgentAvatar } from '../components/common/AgentAvatar';
import { getDisplayTitle } from '../lib/conversationTitles';
import type { PageId } from '../layouts/TopBar';
import './StartPage.css';
import { Button } from '../components/ui';

interface StartPageProps {
  onNavigate: (page: PageId) => void;
}

const COLLAPSE_KEY = 'agentx:startRecentsCollapsed';
const MAX_RECENTS = 8;
// The relay handler registers when ChatPanel mounts the new tab; retry the
// delivery briefly (100ms × 40 ≈ 4s ceiling) instead of racing it.
const DELIVER_RETRY_MS = 100;
const DELIVER_MAX_TRIES = 40;

interface RecentRow {
  key: string;
  kind: 'tab' | 'server';
  id: string;
  title: string;
  subtitle: string;
  lastAt: string;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
  if (days <= 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function daypart(): string {
  const hour = new Date().getHours();
  if (hour < 5) return 'evening';
  if (hour < 12) return 'morning';
  if (hour < 18) return 'afternoon';
  return 'evening';
}

function isToday(dateStr: string | null): boolean {
  if (!dateStr) return false;
  return new Date(dateStr).toDateString() === new Date().toDateString();
}

export function StartPage({ onNavigate }: StartPageProps) {
  const { profiles, activeProfile, getAgentName } = useAgentProfile();
  const {
    tabs, addTab, switchTab, relayToConversation,
    serverConversations, refreshHistory, restoreConversation,
  } = useConversation();
  const consolidation = useConsolidation();
  const agentName = getAgentName();

  const [collapsed, setCollapsed] = useState<boolean>(
    () => localStorage.getItem(COLLAPSE_KEY) === 'true',
  );
  const [draft, setDraft] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Pull fresh history when the Start page mounts so the list isn't stale.
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const toggleCollapsed = () => {
    setCollapsed(prev => {
      const next = !prev;
      localStorage.setItem(COLLAPSE_KEY, String(next));
      return next;
    });
  };

  // Merge open tabs + server history (dedupe server convs already open as tabs),
  // newest first, capped — mirrors ConversationHistoryDropdown's dedupe.
  const recents = useMemo<RecentRow[]>(() => {
    const openSessionIds = new Set(tabs.map(t => t.sessionId).filter(Boolean));
    const tabRows: RecentRow[] = tabs.map(t => ({
      key: `tab:${t.id}`,
      kind: 'tab',
      id: t.id,
      title: t.title,
      subtitle: `${t.messages.length} messages · ${formatDate(t.lastMessageAt)}`,
      lastAt: t.lastMessageAt,
    }));
    const serverRows: RecentRow[] = serverConversations
      .filter(c => !openSessionIds.has(c.conversation_id))
      .map(c => ({
        key: `srv:${c.conversation_id}`,
        kind: 'server',
        id: c.conversation_id,
        title: getDisplayTitle(c.conversation_id, c.title),
        subtitle: `${c.message_count} messages · ${formatDate(c.last_message_at)}`,
        lastAt: c.last_message_at || '',
      }));
    return [...tabRows, ...serverRows]
      .sort((a, b) => new Date(b.lastAt).getTime() - new Date(a.lastAt).getTime())
      .slice(0, MAX_RECENTS);
  }, [tabs, serverConversations]);

  // Ambient status line — only from data the page already has.
  const ambient = useMemo(() => {
    const parts: string[] = [];
    const agents = profiles.filter(p => p.kind === 'agent').length;
    if (agents > 0) parts.push(`${agents} agent${agents === 1 ? '' : 's'} ready`);
    const today =
      tabs.filter(t => isToday(t.lastMessageAt) && t.messages.length > 0).length +
      serverConversations.filter(
        c => isToday(c.last_message_at) && !tabs.some(t => t.sessionId === c.conversation_id),
      ).length;
    if (today > 0) parts.push(`${today} conversation${today === 1 ? '' : 's'} today`);
    parts.push(consolidation.isActive ? 'memory consolidating' : 'memory idle');
    return parts.join(' · ');
  }, [profiles, tabs, serverConversations, consolidation.isActive]);

  // Deliver the hero draft into the freshly-created tab once ChatPanel has
  // registered its relay handler (returns false until then).
  const deliver = (tabId: string, text: string, attempt = 0) => {
    if (relayToConversation(tabId, text)) return;
    if (attempt < DELIVER_MAX_TRIES) {
      setTimeout(() => deliver(tabId, text, attempt + 1), DELIVER_RETRY_MS);
    }
  };

  const launchConversation = () => {
    const text = draft.trim();
    const tab = addTab();
    onNavigate('agentx');
    if (text) {
      setDraft('');
      deliver(tab.id, text);
    }
  };

  const handleComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (draft.trim()) launchConversation();
    }
  };

  // Auto-grow the composer up to its CSS max-height.
  const handleComposerInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };

  const handleOpenDashboard = () => onNavigate('dashboard');

  const handleOpenRecent = async (row: RecentRow) => {
    if (row.kind === 'tab') {
      switchTab(row.id);
    } else {
      try {
        await restoreConversation(row.id);
      } catch {
        // notify is handled by the history surfaces; Start stays quiet
      }
    }
    onNavigate('agentx');
  };

  return (
    <div className="start-page">
      {/* Cosmic backdrop — stars via layered radial-gradients, drifting motes */}
      <div className="start-backdrop" aria-hidden="true">
        <span className="start-mote start-mote-1" />
        <span className="start-mote start-mote-2" />
        <span className="start-mote start-mote-3" />
      </div>

      <div className="start-content">
        <div className="start-hero">
          {/* The agent as a living mark: glow halo + orbit + avatar core */}
          <div className="start-mark" aria-hidden="true">
            <span className="start-mark-halo" />
            <span className="start-mark-orbit">
              <span className="start-mark-star" />
            </span>
            <div className="start-mark-core">
              <AgentAvatar avatar={activeProfile?.avatar} size={28} fill />
            </div>
          </div>

          <h1 className="start-title">Good {daypart()} — I'm {agentName}</h1>
          <p className="start-status">{ambient}</p>

          <div className="ax-fieldwrap start-composer">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onInput={handleComposerInput}
              onKeyDown={handleComposerKeyDown}
              rows={1}
              placeholder={`Message ${agentName} — starts a new conversation…`}
              className="start-composer-input max-h-32 flex-1 resize-none bg-transparent p-0 text-sm text-fg outline-none placeholder:text-fg-muted max-[600px]:text-base"
            />
            <button
              type="button"
              onClick={launchConversation}
              disabled={!draft.trim()}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent text-fg-inverse shadow-sm transition hover:brightness-110 active:brightness-95 disabled:opacity-40"
              title="Start the conversation"
              aria-label="Start the conversation"
            >
              <Send size={14} />
            </button>
          </div>

          <div className="start-actions">
            <Button variant="secondary" className="start-cta-secondary" onClick={launchConversation}>
              <MessageSquarePlus size={16} />
              <span>New Conversation</span>
            </Button>
            <Button variant="secondary" className="start-cta-secondary" onClick={handleOpenDashboard}>
              <LayoutDashboard size={16} />
              <span>Open Dashboard</span>
            </Button>
          </div>
        </div>

        {recents.length > 0 && (
          <div className="start-recents">
            <button
              className="start-recents-header"
              onClick={toggleCollapsed}
              aria-expanded={!collapsed}
            >
              {collapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
              <span>Recent Conversations</span>
              <span className="start-recents-count">{recents.length}</span>
            </button>

            {!collapsed && (
              <div className="start-recents-list">
                {recents.map(row => (
                  <button
                    key={row.key}
                    className="start-recent-item"
                    onClick={() => handleOpenRecent(row)}
                    title={row.title}
                  >
                    <span className="start-recent-icon">
                      {row.kind === 'tab' ? <MessageSquare size={15} /> : <Download size={15} />}
                    </span>
                    <span className="start-recent-info">
                      <span className="start-recent-title">{row.title}</span>
                      <span className="start-recent-meta">{row.subtitle}</span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

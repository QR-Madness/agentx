/**
 * StartPage — Welcome screen with agent greeting, quick actions, and a
 * collapsible list of recent conversations (open tabs + server history).
 */

import { useEffect, useMemo, useState } from 'react';
import {
  MessageSquarePlus,
  LayoutDashboard,
  MessageSquare,
  Download,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { useAgentProfile } from '../contexts/AgentProfileContext';
import { useConversation } from '../contexts/ConversationContext';
import { getAvatarIcon } from '../lib/avatars';
import { getDisplayTitle } from '../lib/conversationTitles';
import type { PageId } from '../layouts/TopBar';
import './StartPage.css';
import { Button } from '../components/ui';

interface StartPageProps {
  onNavigate: (page: PageId) => void;
}

const COLLAPSE_KEY = 'agentx:startRecentsCollapsed';
const MAX_RECENTS = 8;

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

export function StartPage({ onNavigate }: StartPageProps) {
  const { activeProfile, getAgentName } = useAgentProfile();
  const {
    tabs, switchTab, serverConversations, refreshHistory, restoreConversation,
  } = useConversation();
  const agentName = getAgentName();
  const AvatarIcon = getAvatarIcon(activeProfile?.avatar);

  const [collapsed, setCollapsed] = useState<boolean>(
    () => localStorage.getItem(COLLAPSE_KEY) === 'true',
  );

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

  const handleNewConversation = () => onNavigate('agentx');
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
      <div className="start-content">
        <div className="start-hero">
          <div className="start-logo">
            <AvatarIcon size={48} />
          </div>
          <h1 className="start-title">Hello, I'm {agentName}</h1>
          <p className="start-subtitle">How can I assist you today?</p>
          <div className="start-actions">
            <Button variant="primary" className="start-cta" onClick={handleNewConversation}>
              <MessageSquarePlus size={18} />
              <span>New Conversation</span>
            </Button>
            <Button variant="secondary" className="start-cta-secondary" onClick={handleOpenDashboard}>
              <LayoutDashboard size={18} />
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

/**
 * UserHistoryView — shared presentation for `recall_user_history` results.
 *
 * Rendered both inline in chat (when the agent calls the tool) and in the
 * Memory drawer's "User History" tab (manual browse via /api/memory/user-history).
 * Pure: takes already-fetched turns + facts.
 */

import { MessageSquare, Sparkles } from 'lucide-react';
import type { UserHistoryFact, UserHistoryTurn } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import './UserHistoryView.css';

interface UserHistoryViewProps {
  turns: UserHistoryTurn[];
  facts: UserHistoryFact[];
  topic?: string | null;
  compact?: boolean;
}

export function UserHistoryView({ turns, facts, topic, compact }: UserHistoryViewProps) {
  const empty = turns.length === 0 && facts.length === 0;

  return (
    <div className={`user-history-view ${compact ? 'compact' : ''}`}>
      {topic && (
        <div className="user-history-topic">
          Topic: <span>{topic}</span>
        </div>
      )}

      {empty && (
        <div className="user-history-empty">No prior history found.</div>
      )}

      {turns.length > 0 && (
        <div className="user-history-section">
          <div className="user-history-section-title">
            <MessageSquare size={12} />
            <span>Past messages ({turns.length})</span>
          </div>
          <ul className="user-history-turns">
            {turns.map((t, i) => (
              <li key={`${t.conversation_id ?? 'c'}-${t.timestamp}-${i}`} className="user-history-turn">
                <div className="user-history-turn-content">{t.content}</div>
                <div className="user-history-turn-time">{formatTimestamp(t.timestamp)}</div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {facts.length > 0 && (
        <div className="user-history-section">
          <div className="user-history-section-title">
            <Sparkles size={12} />
            <span>Known facts ({facts.length})</span>
          </div>
          <ul className="user-history-facts">
            {facts.map((f, i) => (
              <li key={i} className="user-history-fact">
                <span className="user-history-fact-claim">{f.claim}</span>
                <span className="user-history-fact-confidence">
                  {Math.round((f.confidence ?? 0) * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * UserHistoryTab — manual browse of the user's recalled history in the Memory
 * drawer. Wraps POST /api/memory/user-history (the HTTP twin of the
 * `recall_user_history` agent tool, which needs an active chat context).
 */

import { useEffect, useState, useCallback } from 'react';
import { Search, RefreshCw } from 'lucide-react';
import { api } from '../../lib/api';
import type { UserHistoryResponse } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { UserHistoryView } from './UserHistoryView';

export function UserHistoryTab() {
  const { notifyError } = useNotify();
  const [topic, setTopic] = useState('');
  const [data, setData] = useState<UserHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (t: string) => {
    setLoading(true);
    try {
      const res = await api.getUserHistory({ topic: t || undefined, limit: 20 });
      setData(res);
    } catch (err) {
      notifyError(err, 'Failed to load user history');
    } finally {
      setLoading(false);
    }
  }, [notifyError]);

  useEffect(() => {
    load('');
  }, [load]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    load(topic.trim());
  };

  return (
    <div className="memory-list-container card full-height">
      <form className="memory-filters" onSubmit={onSubmit} style={{ marginBottom: 12 }}>
        <div className="filter-group search always-visible">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search the user's past messages (Enter)..."
            value={topic}
            onChange={e => setTopic(e.target.value)}
          />
        </div>
        <button type="button" className="memory-refresh-btn" onClick={() => load(topic.trim())} disabled={loading} aria-label="Refresh">
          <RefreshCw size={16} className={loading ? 'spin' : ''} />
        </button>
      </form>

      {loading && !data ? (
        <div className="user-history-empty">Loading…</div>
      ) : data ? (
        <UserHistoryView turns={data.user_turns} facts={data.facts} topic={data.topic} summary={data.summary} />
      ) : null}
    </div>
  );
}

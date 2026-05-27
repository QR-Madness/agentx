// ─── Entity list view ──────────────────────────────────────────────────────

import { Users, RefreshCw } from 'lucide-react';
import type { MemoryEntity, ApiError } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';

export function EntityListView({
  entities, total, loading, error, selectedEntityId, onSelectEntity,
}: {
  entities: MemoryEntity[];
  total: number;
  loading: boolean;
  error: ApiError | null;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
}) {
  if (loading) return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading entities...</p></div>;
  if (error) return <div className="memory-error"><p>Failed to load entities: {error.message}</p></div>;
  if (entities.length === 0) return <div className="memory-empty"><Users size={32} /><p>No entities found</p></div>;

  return (
    <div className="memory-list">
      <div className="memory-list-header">
        <span>Name</span><span>Type</span><span>Channel</span><span>Salience</span><span>Last Accessed</span>
      </div>
      {entities.map(entity => (
        <div
          key={entity.id}
          className={`memory-row${selectedEntityId === entity.id ? ' selected' : ''}`}
          onClick={() => onSelectEntity(selectedEntityId === entity.id ? null : entity.id)}
        >
          <span className="entity-name">{entity.name}</span>
          <span className="entity-type badge">{entity.type}</span>
          <span className="entity-channel">{entity.channel}</span>
          <span className="entity-salience">
            <div className="salience-bar">
              <div className="salience-fill" style={{ width: `${(entity.salience || 0) * 100}%` }} />
            </div>
            <span>{((entity.salience || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="entity-accessed">{formatTimestamp(entity.last_accessed)}</span>
        </div>
      ))}
      <div className="memory-list-footer">Showing {entities.length} of {total} entities</div>
    </div>
  );
}

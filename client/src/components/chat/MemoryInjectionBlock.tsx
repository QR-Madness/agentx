/**
 * MemoryInjectionBlock — Renders memory context injection with facts and entities
 */

import { useState } from 'react';
import { Database, ChevronDown, ChevronRight, Lightbulb, Tag } from 'lucide-react';
import './MemoryInjectionBlock.css';

interface Fact {
  claim: string;
  confidence: number;
  source?: string;
}

interface Entity {
  name: string;
  type: string;
}

export interface MemoryInjectionBlockProps {
  facts: Fact[];
  entities: Entity[];
  queryUsed: string;
}

// Entity type colors for badges
const ENTITY_TYPE_COLORS: Record<string, string> = {
  person: '#3b82f6',
  organization: '#8b5cf6',
  location: '#10b981',
  event: '#f59e0b',
  concept: '#ec4899',
  default: '#6b7280',
};

function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);
  const getColor = () => {
    if (percentage >= 80) return '#22c55e';
    if (percentage >= 60) return '#eab308';
    return '#ef4444';
  };

  return (
    <div className="confidence-bar">
      <div
        className="confidence-fill"
        style={{ width: `${percentage}%`, background: getColor() }}
      />
      <span className="confidence-label">{percentage}%</span>
    </div>
  );
}

function EntityBadge({ entity }: { entity: Entity }) {
  const color = ENTITY_TYPE_COLORS[entity.type.toLowerCase()] || ENTITY_TYPE_COLORS.default;

  return (
    <span
      className="entity-badge"
      style={{ borderColor: color, color }}
    >
      <Tag size={10} />
      <span className="entity-name">{entity.name}</span>
      <span className="entity-type">{entity.type}</span>
    </span>
  );
}

export function MemoryInjectionBlock({ facts, entities, queryUsed }: MemoryInjectionBlockProps) {
  const [expanded, setExpanded] = useState(false);

  const totalItems = facts.length + entities.length;

  return (
    <div className="memory-injection-block">
      <div className="memory-header" onClick={() => setExpanded(!expanded)}>
        <div className="memory-icon">
          <Database size={14} />
        </div>
        <div className="memory-info">
          <span className="memory-title">Memory Context</span>
          <span className="memory-count">
            {facts.length} facts, {entities.length} entities
          </span>
        </div>
        <button className="memory-toggle">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {expanded && (
        <div className="memory-content">
          {/* Query Used */}
          <div className="memory-query">
            <span className="section-label">Query:</span>
            <span className="query-text">{queryUsed}</span>
          </div>

          {/* Facts */}
          {facts.length > 0 && (
            <div className="memory-facts">
              <div className="section-header">
                <Lightbulb size={12} />
                <span>Facts ({facts.length})</span>
              </div>
              <ul className="facts-list">
                {facts.map((fact, idx) => (
                  <li key={idx} className="fact-item">
                    <div className="fact-claim">{fact.claim}</div>
                    <div className="fact-meta">
                      <ConfidenceBar confidence={fact.confidence} />
                      {fact.source && (
                        <span className="fact-source">Source: {fact.source}</span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Entities */}
          {entities.length > 0 && (
            <div className="memory-entities">
              <div className="section-header">
                <Tag size={12} />
                <span>Entities ({entities.length})</span>
              </div>
              <div className="entities-list">
                {entities.map((entity, idx) => (
                  <EntityBadge key={idx} entity={entity} />
                ))}
              </div>
            </div>
          )}

          {totalItems === 0 && (
            <div className="memory-empty">
              No relevant memories found for this query.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

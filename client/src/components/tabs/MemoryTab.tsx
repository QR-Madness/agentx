/**
 * Memory Explorer Tab
 * Browse and inspect memory contents: entities, facts, and strategies.
 */

import React, { useState, useMemo } from 'react';
import {
  Database,
  Users,
  FileText,
  Zap,
  Search,
  RefreshCw,
  ChevronRight,
  X,
  ArrowUpRight,
  ChevronLeft,
  Clock
} from 'lucide-react';
import {
  useMemoryChannels,
  useMemoryEntities,
  useMemoryFacts,
  useMemoryStrategies,
  useMemoryStats,
  useEntityGraph,
  useConsolidate
} from '../../lib/hooks';
import { MemoryEntity, MemoryFact, MemoryStrategy } from '../../lib/api';
import { JobsPanel } from '../JobsPanel';
import '../../styles/MemoryTab.css';

type MemorySection = 'entities' | 'facts' | 'strategies' | 'jobs';

// Format timestamp for display
function formatTimestamp(timestamp: string | undefined): string {
  if (!timestamp) return 'Never';
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  } catch {
    return 'Unknown';
  }
}

// Entity List View Component
function EntityListView({
  channel,
  page,
  search,
  onSelectEntity,
  selectedEntityId
}: {
  channel: string;
  page: number;
  search: string;
  onSelectEntity: (id: string | null) => void;
  selectedEntityId: string | null;
}) {
  const { entities, total, loading, error } = useMemoryEntities(channel, page, search);

  if (loading) {
    return (
      <div className="memory-loading">
        <RefreshCw size={24} className="spin" />
        <p>Loading entities...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="memory-error">
        <p>Failed to load entities: {error.message}</p>
      </div>
    );
  }

  if (entities.length === 0) {
    return (
      <div className="memory-empty">
        <Users size={32} />
        <p>No entities found</p>
        {search && <p className="hint">Try adjusting your search query</p>}
      </div>
    );
  }

  return (
    <div className="memory-list">
      <div className="memory-list-header">
        <span>Name</span>
        <span>Type</span>
        <span>Channel</span>
        <span>Salience</span>
        <span>Last Accessed</span>
      </div>
      {entities.map((entity: MemoryEntity) => (
        <div
          key={entity.id}
          className={`memory-row ${selectedEntityId === entity.id ? 'selected' : ''}`}
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
      <div className="memory-list-footer">
        Showing {entities.length} of {total} entities
      </div>
    </div>
  );
}

// Fact List View Component
function FactListView({
  channel,
  page,
  search,
  minConfidence
}: {
  channel: string;
  page: number;
  search: string;
  minConfidence: number;
}) {
  const { facts, total, loading, error } = useMemoryFacts(channel, page, minConfidence, search);

  if (loading) {
    return (
      <div className="memory-loading">
        <RefreshCw size={24} className="spin" />
        <p>Loading facts...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="memory-error">
        <p>Failed to load facts: {error.message}</p>
      </div>
    );
  }

  if (facts.length === 0) {
    return (
      <div className="memory-empty">
        <FileText size={32} />
        <p>No facts found</p>
        {(search || minConfidence > 0) && <p className="hint">Try adjusting your filters</p>}
      </div>
    );
  }

  return (
    <div className="memory-list">
      <div className="memory-list-header facts-header">
        <span>Claim</span>
        <span>Confidence</span>
        <span>Source</span>
        <span>Channel</span>
        <span>Created</span>
      </div>
      {facts.map((fact: MemoryFact) => (
        <div key={fact.id} className="memory-row fact-row">
          <span className="fact-claim">
            {fact.claim}
            {fact.promoted_from && (
              <span className="promoted-badge">
                <ArrowUpRight size={12} />
                from {fact.promoted_from}
              </span>
            )}
          </span>
          <span className="fact-confidence">
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${(fact.confidence || 0) * 100}%` }}
              />
            </div>
            <span>{((fact.confidence || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="fact-source badge">{fact.source}</span>
          <span className="fact-channel">{fact.channel}</span>
          <span className="fact-created">{formatTimestamp(fact.created_at)}</span>
        </div>
      ))}
      <div className="memory-list-footer">
        Showing {facts.length} of {total} facts
      </div>
    </div>
  );
}

// Strategy List View Component
function StrategyListView({
  channel,
  page
}: {
  channel: string;
  page: number;
}) {
  const { strategies, total, loading, error } = useMemoryStrategies(channel, page);

  if (loading) {
    return (
      <div className="memory-loading">
        <RefreshCw size={24} className="spin" />
        <p>Loading strategies...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="memory-error">
        <p>Failed to load strategies: {error.message}</p>
      </div>
    );
  }

  if (strategies.length === 0) {
    return (
      <div className="memory-empty">
        <Zap size={32} />
        <p>No strategies found</p>
        <p className="hint">Strategies are learned from successful tool usage patterns</p>
      </div>
    );
  }

  return (
    <div className="memory-list">
      <div className="memory-list-header strategies-header">
        <span>Description</span>
        <span>Tool Sequence</span>
        <span>Success Rate</span>
        <span>Channel</span>
        <span>Last Used</span>
      </div>
      {strategies.map((strategy: MemoryStrategy) => (
        <div key={strategy.id} className="memory-row strategy-row">
          <span className="strategy-description">{strategy.description}</span>
          <span className="strategy-tools">
            <div className="tool-sequence">
              {strategy.tool_sequence.map((tool, i) => (
                <span key={i} className="tool-chip">{tool}</span>
              ))}
            </div>
          </span>
          <span className="strategy-success">
            <div className="success-bar">
              <div
                className="success-fill"
                style={{ width: `${(strategy.success_rate || 0) * 100}%` }}
              />
            </div>
            <span>{((strategy.success_rate || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="strategy-channel">{strategy.channel}</span>
          <span className="strategy-used">{formatTimestamp(strategy.last_used)}</span>
        </div>
      ))}
      <div className="memory-list-footer">
        Showing {strategies.length} of {total} strategies
      </div>
    </div>
  );
}

// Entity Detail Panel Component
function EntityDetailPanel({
  entityId,
  onClose
}: {
  entityId: string;
  onClose: () => void;
}) {
  const { graph, loading, error } = useEntityGraph(entityId);

  if (loading) {
    return (
      <div className="entity-detail-panel">
        <div className="panel-header">
          <h3>Entity Details</h3>
          <button className="button-ghost" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <div className="panel-content">
          <div className="memory-loading">
            <RefreshCw size={24} className="spin" />
            <p>Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !graph?.entity) {
    return (
      <div className="entity-detail-panel">
        <div className="panel-header">
          <h3>Entity Details</h3>
          <button className="button-ghost" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <div className="panel-content">
          <div className="memory-error">
            <p>Failed to load entity details</p>
          </div>
        </div>
      </div>
    );
  }

  const { entity, facts, relationships } = graph;

  return (
    <div className="entity-detail-panel">
      <div className="panel-header">
        <h3>{entity.name}</h3>
        <button className="button-ghost" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      <div className="panel-content">
        <div className="entity-info">
          <div className="info-row">
            <span className="label">Type</span>
            <span className="value badge">{entity.type}</span>
          </div>
          <div className="info-row">
            <span className="label">Channel</span>
            <span className="value">{entity.channel}</span>
          </div>
          <div className="info-row">
            <span className="label">Salience</span>
            <span className="value">{((entity.salience || 0) * 100).toFixed(0)}%</span>
          </div>
          {entity.description && (
            <div className="info-row">
              <span className="label">Description</span>
              <span className="value">{entity.description}</span>
            </div>
          )}
          {entity.aliases && entity.aliases.length > 0 && (
            <div className="info-row">
              <span className="label">Aliases</span>
              <span className="value">{entity.aliases.join(', ')}</span>
            </div>
          )}
        </div>

        {facts.length > 0 && (
          <div className="entity-section">
            <h4>Connected Facts ({facts.length})</h4>
            <div className="connected-facts">
              {facts.map(fact => (
                <div key={fact.id} className="connected-fact">
                  <span className="fact-text">{fact.claim}</span>
                  <span className="fact-meta">
                    {((fact.confidence || 0) * 100).toFixed(0)}% confidence
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {relationships.length > 0 && (
          <div className="entity-section">
            <h4>Relationships ({relationships.length})</h4>
            <div className="entity-relationships">
              {relationships.map((rel, i) => (
                <div key={i} className="relationship">
                  <span className="rel-type">{rel.type}</span>
                  <span className="rel-arrow">→</span>
                  <span className="rel-target">
                    <span className="target-name">{rel.target.name}</span>
                    <span className="target-type badge">{rel.target.type}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Pagination Component
function Pagination({
  page,
  hasNext,
  onPageChange
}: {
  page: number;
  hasNext: boolean;
  onPageChange: (page: number) => void;
}) {
  return (
    <div className="pagination">
      <button
        className="button-ghost"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        <ChevronLeft size={16} />
        Previous
      </button>
      <span className="page-info">Page {page}</span>
      <button
        className="button-ghost"
        disabled={!hasNext}
        onClick={() => onPageChange(page + 1)}
      >
        Next
        <ChevronRight size={16} />
      </button>
    </div>
  );
}

// Main Memory Tab Component
export const MemoryTab: React.FC = () => {
  const [activeSection, setActiveSection] = useState<MemorySection>('entities');
  const [selectedChannel, setSelectedChannel] = useState('_global');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [confidenceFilter, setConfidenceFilter] = useState(0);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [searchExpanded, setSearchExpanded] = useState(false);

  const [consolidateMessage, setConsolidateMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const { channels } = useMemoryChannels();
  const { stats, refresh: refreshStats } = useMemoryStats();
  const { consolidate, loading: consolidating } = useConsolidate();

  // Compute hasNext based on current data
  const { hasNext: entitiesHasNext } = useMemoryEntities(selectedChannel, currentPage, searchQuery);
  const { hasNext: factsHasNext } = useMemoryFacts(selectedChannel, currentPage, confidenceFilter / 100, searchQuery);
  const { hasNext: strategiesHasNext } = useMemoryStrategies(selectedChannel, currentPage);

  const hasNext = useMemo(() => {
    switch (activeSection) {
      case 'entities': return entitiesHasNext ?? false;
      case 'facts': return factsHasNext ?? false;
      case 'strategies': return strategiesHasNext ?? false;
      case 'jobs': return false; // Jobs panel handles its own pagination
      default: return false;
    }
  }, [activeSection, entitiesHasNext, factsHasNext, strategiesHasNext]);

  const memorySections = [
    { id: 'entities' as const, label: 'Entities', icon: <Users size={18} /> },
    { id: 'facts' as const, label: 'Facts', icon: <FileText size={18} /> },
    { id: 'strategies' as const, label: 'Strategies', icon: <Zap size={18} /> },
    { id: 'jobs' as const, label: 'Jobs', icon: <Clock size={18} /> },
  ];

  const handleSectionChange = (section: MemorySection) => {
    setActiveSection(section);
    setCurrentPage(1);
    setSelectedEntityId(null);
    setSearchQuery('');
  };

  const handleChannelChange = (channel: string) => {
    setSelectedChannel(channel);
    setCurrentPage(1);
    setSelectedEntityId(null);
  };

  const handleConsolidate = async () => {
    try {
      const result = await consolidate();
      const totalEntities = result.results?.consolidate?.entities ?? 0;
      const totalFacts = result.results?.consolidate?.facts ?? 0;
      const totalRelationships = result.results?.consolidate?.relationships ?? 0;

      setConsolidateMessage({
        type: 'success',
        text: `Extracted ${totalEntities} entities, ${totalFacts} facts, ${totalRelationships} relationships`
      });

      // Refresh stats and lists
      refreshStats();

      // Auto-dismiss after 5 seconds
      setTimeout(() => setConsolidateMessage(null), 5000);
    } catch (err) {
      setConsolidateMessage({
        type: 'error',
        text: `Consolidation failed: ${(err as Error).message}`
      });
      setTimeout(() => setConsolidateMessage(null), 5000);
    }
  };

  return (
    <div className="memory-tab">
      {/* Header */}
      <div className="memory-header fade-in">
        <div className="header-title-row">
          <h1 className="page-title">
            <Database className="page-icon-svg" />
            <span>Memory Explorer</span>
          </h1>
          <div className="header-actions">
            <button
              className="button-primary consolidate-button"
              onClick={handleConsolidate}
              disabled={consolidating}
              title="Run consolidation to extract entities and facts from conversations"
            >
              {consolidating ? (
                <><RefreshCw size={16} className="spin" /> Consolidating...</>
              ) : (
                <><Zap size={16} /> Consolidate Now</>
              )}
            </button>
            <button className="button-ghost" onClick={refreshStats} title="Refresh stats">
              <RefreshCw size={18} />
            </button>
          </div>
        </div>
        <p className="page-subtitle">Browse and inspect stored memories</p>

        {/* Consolidation message */}
        {consolidateMessage && (
          <div className={`consolidate-message ${consolidateMessage.type}`}>
            {consolidateMessage.type === 'success' ? (
              <Zap size={16} />
            ) : (
              <X size={16} />
            )}
            <span>{consolidateMessage.text}</span>
            <button className="dismiss-btn" onClick={() => setConsolidateMessage(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* Stats badges */}
        <div className="memory-stats-bar">
          <span className="stat-badge">
            <Users size={14} /> {stats?.totals.entities ?? 0} Entities
          </span>
          <span className="stat-badge">
            <FileText size={14} /> {stats?.totals.facts ?? 0} Facts
          </span>
          <span className="stat-badge">
            <Zap size={14} /> {stats?.totals.strategies ?? 0} Strategies
          </span>
        </div>
      </div>

      <div className="memory-layout">
        {/* Sidebar Navigation */}
        <nav className="memory-nav card">
          {memorySections.map(section => (
            <button
              key={section.id}
              className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => handleSectionChange(section.id)}
            >
              <span className="nav-icon">{section.icon}</span>
              <span className="nav-label">{section.label}</span>
              <ChevronRight size={16} className="nav-arrow" />
            </button>
          ))}
        </nav>

        {/* Content Area */}
        <div className="memory-content">
          {activeSection === 'jobs' ? (
            /* Jobs Panel - has its own layout */
            <div className="memory-list-container card">
              <JobsPanel />
            </div>
          ) : (
            <>
              {/* Filters */}
              <div className="memory-filters card">
                <div className="filter-group">
                  <label>Channel</label>
                  <select
                    value={selectedChannel}
                    onChange={(e) => handleChannelChange(e.target.value)}
                  >
                    <option value="_global">_global (default)</option>
                    {channels
                      .filter(ch => ch.name !== '_global')
                      .map(ch => (
                        <option key={ch.name} value={ch.name}>
                          {ch.name}
                        </option>
                      ))}
                  </select>
                </div>

                <div className={`filter-group search ${searchExpanded ? 'expanded' : ''}`}>
                  <button
                    className="search-toggle button-ghost"
                    onClick={() => setSearchExpanded(!searchExpanded)}
              >
                <Search size={16} />
              </button>
              {searchExpanded && (
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setCurrentPage(1);
                  }}
                  autoFocus
                />
              )}
            </div>

            {activeSection === 'facts' && (
              <div className="filter-group confidence">
                <label>Min Confidence: {confidenceFilter}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={confidenceFilter}
                  onChange={(e) => {
                    setConfidenceFilter(Number(e.target.value));
                    setCurrentPage(1);
                  }}
                />
              </div>
            )}
          </div>

          {/* List Views */}
          <div className="memory-list-container card">
            {activeSection === 'entities' && (
              <EntityListView
                channel={selectedChannel}
                page={currentPage}
                search={searchQuery}
                onSelectEntity={setSelectedEntityId}
                selectedEntityId={selectedEntityId}
              />
            )}

            {activeSection === 'facts' && (
              <FactListView
                channel={selectedChannel}
                page={currentPage}
                search={searchQuery}
                minConfidence={confidenceFilter / 100}
              />
            )}

            {activeSection === 'strategies' && (
              <StrategyListView
                channel={selectedChannel}
                page={currentPage}
              />
            )}
          </div>

              {/* Pagination */}
              <Pagination
                page={currentPage}
                hasNext={hasNext}
                onPageChange={setCurrentPage}
              />
            </>
          )}
        </div>
      </div>

      {/* Entity Detail Panel */}
      {selectedEntityId && (
        <EntityDetailPanel
          entityId={selectedEntityId}
          onClose={() => setSelectedEntityId(null)}
        />
      )}
    </div>
  );
};

export default MemoryTab;

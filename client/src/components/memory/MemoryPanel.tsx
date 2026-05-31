/**
 * Memory Explorer — Split-pane data browser with inline edit/delete and graph visualization.
 *
 * Orchestrator: owns section/selection/pagination state and the memory data hooks,
 * and composes the extracted list views, detail panels, graph, and pagination
 * (all in this folder). Pure presentational pieces live in their own files.
 */

import React, { useState, useMemo, useRef } from 'react';
import {
  Database, Users, FileText, Zap, Search, RefreshCw,
  ChevronRight, X, Clock, GitBranch, History, Download, Upload, AlertTriangle, Check
} from 'lucide-react';
import {
  useMemoryEntities, useMemoryFacts, useMemoryStrategies,
  useMemoryStats, useConsolidate, useExportMemory, useImportMemory,
} from '../../lib/hooks';
import type { MemoryExport, MemoryFact, MemoryStrategy } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { downloadJson, readJsonFile, fileTimestamp } from '../../lib/fileTransfer';
import { JobsPanel } from '../JobsPanel';
import { EntityListView } from './EntityListView';
import { FactListView } from './FactListView';
import { StrategyListView } from './StrategyListView';
import { EntityDetail } from './EntityDetail';
import { FactDetail } from './FactDetail';
import { StrategyDetail } from './StrategyDetail';
import { MemoryGraphView } from './MemoryGraphView';
import { UserHistoryTab } from './UserHistoryTab';
import { Pagination } from './Pagination';
import type { MemorySection } from './types';
import '../../styles/MemoryPanel.css';
import { Button } from '../ui';

export const MemoryPanel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<MemorySection>('entities');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [confidenceFilter, setConfidenceFilter] = useState(0);

  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedFact, setSelectedFact] = useState<MemoryFact | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<MemoryStrategy | null>(null);

  const [consolidateMessage, setConsolidateMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const { stats, loading: statsLoading, refresh: refreshStats } = useMemoryStats();
  const { consolidate, loading: consolidating } = useConsolidate();
  const { mutate: exportMemory, loading: exporting } = useExportMemory();
  const { mutate: importMemory, loading: importing } = useImportMemory();
  const { notifySuccess, notifyError } = useNotify();

  const fileInputRef = useRef<HTMLInputElement>(null);
  // Staged import: file parsed, awaiting mode choice + confirm (replace is destructive).
  const [pendingImport, setPendingImport] = useState<{ data: MemoryExport; fileName: string } | null>(null);
  const [importMode, setImportMode] = useState<'merge' | 'replace'>('merge');

  const {
    entities, total: entitiesTotal, hasNext: entitiesHasNext,
    loading: entitiesLoading, error: entitiesError, refresh: refreshEntities,
  } = useMemoryEntities('_all', currentPage, searchQuery);

  const {
    facts, total: factsTotal, hasNext: factsHasNext,
    loading: factsLoading, error: factsError, refresh: refreshFacts,
  } = useMemoryFacts('_all', currentPage, confidenceFilter / 100, searchQuery);

  const {
    strategies, total: strategiesTotal, hasNext: strategiesHasNext,
    loading: strategiesLoading, error: strategiesError,
  } = useMemoryStrategies('_all', currentPage);

  const hasNext = useMemo(() => {
    switch (activeSection) {
      case 'entities': return entitiesHasNext ?? false;
      case 'facts': return factsHasNext ?? false;
      case 'strategies': return strategiesHasNext ?? false;
      default: return false;
    }
  }, [activeSection, entitiesHasNext, factsHasNext, strategiesHasNext]);

  const selectedId = selectedEntityId ?? selectedFact?.id ?? selectedStrategy?.id ?? null;

  const memorySections = [
    { id: 'entities' as const, label: 'Entities', icon: <Users size={18} /> },
    { id: 'facts' as const, label: 'Facts', icon: <FileText size={18} /> },
    { id: 'strategies' as const, label: 'Strategies', icon: <Zap size={18} /> },
    { id: 'graph' as const, label: 'Graph', icon: <GitBranch size={18} /> },
    { id: 'user-history' as const, label: 'User History', icon: <History size={18} /> },
    { id: 'jobs' as const, label: 'Jobs', icon: <Clock size={18} /> },
  ];

  const handleSectionChange = (section: MemorySection) => {
    setActiveSection(section);
    setCurrentPage(1);
    setSelectedEntityId(null);
    setSelectedFact(null);
    setSelectedStrategy(null);
    setSearchQuery('');
  };

  const handleConsolidate = async () => {
    try {
      const result = await consolidate();
      const totalEntities = result.results?.consolidate?.entities ?? 0;
      const totalFacts = result.results?.consolidate?.facts ?? 0;
      const totalRelationships = result.results?.consolidate?.relationships ?? 0;
      setConsolidateMessage({
        type: 'success',
        text: `Extracted ${totalEntities} entities, ${totalFacts} facts, ${totalRelationships} relationships`,
      });
      refreshStats();
      setTimeout(() => setConsolidateMessage(null), 5000);
    } catch (err) {
      setConsolidateMessage({ type: 'error', text: `Consolidation failed: ${(err as Error).message}` });
      setTimeout(() => setConsolidateMessage(null), 5000);
    }
  };

  const handleExport = async () => {
    const data = await exportMemory({ channel: '_all' });
    if (!data) {
      notifyError('Could not export memory', 'Export failed');
      return;
    }
    downloadJson(data, `agentx-memory-${fileTimestamp()}.json`);
    notifySuccess('Memory snapshot downloaded', 'Export complete');
  };

  const handleFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow re-picking the same file
    if (!file) return;
    try {
      const data = await readJsonFile<MemoryExport>(file);
      if (!data || typeof data.schema_version !== 'number') {
        notifyError('That file is not a memory export', 'Invalid file');
        return;
      }
      setImportMode('merge');
      setPendingImport({ data, fileName: file.name });
    } catch (err) {
      notifyError(err as Error, 'Could not read file');
    }
  };

  const handleConfirmImport = async () => {
    if (!pendingImport) return;
    const result = await importMemory(pendingImport.data, importMode);
    if (!result) {
      notifyError('Import failed', 'Import error');
      return;
    }
    const created = Object.values(result.imported).reduce((sum, c) => sum + c.created, 0);
    notifySuccess(
      `${importMode === 'replace' ? 'Replaced' : 'Merged'} memory — ${created} new node(s)` +
        (result.recomputed_embeddings ? `, ${result.recomputed_embeddings} embeddings recomputed` : ''),
      'Import complete',
    );
    setPendingImport(null);
    refreshStats();
    refreshEntities();
    refreshFacts();
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
            <Button
              variant="primary" className="consolidate-button"
              onClick={handleConsolidate}
              disabled={consolidating}
              title="Run consolidation to extract entities and facts"
            >
              {consolidating
                ? <><RefreshCw size={16} className="spin" /> Consolidating...</>
                : <><Zap size={16} /> Consolidate Now</>}
            </Button>
            <Button
              variant="ghost" onClick={handleExport} disabled={exporting}
              title="Download a round-trippable JSON snapshot of all memory"
            >
              {exporting ? <RefreshCw size={16} className="spin" /> : <Download size={16} />} Export
            </Button>
            <Button
              variant="ghost" onClick={() => fileInputRef.current?.click()} disabled={importing}
              title="Import a memory snapshot from a JSON file"
            >
              <Upload size={16} /> Import
            </Button>
            <Button variant="ghost" onClick={refreshStats} disabled={statsLoading}>
              <RefreshCw size={18} className={statsLoading ? 'spin' : ''} />
            </Button>
          </div>
        </div>
        <p className="page-subtitle">Browse and inspect stored memories</p>

        <input
          ref={fileInputRef} type="file" accept="application/json,.json"
          onChange={handleFilePicked} style={{ display: 'none' }}
        />

        {pendingImport && (
          <div className={`import-confirm ${importMode === 'replace' ? 'destructive' : ''}`}>
            <div className="import-confirm-head">
              <Upload size={16} />
              <span>Import <strong>{pendingImport.fileName}</strong></span>
              <button className="dismiss-btn" onClick={() => setPendingImport(null)} title="Cancel">
                <X size={14} />
              </button>
            </div>
            <div className="import-mode-toggle">
              <button
                className={importMode === 'merge' ? 'active' : ''}
                onClick={() => setImportMode('merge')}
              >
                Merge <span>upsert; keeps existing</span>
              </button>
              <button
                className={importMode === 'replace' ? 'active' : ''}
                onClick={() => setImportMode('replace')}
              >
                Replace <span>wipe channel first</span>
              </button>
            </div>
            {importMode === 'replace' && (
              <p className="import-warning">
                <AlertTriangle size={14} /> Replace deletes the snapshot's channel(s) for this
                user before importing. This cannot be undone.
              </p>
            )}
            <div className="import-confirm-actions">
              <Button
                variant={importMode === 'replace' ? 'danger' : 'primary'}
                onClick={handleConfirmImport} disabled={importing}
              >
                {importing
                  ? <><RefreshCw size={14} className="spin" /> Importing…</>
                  : <><Check size={14} /> {importMode === 'replace' ? 'Replace & Import' : 'Merge Import'}</>}
              </Button>
              <Button variant="ghost" onClick={() => setPendingImport(null)} disabled={importing}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        {consolidateMessage && (
          <div className={`consolidate-message ${consolidateMessage.type}`}>
            {consolidateMessage.type === 'success' ? <Zap size={16} /> : <X size={16} />}
            <span>{consolidateMessage.text}</span>
            <button className="dismiss-btn" onClick={() => setConsolidateMessage(null)}><X size={14} /></button>
          </div>
        )}

        <div className="memory-stats-bar">
          <span className="stat-badge"><Users size={14} /> {stats?.totals.entities ?? 0} Entities</span>
          <span className="stat-badge"><FileText size={14} /> {stats?.totals.facts ?? 0} Facts</span>
          <span className="stat-badge"><Zap size={14} /> {stats?.totals.strategies ?? 0} Strategies</span>
        </div>
      </div>

      <div className="memory-layout">
        {/* Sidebar Navigation */}
        <nav className="memory-nav card">
          {memorySections.map(section => (
            <button
              key={section.id}
              className={`nav-item${activeSection === section.id ? ' active' : ''}`}
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
            <div className="memory-list-container card"><JobsPanel /></div>
          ) : activeSection === 'graph' ? (
            <div className="memory-list-container card full-height"><MemoryGraphView /></div>
          ) : activeSection === 'user-history' ? (
            <UserHistoryTab />
          ) : (
            <div className={`memory-split${selectedId ? ' detail-open' : ''}`}>
              {/* Left: filters + list + pagination */}
              <div className="split-list">
                <div className="memory-filters card">
                  <div className="filter-group search always-visible">
                    <Search size={16} />
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                    />
                  </div>
                  {activeSection === 'facts' && (
                    <div className="filter-group confidence">
                      <label>Min Confidence: {confidenceFilter}%</label>
                      <input
                        type="range" min="0" max="100" value={confidenceFilter}
                        onChange={e => { setConfidenceFilter(Number(e.target.value)); setCurrentPage(1); }}
                      />
                    </div>
                  )}
                </div>

                <div className="memory-list-container card">
                  {activeSection === 'entities' && (
                    <EntityListView
                      entities={entities}
                      total={entitiesTotal ?? 0}
                      loading={entitiesLoading}
                      error={entitiesError}
                      selectedEntityId={selectedEntityId}
                      onSelectEntity={setSelectedEntityId}
                    />
                  )}
                  {activeSection === 'facts' && (
                    <FactListView
                      facts={facts}
                      total={factsTotal ?? 0}
                      loading={factsLoading}
                      error={factsError}
                      selectedFactId={selectedFact?.id ?? null}
                      onSelectFact={setSelectedFact}
                    />
                  )}
                  {activeSection === 'strategies' && (
                    <StrategyListView
                      strategies={strategies}
                      total={strategiesTotal ?? 0}
                      loading={strategiesLoading}
                      error={strategiesError}
                      selectedStrategyId={selectedStrategy?.id ?? null}
                      onSelectStrategy={setSelectedStrategy}
                    />
                  )}
                </div>

                <Pagination page={currentPage} hasNext={hasNext} onPageChange={setCurrentPage} />
              </div>

              {/* Right: detail panel */}
              <div className={`split-detail${selectedId ? ' is-open' : ''}`}>
                {activeSection === 'entities' && selectedEntityId ? (
                  <EntityDetail
                    entityId={selectedEntityId}
                    onClose={() => setSelectedEntityId(null)}
                    onDeleted={() => { setSelectedEntityId(null); refreshEntities(); }}
                    onRefreshList={refreshEntities}
                  />
                ) : activeSection === 'facts' && selectedFact ? (
                  <FactDetail
                    fact={selectedFact}
                    onClose={() => setSelectedFact(null)}
                    onDeleted={() => { setSelectedFact(null); refreshFacts(); }}
                    onUpdated={updated => { setSelectedFact(updated); refreshFacts(); }}
                  />
                ) : activeSection === 'strategies' && selectedStrategy ? (
                  <StrategyDetail
                    strategy={selectedStrategy}
                    onClose={() => setSelectedStrategy(null)}
                  />
                ) : (
                  <div className="detail-placeholder">
                    <ChevronRight size={32} />
                    <p>Select an item to view details</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MemoryPanel;

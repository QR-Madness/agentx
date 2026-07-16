/**
 * MemoryWorkbench — full-screen, immersive Memory explorer.
 *
 * Replaces the old xxl-drawer `MemoryPanel` (200px rail + single canvas) with a
 * bare `inset:0` surface (the UnifiedSettings pattern — registered in
 * FULLSCREEN_SURFACES so it owns its own backdrop + close) laid out as a top tab
 * bar over a list+detail canvas per area. Channel scoping, an Overview home, and
 * a Procedures area surface data the old panel never showed; the existing
 * editors (EntityDetail / FactDetail) are reused, re-hosted in a roomy detail
 * pane. Mobile collapses the split to one pane at a time.
 */

import { useState, useRef, useEffect } from 'react';
import {
  LayoutDashboard, Users, FileText, Zap, ListChecks, GitBranch, History, Clock,
  Search, RefreshCw, ChevronLeft, X, Download, Upload, AlertTriangle, Check,
} from 'lucide-react';
import { MemoryIcon } from '../common/MemoryIcon';
import {
  useMemoryEntities, useMemoryFacts, useMemoryStrategies, useMemoryProcedures,
  useMemoryStats, useMemoryChannels, useConsolidate, useExportMemory, useImportMemory,
} from '../../lib/hooks';
import { useIsMobile } from '../../lib/hooks';
import type { MemoryExport, MemoryFact, MemoryStrategy, MemoryProcedure } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { downloadJson, readJsonFile, fileTimestamp } from '../../lib/fileTransfer';
import { JobsPanel } from '../JobsPanel';
import { EntityListView } from './EntityListView';
import { FactListView } from './FactListView';
import { StrategyListView } from './StrategyListView';
import { ProcedureListView } from './ProcedureListView';
import { EntityDetail } from './EntityDetail';
import { FactDetail } from './FactDetail';
import { StrategyDetail } from './StrategyDetail';
import { ProcedureDetail } from './ProcedureDetail';
import { MemoryGraphView } from './MemoryGraphView';
import { UserHistoryTab } from './UserHistoryTab';
import { OverviewPanel } from './OverviewPanel';
import { Pagination } from './Pagination';
import type { MemoryArea } from './types';
import {
  Button, Input, Slider,
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from '../ui';
import '../../styles/MemoryPanel.css';
import '../../styles/MemoryWorkbench.css';

const AREAS: { id: MemoryArea; label: string; icon: React.ReactNode }[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={16} /> },
  { id: 'entities', label: 'Entities', icon: <Users size={16} /> },
  { id: 'facts', label: 'Facts', icon: <FileText size={16} /> },
  { id: 'strategies', label: 'Strategies', icon: <Zap size={16} /> },
  { id: 'procedures', label: 'Procedures', icon: <ListChecks size={16} /> },
  { id: 'explore', label: 'Explore', icon: <GitBranch size={16} /> },
  { id: 'user-history', label: 'History', icon: <History size={16} /> },
  { id: 'jobs', label: 'Jobs', icon: <Clock size={16} /> },
];

const ALL_CHANNELS = '_all';

export function MemoryWorkbench({ onClose }: { onClose: () => void }) {
  const isMobile = useIsMobile();
  const [activeArea, setActiveArea] = useState<MemoryArea>('overview');
  const [channel, setChannel] = useState<string>(ALL_CHANNELS);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [confidenceFilter, setConfidenceFilter] = useState(0);

  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedFact, setSelectedFact] = useState<MemoryFact | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<MemoryStrategy | null>(null);
  const [selectedProcedure, setSelectedProcedure] = useState<MemoryProcedure | null>(null);

  const [consolidateMessage, setConsolidateMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const { stats, loading: statsLoading, refresh: refreshStats } = useMemoryStats();
  const { channels, refresh: refreshChannels } = useMemoryChannels();
  const { consolidate, loading: consolidating } = useConsolidate();
  const { mutate: exportMemory, loading: exporting } = useExportMemory();
  const { mutate: importMemory, loading: importing } = useImportMemory();
  const { notifySuccess, notifyError } = useNotify();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingImport, setPendingImport] = useState<{ data: MemoryExport; fileName: string } | null>(null);
  const [importMode, setImportMode] = useState<'merge' | 'replace'>('merge');

  const {
    entities, total: entitiesTotal, hasNext: entitiesHasNext,
    loading: entitiesLoading, error: entitiesError, refresh: refreshEntities,
  } = useMemoryEntities(channel, currentPage, searchQuery);

  const {
    facts, total: factsTotal, hasNext: factsHasNext,
    loading: factsLoading, error: factsError, refresh: refreshFacts,
  } = useMemoryFacts(channel, currentPage, confidenceFilter / 100, searchQuery);

  const {
    strategies, total: strategiesTotal, hasNext: strategiesHasNext,
    loading: strategiesLoading, error: strategiesError,
  } = useMemoryStrategies(channel, currentPage);

  const {
    procedures, total: proceduresTotal, hasNext: proceduresHasNext,
    loading: proceduresLoading, error: proceduresError,
  } = useMemoryProcedures(channel, currentPage);

  // ESC to close + body scroll lock (bare full-screen surface owns these).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handler);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const clearSelection = () => {
    setSelectedEntityId(null);
    setSelectedFact(null);
    setSelectedStrategy(null);
    setSelectedProcedure(null);
  };

  const handleAreaChange = (area: MemoryArea) => {
    setActiveArea(area);
    setCurrentPage(1);
    setSearchQuery('');
    setConfidenceFilter(0);
    clearSelection();
  };

  const handleChannelChange = (next: string) => {
    setChannel(next);
    setCurrentPage(1);
    clearSelection();
  };

  const openChannelEntities = (next: string) => {
    setChannel(next);
    setCurrentPage(1);
    setSearchQuery('');
    clearSelection();
    setActiveArea('entities');
  };

  const handleConsolidate = async () => {
    try {
      const result = await consolidate();
      const e = result.results?.consolidate?.entities ?? 0;
      const f = result.results?.consolidate?.facts ?? 0;
      const r = result.results?.consolidate?.relationships ?? 0;
      setConsolidateMessage({ type: 'success', text: `Extracted ${e} entities, ${f} facts, ${r} relationships` });
      refreshStats();
      refreshChannels();
      setTimeout(() => setConsolidateMessage(null), 5000);
    } catch (err) {
      setConsolidateMessage({ type: 'error', text: `Consolidation failed: ${(err as Error).message}` });
      setTimeout(() => setConsolidateMessage(null), 5000);
    }
  };

  const handleExport = async () => {
    const data = await exportMemory({ channel });
    if (!data) { notifyError('Could not export memory', 'Export failed'); return; }
    downloadJson(data, `agentx-memory-${fileTimestamp()}.json`);
    notifySuccess('Memory snapshot downloaded', 'Export complete');
  };

  const handleFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
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
    if (!result) { notifyError('Import failed', 'Import error'); return; }
    const created = Object.values(result.imported).reduce((sum, c) => sum + c.created, 0);
    notifySuccess(
      `${importMode === 'replace' ? 'Replaced' : 'Merged'} memory — ${created} new node(s)` +
        (result.recomputed_embeddings ? `, ${result.recomputed_embeddings} embeddings recomputed` : ''),
      'Import complete',
    );
    setPendingImport(null);
    refreshStats();
    refreshChannels();
    refreshEntities();
    refreshFacts();
  };

  // ── Master-detail (desktop: side-by-side; mobile: one pane at a time) ──
  const renderSplit = (hasSelection: boolean, master: React.ReactNode, detail: React.ReactNode) => {
    if (isMobile) {
      if (hasSelection) {
        return (
          <div className="mem-detail is-mobile">
            <button type="button" className="mem-back" onClick={clearSelection}>
              <ChevronLeft size={16} /> Back
            </button>
            {detail}
          </div>
        );
      }
      return <div className="mem-master is-mobile">{master}</div>;
    }
    return (
      <div className={`mem-split${hasSelection ? ' detail-open' : ''}`}>
        <div className="mem-master">{master}</div>
        {hasSelection && <div className="mem-detail">{detail}</div>}
      </div>
    );
  };

  const searchField = (placeholder: string) => (
    <div className="mem-filters">
      <Input
        icon={<Search size={16} />}
        placeholder={placeholder}
        value={searchQuery}
        onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
      />
    </div>
  );

  const renderArea = () => {
    switch (activeArea) {
      case 'overview':
        return <OverviewPanel stats={stats} loading={statsLoading} onOpenChannel={openChannelEntities} />;

      case 'explore':
        return <div className="mem-fill"><MemoryGraphView channel={channel} /></div>;

      case 'user-history':
        return <div className="mem-fill"><UserHistoryTab /></div>;

      case 'jobs':
        return <div className="mem-fill mem-scroll"><JobsPanel /></div>;

      case 'entities':
        return renderSplit(
          !!selectedEntityId,
          <>
            {searchField('Search entities…')}
            <div className="memory-list-container card">
              <EntityListView
                entities={entities} total={entitiesTotal ?? 0}
                loading={entitiesLoading} error={entitiesError}
                selectedEntityId={selectedEntityId} onSelectEntity={setSelectedEntityId}
              />
            </div>
            <Pagination page={currentPage} hasNext={entitiesHasNext ?? false} onPageChange={setCurrentPage} />
          </>,
          selectedEntityId && (
            <EntityDetail
              entityId={selectedEntityId}
              onClose={clearSelection}
              onDeleted={() => { clearSelection(); refreshEntities(); refreshStats(); }}
              onRefreshList={refreshEntities}
            />
          ),
        );

      case 'facts':
        return renderSplit(
          !!selectedFact,
          <>
            <div className="mem-filters">
              <Input
                icon={<Search size={16} />}
                placeholder="Search facts…"
                value={searchQuery}
                onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
              />
              <div className="mem-confidence">
                <label>Min confidence: {confidenceFilter}%</label>
                <Slider
                  aria-label="Minimum confidence"
                  min={0} max={100} step={1}
                  value={[confidenceFilter]}
                  onValueChange={([v]) => { setConfidenceFilter(v); setCurrentPage(1); }}
                />
              </div>
            </div>
            <div className="memory-list-container card">
              <FactListView
                facts={facts} total={factsTotal ?? 0}
                loading={factsLoading} error={factsError}
                selectedFactId={selectedFact?.id ?? null} onSelectFact={setSelectedFact}
              />
            </div>
            <Pagination page={currentPage} hasNext={factsHasNext ?? false} onPageChange={setCurrentPage} />
          </>,
          selectedFact && (
            <FactDetail
              fact={selectedFact}
              onClose={clearSelection}
              onDeleted={() => { clearSelection(); refreshFacts(); refreshStats(); }}
              onUpdated={updated => { setSelectedFact(updated); refreshFacts(); }}
              onNavigateEntity={id => { clearSelection(); setActiveArea('entities'); setSelectedEntityId(id); }}
            />
          ),
        );

      case 'strategies':
        return renderSplit(
          !!selectedStrategy,
          <>
            <div className="memory-list-container card">
              <StrategyListView
                strategies={strategies} total={strategiesTotal ?? 0}
                loading={strategiesLoading} error={strategiesError}
                selectedStrategyId={selectedStrategy?.id ?? null} onSelectStrategy={setSelectedStrategy}
              />
            </div>
            <Pagination page={currentPage} hasNext={strategiesHasNext ?? false} onPageChange={setCurrentPage} />
          </>,
          selectedStrategy && <StrategyDetail strategy={selectedStrategy} onClose={clearSelection} />,
        );

      case 'procedures':
        return renderSplit(
          !!selectedProcedure,
          <>
            <div className="memory-list-container card">
              <ProcedureListView
                procedures={procedures} total={proceduresTotal ?? 0}
                loading={proceduresLoading} error={proceduresError}
                selectedProcedureId={selectedProcedure?.id ?? null} onSelectProcedure={setSelectedProcedure}
              />
            </div>
            <Pagination page={currentPage} hasNext={proceduresHasNext ?? false} onPageChange={setCurrentPage} />
          </>,
          selectedProcedure && <ProcedureDetail procedure={selectedProcedure} onClose={clearSelection} />,
        );

      default:
        return null;
    }
  };

  return (
    <>
      <div className="mem-workbench-backdrop" onClick={onClose} />
      <div className="mem-workbench" role="dialog" aria-modal="true" aria-label="Memory Explorer">
        {/* Header */}
        <header className="mem-header">
          <div className="mem-title">
            <MemoryIcon size={16} className="mem-title-icon" />
            <span>Memory</span>
          </div>
          <div className="mem-header-actions">
            <Select value={channel} onValueChange={handleChannelChange}>
              <SelectTrigger className="mem-channel-select" aria-label="Memory channel">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_CHANNELS}>All channels</SelectItem>
                {channels.map(c => (
                  <SelectItem key={c.name} value={c.name}>
                    {c.name} · {c.item_counts.facts} facts
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="primary" onClick={handleConsolidate} disabled={consolidating}
              title="Run consolidation to extract entities and facts">
              {consolidating
                ? <><RefreshCw size={16} className="spin" /> Consolidating…</>
                : <><Zap size={16} /> Consolidate</>}
            </Button>
            <Button variant="ghost" onClick={handleExport} disabled={exporting}
              title="Download a round-trippable JSON snapshot">
              {exporting ? <RefreshCw size={16} className="spin" /> : <Download size={16} />}
              <span className="mem-action-label"> Export</span>
            </Button>
            <Button variant="ghost" onClick={() => fileInputRef.current?.click()} disabled={importing}
              title="Import a memory snapshot from JSON">
              <Upload size={16} /><span className="mem-action-label"> Import</span>
            </Button>
            <Button variant="ghost" onClick={() => { refreshStats(); refreshChannels(); refreshEntities(); refreshFacts(); }}
              disabled={statsLoading} title="Refresh">
              <RefreshCw size={18} className={statsLoading ? 'spin' : ''} />
            </Button>
            <Button variant="ghost" onClick={onClose} title="Close" aria-label="Close Memory">
              <X size={20} />
            </Button>
          </div>
        </header>

        <input ref={fileInputRef} type="file" accept="application/json,.json"
          onChange={handleFilePicked} style={{ display: 'none' }} />

        {/* Tab bar */}
        <nav className="mem-tabs" aria-label="Memory areas">
          {AREAS.map(area => (
            <button
              key={area.id}
              type="button"
              className={`mem-tab${activeArea === area.id ? ' active' : ''}`}
              onClick={() => handleAreaChange(area.id)}
            >
              {area.icon}
              <span>{area.label}</span>
            </button>
          ))}
        </nav>

        {/* Transient banners */}
        {(pendingImport || consolidateMessage) && (
          <div className="mem-banners">
            {pendingImport && (
              <div className={`import-confirm ${importMode === 'replace' ? 'destructive' : ''}`}>
                <div className="import-confirm-head">
                  <Upload size={16} />
                  <span>Import <strong>{pendingImport.fileName}</strong></span>
                  <button className="dismiss-btn" onClick={() => setPendingImport(null)} title="Cancel"><X size={14} /></button>
                </div>
                <div className="import-mode-toggle">
                  <button className={importMode === 'merge' ? 'active' : ''} onClick={() => setImportMode('merge')}>
                    Merge <span>upsert; keeps existing</span>
                  </button>
                  <button className={importMode === 'replace' ? 'active' : ''} onClick={() => setImportMode('replace')}>
                    Replace <span>wipe channel first</span>
                  </button>
                </div>
                {importMode === 'replace' && (
                  <p className="import-warning">
                    <AlertTriangle size={14} /> Replace deletes the snapshot's channel(s) for this user before importing. This cannot be undone.
                  </p>
                )}
                <div className="import-confirm-actions">
                  <Button variant={importMode === 'replace' ? 'danger' : 'primary'} onClick={handleConfirmImport} disabled={importing}>
                    {importing
                      ? <><RefreshCw size={14} className="spin" /> Importing…</>
                      : <><Check size={14} /> {importMode === 'replace' ? 'Replace & Import' : 'Merge Import'}</>}
                  </Button>
                  <Button variant="ghost" onClick={() => setPendingImport(null)} disabled={importing}>Cancel</Button>
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
          </div>
        )}

        {/* Active area */}
        <div className="mem-body">{renderArea()}</div>
      </div>
    </>
  );
}

export default MemoryWorkbench;

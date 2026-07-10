/**
 * ConnectorCatalogView — the "add real connectors" shelf of Connectors & Tools:
 * a curated catalog (lib/connectorCatalog.ts) plus live search of the official
 * MCP registry. Both paths converge on the same ServerForm, prefilled via its
 * `initialDraft` prop, so the user always reviews before anything is created.
 */
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Loader2, Plus, Search } from 'lucide-react';
import { api, apiErrorMessage } from '../../lib/api';
import type { MCPRegistryResult } from '../../lib/api';
import { useMCPServers } from '../../lib/hooks';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { Button, Input } from '../ui';
import {
  AUTH_KIND_LABELS, CATALOG_CATEGORIES, CONNECTOR_CATALOG,
  catalogEntryConfigured, draftFromCatalogEntry, draftFromRegistryResult,
  type ServerDraft,
} from '../../lib/connectorCatalog';
import { ServerForm } from './ServerForm';

export function ConnectorCatalogView() {
  const { servers, createServer, validateServer } = useMCPServers();
  const { profiles } = useAgentProfile();
  const [draft, setDraft] = useState<ServerDraft | null>(null);

  // Registry search — debounced; empty query clears results back to the shelf.
  const [q, setQ] = useState('');
  const [results, setResults] = useState<MCPRegistryResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  useEffect(() => {
    const query = q.trim();
    if (!query) { setResults(null); setSearchError(null); setSearching(false); return; }
    let cancelled = false;
    setSearching(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await api.searchMCPRegistry(query);
        if (!cancelled) { setResults(r.results); setSearchError(null); }
      } catch (err) {
        if (!cancelled) { setResults([]); setSearchError(apiErrorMessage(err)); }
      } finally {
        if (!cancelled) setSearching(false);
      }
    }, 450);
    return () => { cancelled = true; window.clearTimeout(t); };
  }, [q]);

  const searchActive = q.trim().length > 0;

  return (
    <>
      <p className="toolkit-section-note">
        Known-good connectors with guided setup — or search the public MCP registry.
        Adding opens a prefilled form; nothing connects until you review it.
      </p>
      <Input
        icon={<Search size={16} />}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search the MCP registry…"
      />

      {searchActive ? (
        /* ── Registry results ── */
        searching && !results ? (
          <div className="toolkit-empty"><Loader2 className="spin" /> Searching the registry…</div>
        ) : searchError ? (
          <div className="toolkit-error-banner"><span>{searchError}</span></div>
        ) : !results || results.length === 0 ? (
          <div className="toolkit-empty">No registry entries match “{q.trim()}”.</div>
        ) : (
          <div className="toolkit-registry-list">
            {results.map(r => (
              <div key={r.name} className="toolkit-registry-row">
                <div className="info">
                  <span className="name">{r.name}</span>
                  <span className="desc">{r.description || 'No description'}</span>
                </div>
                <span className="toolkit-chip">
                  {r.remotes.length > 0 ? 'remote' : r.packages[0]?.registry_type ?? 'package'}
                </span>
                <Button variant="secondary" size="sm" onClick={() => setDraft(draftFromRegistryResult(r))}>
                  <Plus size={14} /> Add
                </Button>
              </div>
            ))}
          </div>
        )
      ) : (
        /* ── Curated shelf ── */
        CATALOG_CATEGORIES.map(cat => {
          const entries = CONNECTOR_CATALOG.filter(e => e.category === cat.id);
          if (entries.length === 0) return null;
          return (
            <div key={cat.id}>
              <div className="toolkit-catalog-category">{cat.label}</div>
              <div className="toolkit-card-grid">
                {entries.map(entry => {
                  const configured = catalogEntryConfigured(entry, servers);
                  return (
                    <div key={entry.id} className="toolkit-card">
                      <div className="toolkit-card-header">
                        <span
                          className="toolkit-brand-tile"
                          style={{ background: `${entry.brand.color}26`, color: entry.brand.color }}
                        >
                          {entry.brand.initial}
                        </span>
                        <span className="name" title={entry.title}>{entry.title}</span>
                        <span className="toolkit-chip toolkit-catalog-badge">{AUTH_KIND_LABELS[entry.authKind]}</span>
                      </div>
                      <div className="meta" style={{ color: 'var(--text-secondary)' }}>
                        {entry.description}
                      </div>
                      <div className="toolkit-card-actions">
                        {configured ? (
                          <Button variant="ghost" size="sm" disabled>
                            <Check size={14} /> Added
                          </Button>
                        ) : (
                          <Button variant="primary" size="sm" onClick={() => setDraft(draftFromCatalogEntry(entry))}>
                            <Plus size={14} /> Add connector
                          </Button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })
      )}

      {draft && createPortal(
        <div className="toolkit-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setDraft(null); }}>
          <div className="toolkit-modal-card">
            <h3>Add {draft.name ?? 'MCP server'}</h3>
            <ServerForm
              initialDraft={draft}
              agentProfiles={profiles}
              onCancel={() => setDraft(null)}
              onSubmit={async (name, cfg) => { await createServer(name, cfg); }}
              onValidate={validateServer}
            />
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

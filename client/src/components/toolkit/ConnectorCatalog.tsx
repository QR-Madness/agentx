/**
 * ConnectorCatalogView — the "add real connectors" shelf of Connectors & Tools.
 *
 * The shelf is a grid of compact clickable tiles (brand + name + status pip —
 * no per-tile action buttons). Each tile opens ONE **connector dialog** that
 * covers the whole lifecycle: not-added entries get the guided quick-add
 * (setup steps + only the fields that entry needs; OAuth chains straight into
 * browser sign-in), added entries show live status + Connect, and gated
 * entries explain their "Coming soon". "Open full form" is the escape hatch
 * to the complete ServerForm. Registry search results always open the full
 * form — they're untrusted and deserve review.
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Clock, ExternalLink, Loader2, Plus, Search } from 'lucide-react';
import { api, apiErrorMessage } from '../../lib/api';
import type { MCPRegistryResult } from '../../lib/api';
import type { McpServersState } from '../../lib/hooks';
import { useNotify } from '../../contexts/NotificationContext';
import { openExternal } from '../../lib/openExternal';
import { Button, Input, StatusDot } from '../ui';
import {
  AUTH_KIND_LABELS, CATALOG_CATEGORIES, CONNECTOR_CATALOG,
  applyQuickInputs, catalogEntryConfigured, draftFromCatalogEntry, draftFromRegistryResult,
  findConfiguredServer,
  type CatalogEntry, type ServerDraft,
} from '../../lib/connectorCatalog';
import { ServerForm } from './ServerForm';
import { useAgentProfile } from '../../contexts/AgentProfileContext';

export function ConnectorCatalogView({ mcp }: { mcp: McpServersState }) {
  const { servers } = mcp;
  const [quickAdd, setQuickAdd] = useState<CatalogEntry | null>(null);
  // Full-form drafts (registry results + the quick-add escape hatch).
  const [formDraft, setFormDraft] = useState<ServerDraft | null>(null);

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
      </p>
      <Input
        icon={<Search size={16} />}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search the MCP registry…"
      />

      {searchActive ? (
        /* ── Registry results (untrusted → always the full review form) ── */
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
                <Button variant="secondary" size="sm" onClick={() => setFormDraft(draftFromRegistryResult(r))}>
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
                    <button
                      key={entry.id}
                      type="button"
                      className={`toolkit-connector-tile${entry.comingSoon ? ' toolkit-connector-tile--soon' : ''}`}
                      title={entry.comingSoon ?? entry.description}
                      onClick={() => setQuickAdd(entry)}
                    >
                      <span
                        className="toolkit-brand-tile"
                        style={{ background: `${entry.brand.color}26`, color: entry.brand.color }}
                      >
                        {entry.brand.initial}
                      </span>
                      <span className="toolkit-connector-tile-info">
                        <span className="name">{entry.title}</span>
                        <span className="desc">{entry.description}</span>
                      </span>
                      {entry.comingSoon ? (
                        <span className="toolkit-connector-tile-status soon">
                          <Clock size={12} /> Soon
                        </span>
                      ) : configured ? (
                        <span className="toolkit-connector-tile-status added">
                          <Check size={12} /> Added
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })
      )}

      {quickAdd && (
        <ConnectorDialog
          entry={quickAdd}
          mcp={mcp}
          onClose={() => setQuickAdd(null)}
          onOpenFullForm={() => {
            setFormDraft(draftFromCatalogEntry(quickAdd));
            setQuickAdd(null);
          }}
        />
      )}

      {formDraft && createPortal(
        <div className="toolkit-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setFormDraft(null); }}>
          <div className="toolkit-modal-card">
            <h3>Add {formDraft.name ?? 'MCP server'}</h3>
            <ServerFormHost draft={formDraft} mcp={mcp} onClose={() => setFormDraft(null)} />
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

/** The full ServerForm wired to the shared server state (escape hatch + registry adds). */
function ServerFormHost({
  draft, mcp, onClose,
}: {
  draft: ServerDraft;
  mcp: McpServersState;
  onClose: () => void;
}) {
  const { profiles } = useAgentProfile();
  return (
    <ServerForm
      initialDraft={draft}
      agentProfiles={profiles}
      onCancel={onClose}
      onSubmit={async (name, cfg) => { await mcp.createServer(name, cfg); }}
      onValidate={mcp.validateServer}
    />
  );
}

/* ── Quick add ─────────────────────────────────────────────── */

/**
 * One modal per connector, covering its whole lifecycle:
 *   · not added   — setup steps + only the fields the entry declares; OAuth
 *     entries add-and-sign-in in one action (create → connect → browser
 *     consent; the shared poll flips it when the round-trip lands)
 *   · added       — live status for the backing server + Connect / sign-in
 *     when it's down (full management stays in the Servers section)
 *   · coming soon — the reason it's gated + docs, no add action
 * Dismissing while a sign-in waits cancels it server-side, mirroring
 * ServersView.
 */
function ConnectorDialog({
  entry, mcp, onClose, onOpenFullForm,
}: {
  entry: CatalogEntry;
  mcp: McpServersState;
  onClose: () => void;
  onOpenFullForm: () => void;
}) {
  const { notify } = useNotify();
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries((entry.inputs ?? [])
      .filter(i => i.defaultValue != null)
      .map(i => [i.key, i.defaultValue as string])),
  );
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<'idle' | 'saving' | 'waiting'>('idle');
  const isOauth = entry.config.auth?.type === 'oauth';
  const createdRef = useRef(false);

  // The server already backing this entry (URL/command match — its name may
  // differ from the suggested one). Live: mcp.servers is shared state.
  const existing = findConfiguredServer(entry, mcp.servers);
  const serverName = existing?.name ?? entry.serverName;
  const soon = !existing && !!entry.comingSoon;

  // While waiting on browser consent, poll shared state; stop when the server
  // connects or errors (mirrors ServersView's authWait loop).
  useEffect(() => {
    if (phase !== 'waiting') return;
    const started = Date.now();
    const t = window.setInterval(() => {
      void mcp.refresh();
      if (Date.now() - started > 5 * 60_000) setPhase('idle');
    }, 2500);
    return () => window.clearInterval(t);
  }, [phase, mcp]);
  useEffect(() => {
    if (phase !== 'waiting') return;
    const s = mcp.servers.find(x => x.name === serverName);
    if (s?.status === 'connected') {
      notify({ kind: 'success', title: `${entry.title} connected`, message: `${s.tools_count ?? 0} tools available.` });
      onClose();
    } else if (s?.auth_state?.error) {
      setError(`Sign-in failed: ${s.auth_state.error}`);
      setPhase('idle');
    }
  }, [phase, mcp.servers, serverName, entry, notify, onClose]);

  const missingRequired = (entry.inputs ?? [])
    .filter(i => i.required && !(values[i.key] ?? '').trim())
    .map(i => i.label);

  /** Connect (or OAuth sign in) the named server — shared by add + manage. */
  const startConnect = async () => {
    const result = await mcp.connectServer(serverName);
    if (result?.status === 'connected') {
      notify({ kind: 'success', title: `${entry.title} connected`, message: 'Its tools are live.' });
      onClose();
    } else if (result?.status === 'auth_required' && 'authorization_url' in result && result.authorization_url) {
      void openExternal(result.authorization_url);
      setPhase('waiting');
    } else {
      setError('Connect failed — the server was added; try Connect from the Servers list.');
      setPhase('idle');
    }
  };

  const submit = async () => {
    if (missingRequired.length > 0) {
      setError(`Required: ${missingRequired.join(', ')}`);
      return;
    }
    setError(null);
    setPhase('saving');
    try {
      if (!createdRef.current) {
        await mcp.createServer(entry.serverName, {
          transport: entry.config.transport ?? 'streamable_http',
          ...applyQuickInputs(entry, values),
        });
        createdRef.current = true;
      }
      if (!isOauth) {
        notify({
          kind: 'success',
          title: `${entry.title} added`,
          message: 'Connect it from the Servers list when ready.',
        });
        onClose();
        return;
      }
      // OAuth: chain straight into the sign-in round-trip.
      await startConnect();
    } catch (err) {
      setError(apiErrorMessage(err));
      setPhase('idle');
    }
  };

  const connectExisting = async () => {
    setError(null);
    setPhase('saving');
    try {
      await startConnect();
    } catch (err) {
      setError(apiErrorMessage(err));
      setPhase('idle');
    }
  };

  const cancel = async () => {
    if (phase === 'waiting') {
      try { await api.cancelMCPServerAuth(serverName); } catch { /* best-effort */ }
      void mcp.refresh();
    }
    onClose();
  };

  const connected = existing?.status === 'connected';

  return createPortal(
    <div className="toolkit-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) void cancel(); }}>
      <div className="toolkit-modal-card toolkit-quickadd">
        <div className="toolkit-quickadd-header">
          <span
            className="toolkit-brand-tile"
            style={{ background: `${entry.brand.color}26`, color: entry.brand.color }}
          >
            {entry.brand.initial}
          </span>
          <div>
            <h3>{existing || soon ? entry.title : `Add ${entry.title}`}</h3>
            <span className="meta">{AUTH_KIND_LABELS[entry.authKind]}</span>
          </div>
          {entry.docsUrl && (
            <button
              type="button"
              className="toolkit-guidance-link"
              style={{ marginLeft: 'auto' }}
              onClick={() => void openExternal(entry.docsUrl!)}
            >
              Docs <ExternalLink size={12} />
            </button>
          )}
        </div>
        <p className="toolkit-section-note" style={{ margin: 0 }}>{entry.description}</p>

        {soon ? (
          /* ── Gated: explain why, nothing to add ── */
          <div className="toolkit-guidance">
            <div className="toolkit-guidance-title"><span><Clock size={12} /> Coming soon</span></div>
            <div className="toolkit-guidance-note">{entry.comingSoon}</div>
          </div>
        ) : existing ? (
          /* ── Added: live status + connect when down ── */
          <div className="toolkit-guidance">
            <div className="toolkit-guidance-title">
              <span>
                <StatusDot tone={connected ? 'online' : existing.auth_state?.error ? 'error' : 'inactive'} />
                {' '}Added as “{existing.name}” — {connected
                  ? `connected · ${existing.tools_count ?? 0} tools`
                  : 'not connected'}
              </span>
            </div>
            <div className="toolkit-guidance-note">
              {existing.auth_state?.error
                ? `Last sign-in error: ${existing.auth_state.error}`
                : 'Rename, tool access, reset auth, and removal live in the Servers section above.'}
            </div>
          </div>
        ) : (
          /* ── Not added yet: setup + quick-add fields ── */
          <>
            {entry.setupNote && (
              <div className="toolkit-guidance">
                <div className="toolkit-guidance-title"><span>Setup</span></div>
                <div className="toolkit-guidance-note">{entry.setupNote}</div>
              </div>
            )}

            {(entry.inputs ?? []).length > 0 && (
              <div className="toolkit-form" style={{ gap: 12 }}>
                {entry.inputs!.map(input => (
                  <label key={input.key}>
                    <span>{input.label}</span>
                    <Input
                      type={input.secret ? 'password' : 'text'}
                      value={values[input.key] ?? ''}
                      placeholder={input.placeholder}
                      onChange={(e) => setValues(v => ({ ...v, [input.key]: e.target.value }))}
                      disabled={phase !== 'idle'}
                    />
                    {input.hint && <span className="meta">{input.hint}</span>}
                  </label>
                ))}
              </div>
            )}
          </>
        )}

        {phase === 'waiting' && (
          <div className="toolkit-guidance">
            <div className="toolkit-guidance-note">
              <Loader2 size={13} className="spin" /> Finish signing in in your browser — this closes
              automatically once {entry.title} connects.
            </div>
          </div>
        )}

        {error && <div className="toolkit-error-banner"><span>{error}</span></div>}

        <div className="toolkit-modal-actions">
          <Button variant="ghost" onClick={() => void cancel()}>{soon || connected ? 'Close' : 'Cancel'}</Button>
          {!soon && !existing && (
            <>
              <Button variant="ghost" onClick={onOpenFullForm} disabled={phase !== 'idle'}>
                Open full form
              </Button>
              <Button variant="primary" onClick={() => void submit()} disabled={phase !== 'idle' || missingRequired.length > 0}>
                {phase === 'saving' ? <Loader2 size={14} className="spin" /> : <Plus size={14} />}
                {isOauth ? 'Add & sign in' : 'Add connector'}
              </Button>
            </>
          )}
          {existing && !connected && (
            <Button variant="primary" onClick={() => void connectExisting()} disabled={phase !== 'idle'}>
              {phase === 'saving' ? <Loader2 size={14} className="spin" /> : <Plus size={14} />}
              {isOauth ? 'Connect & sign in' : 'Connect'}
            </Button>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

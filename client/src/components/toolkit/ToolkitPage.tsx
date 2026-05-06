/**
 * ToolkitPage — Phase 18.2 immersive Tools menu replacement.
 *
 * Sub-views: Servers, Tools Browser, Groups & Tags, Access, Raw JSON.
 * Modeled on UnifiedSettings (animations + glass shell).
 */

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Server, Wrench, Tag, Shield, Code, Plus, Pencil, Trash2,
  Plug, Unplug, RefreshCw, Search, Loader2, AlertTriangle, Save,
} from 'lucide-react';
import { useMCPServers, useMCPTools } from '../../lib/hooks';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { ParallaxBackground } from '../unified-settings/animations/ParallaxBackground';
import { backdropVariants, containerVariants } from '../unified-settings/animations/transitions';
import type { MCPServer, MCPServerConfigInput } from '../../lib/api';
import { api } from '../../lib/api';
import { ServerForm } from './ServerForm';
import './ToolkitPage.css';

interface ToolkitPageProps {
  isOpen: boolean;
  onClose: () => void;
}

type SectionId = 'servers' | 'browser' | 'meta' | 'access' | 'raw';

const SECTIONS: { id: SectionId; label: string; icon: React.ReactNode }[] = [
  { id: 'servers', label: 'Servers', icon: <Server size={16} /> },
  { id: 'browser', label: 'Tools Browser', icon: <Wrench size={16} /> },
  { id: 'meta', label: 'Groups & Tags', icon: <Tag size={16} /> },
  { id: 'access', label: 'Access', icon: <Shield size={16} /> },
  { id: 'raw', label: 'Raw JSON', icon: <Code size={16} /> },
];

export function ToolkitPage({ isOpen, onClose }: ToolkitPageProps) {
  const [active, setActive] = useState<SectionId>('servers');

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (isOpen) document.body.style.overflow = 'hidden';
    else document.body.style.overflow = '';
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="toolkit-backdrop"
            variants={backdropVariants}
            initial="initial" animate="animate" exit="exit"
            transition={{ duration: 0.3 }}
            onClick={onClose}
          />
          <motion.div
            className="toolkit-container"
            variants={containerVariants}
            initial="initial" animate="animate" exit="exit"
          >
            <ParallaxBackground />
            <div className="toolkit-header">
              <div className="header-left">
                <h1>Toolkit</h1>
              </div>
              <button className="toolkit-button" onClick={onClose} title="Close">
                <X size={16} />
              </button>
            </div>
            <div className="toolkit-layout">
              <nav className="toolkit-nav">
                {SECTIONS.map(s => (
                  <button
                    key={s.id}
                    className={`toolkit-nav-item ${active === s.id ? 'active' : ''}`}
                    onClick={() => setActive(s.id)}
                  >
                    {s.icon}
                    <span>{s.label}</span>
                  </button>
                ))}
              </nav>
              <div className="toolkit-content">
                {active === 'servers' && <ServersView />}
                {active === 'browser' && <ToolsBrowserView />}
                {active === 'meta' && <MetaView />}
                {active === 'access' && <AccessView />}
                {active === 'raw' && <RawJsonView />}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

/* ── Servers ──────────────────────────────────────────────── */

function ServersView() {
  const {
    servers, loading, refresh,
    connectServer, disconnectServer, connectAll,
    createServer, updateServer, deleteServer, validateServer,
  } = useMCPServers();
  const { profiles } = useAgentProfile();
  const [editing, setEditing] = useState<MCPServer | null | undefined>(undefined); // undefined = closed; null = creating
  const [busy, setBusy] = useState<string | null>(null);

  const handleSubmit = async (name: string, cfg: MCPServerConfigInput, rename?: string) => {
    if (editing && editing.name) {
      await updateServer(editing.name, cfg, rename);
    } else {
      await createServer(name, cfg);
    }
  };

  const onDelete = async (s: MCPServer) => {
    if (!confirm(`Delete server "${s.name}"? This rewrites mcp_servers.json.`)) return;
    setBusy(s.name);
    try { await deleteServer(s.name); } finally { setBusy(null); }
  };

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <h2>MCP Servers</h2>
          <p>{servers.length} configured · edits write to mcp_servers.json</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="toolkit-button" onClick={refresh} disabled={loading}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button className="toolkit-button" onClick={() => connectAll()} disabled={loading}>
            <Plug size={14} /> Connect all
          </button>
          <button className="toolkit-button primary" onClick={() => setEditing(null)}>
            <Plus size={14} /> Add server
          </button>
        </div>
      </div>

      {loading && servers.length === 0 ? (
        <div className="toolkit-empty"><Loader2 className="spin" /> Loading…</div>
      ) : servers.length === 0 ? (
        <div className="toolkit-empty">
          No servers configured yet. Click <strong>Add server</strong> to create one.
        </div>
      ) : (
        <div className="toolkit-card-grid">
          {servers.map(s => (
            <div key={s.name} className="toolkit-card">
              <div className="toolkit-card-header">
                <Server size={16} />
                <span className="name" title={s.name}>{s.name}</span>
                <span className={`status-dot ${s.status === 'connected' ? 'online' : 'offline'}`} />
              </div>
              <div className="meta">
                {s.transport} · {s.status === 'connected'
                  ? `${s.tools_count ?? 0} tools`
                  : s.transport === 'stdio'
                    ? (s.command || 'no command')
                    : (s.url || 'no url')}
              </div>
              {(s.tags?.length || s.groups?.length) ? (
                <div className="toolkit-chips">
                  {s.tags?.map(t => <span key={`t-${t}`} className="toolkit-chip">#{t}</span>)}
                  {s.groups?.map(g => <span key={`g-${g}`} className="toolkit-chip solid">{g}</span>)}
                </div>
              ) : null}
              <div className="meta">
                {s.allowed_agent_ids == null
                  ? 'All agents allowed'
                  : s.allowed_agent_ids.length === 0
                    ? 'No agents allowed'
                    : `${s.allowed_agent_ids.length} agent${s.allowed_agent_ids.length === 1 ? '' : 's'} whitelisted`}
              </div>
              <div className="toolkit-card-actions">
                {s.status === 'connected' ? (
                  <button className="toolkit-button" onClick={() => disconnectServer(s.name)} disabled={busy !== null}>
                    <Unplug size={14} /> Disconnect
                  </button>
                ) : (
                  <button className="toolkit-button" onClick={() => connectServer(s.name)} disabled={busy !== null}>
                    <Plug size={14} /> Connect
                  </button>
                )}
                <button className="toolkit-button" onClick={() => setEditing(s)} disabled={busy !== null}>
                  <Pencil size={14} /> Edit
                </button>
                <button className="toolkit-button danger" onClick={() => onDelete(s)} disabled={busy === s.name}>
                  {busy === s.name ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {editing !== undefined && createPortal(
        <div className="toolkit-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setEditing(undefined); }}>
          <div className="toolkit-modal-card">
            <h3>{editing ? `Edit ${editing.name}` : 'Add MCP server'}</h3>
            <ServerForm
              initial={editing ?? undefined}
              agentProfiles={profiles}
              onCancel={() => setEditing(undefined)}
              onSubmit={handleSubmit}
              onValidate={validateServer}
            />
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

/* ── Tools Browser ─────────────────────────────────────────── */

function ToolsBrowserView() {
  const { tools, loading } = useMCPTools();
  const [q, setQ] = useState('');
  const filtered = tools.filter(t =>
    !q || t.name.toLowerCase().includes(q.toLowerCase()) || (t.description || '').toLowerCase().includes(q.toLowerCase())
  );

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <h2>Tools</h2>
          <p>{tools.length} discovered across connected servers</p>
        </div>
      </div>
      <div className="toolkit-search">
        <Search size={16} />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search tools…" />
      </div>
      {loading ? (
        <div className="toolkit-empty"><Loader2 className="spin" /> Loading…</div>
      ) : filtered.length === 0 ? (
        <div className="toolkit-empty">No tools match.</div>
      ) : (
        <div className="toolkit-card-grid">
          {filtered.map(t => (
            <div key={`${t.server}-${t.name}`} className="toolkit-card">
              <div className="toolkit-card-header">
                <Wrench size={16} />
                <span className="name" title={t.name}>{t.name}</span>
              </div>
              <div className="meta">{t.server}</div>
              <div className="meta" style={{ color: 'var(--text-secondary)' }}>
                {t.description || 'No description'}
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

/* ── Groups & Tags ─────────────────────────────────────────── */

function MetaView() {
  const { servers, updateServer } = useMCPServers();
  const allTags = useMemo(() => uniqueAcross(servers.map(s => s.tags || [])), [servers]);
  const allGroups = useMemo(() => uniqueAcross(servers.map(s => s.groups || [])), [servers]);

  const toggle = async (s: MCPServer, field: 'tags' | 'groups', value: string) => {
    const current = (s[field] as string[] | undefined) ?? [];
    const next = current.includes(value)
      ? current.filter(v => v !== value)
      : [...current, value];
    await updateServer(s.name, toConfigInput({ ...s, [field]: next }));
  };

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <h2>Groups & Tags</h2>
          <p>Use the server form to define new tags/groups; toggle membership here.</p>
        </div>
      </div>
      {servers.length === 0 ? (
        <div className="toolkit-empty">No servers yet.</div>
      ) : (
        <div className="toolkit-card-grid">
          {servers.map(s => (
            <div key={s.name} className="toolkit-card">
              <div className="toolkit-card-header">
                <Server size={16} />
                <span className="name">{s.name}</span>
              </div>
              <div className="meta">Tags</div>
              <div className="toolkit-chips">
                {allTags.length === 0 && <span className="meta">No tags defined</span>}
                {allTags.map(t => (
                  <button
                    key={t}
                    className={`toolkit-chip ${(s.tags || []).includes(t) ? 'solid' : ''}`}
                    onClick={() => toggle(s, 'tags', t)}
                  >#{t}</button>
                ))}
              </div>
              <div className="meta">Groups</div>
              <div className="toolkit-chips">
                {allGroups.length === 0 && <span className="meta">No groups defined</span>}
                {allGroups.map(g => (
                  <button
                    key={g}
                    className={`toolkit-chip ${(s.groups || []).includes(g) ? 'solid' : ''}`}
                    onClick={() => toggle(s, 'groups', g)}
                  >{g}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

/* ── Access ────────────────────────────────────────────────── */

function AccessView() {
  const { servers, updateServer } = useMCPServers();
  const { profiles } = useAgentProfile();

  const toggleAgent = async (s: MCPServer, agentId: string) => {
    const current = s.allowed_agent_ids;
    let next: string[] | null;
    if (current == null) next = profiles.map(p => p.agentId).filter(a => a !== agentId);
    else next = current.includes(agentId) ? current.filter(a => a !== agentId) : [...current, agentId];
    await updateServer(s.name, toConfigInput({ ...s, allowed_agent_ids: next }));
  };
  const setAllowAll = async (s: MCPServer) => {
    await updateServer(s.name, toConfigInput({ ...s, allowed_agent_ids: null }));
  };

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <h2>Access</h2>
          <p>Whitelist which agents can use each server. "All" means no restriction.</p>
        </div>
      </div>
      {servers.length === 0 ? (
        <div className="toolkit-empty">No servers yet.</div>
      ) : (
        <div className="toolkit-card-grid">
          {servers.map(s => {
            const allowAll = s.allowed_agent_ids == null;
            return (
              <div key={s.name} className="toolkit-card">
                <div className="toolkit-card-header">
                  <Server size={16} />
                  <span className="name">{s.name}</span>
                </div>
                <div className="toolkit-chips toolkit-access-chips">
                  <button
                    className={`toolkit-chip ${allowAll ? 'solid' : ''}`}
                    onClick={() => setAllowAll(s)}
                  >All agents</button>
                  {profiles.map(p => {
                    const on = !allowAll && (s.allowed_agent_ids || []).includes(p.agentId);
                    return (
                      <button
                        key={p.agentId}
                        className={`toolkit-chip ${on ? 'solid' : ''}`}
                        onClick={() => toggleAgent(s, p.agentId)}
                      >{p.name}</button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

/* ── Raw JSON ──────────────────────────────────────────────── */

function RawJsonView() {
  const { servers, refresh } = useMCPServers();
  const initialJson = useMemo(() => buildRawJson(servers), [servers]);
  const [text, setText] = useState(initialJson);
  const [parseError, setParseError] = useState<string | null>(null);
  const [validateErrors, setValidateErrors] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => { setText(initialJson); }, [initialJson]);

  // Parse + validate every server in the JSON
  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(async () => {
      let parsed: unknown;
      try { parsed = JSON.parse(text); }
      catch (e) { setParseError(e instanceof Error ? e.message : String(e)); setValidateErrors([]); return; }
      setParseError(null);
      const root = parsed as { servers?: Record<string, MCPServerConfigInput> };
      if (!root || typeof root !== 'object' || !root.servers || typeof root.servers !== 'object') {
        setValidateErrors(['Top-level shape must be {"servers": {<name>: <config>}}']);
        return;
      }
      const errs: string[] = [];
      for (const [name, cfg] of Object.entries(root.servers)) {
        try {
          const r = await api.validateMCPServer(name, cfg);
          if (!r.valid) errs.push(`${name}: ${r.errors.join('; ')}`);
        } catch (e) {
          errs.push(`${name}: ${e instanceof Error ? e.message : String(e)}`);
        }
        if (cancelled) return;
      }
      if (!cancelled) setValidateErrors(errs);
    }, 400);
    return () => { cancelled = true; window.clearTimeout(t); };
  }, [text]);

  const valid = !parseError && validateErrors.length === 0;

  const save = async () => {
    if (!valid) return;
    setSaving(true);
    setSavedMsg(null);
    try {
      const root = JSON.parse(text) as { servers: Record<string, MCPServerConfigInput> };
      const nextNames = new Set(Object.keys(root.servers));
      const currentNames = new Set(servers.map(s => s.name));
      // Delete removed
      for (const name of currentNames) {
        if (!nextNames.has(name)) await api.deleteMCPServer(name);
      }
      // Upsert each
      for (const [name, cfg] of Object.entries(root.servers)) {
        if (currentNames.has(name)) await api.updateMCPServer(name, cfg);
        else await api.createMCPServer(name, cfg);
      }
      await refresh();
      setSavedMsg('Saved.');
    } catch (e) {
      setValidateErrors([e instanceof Error ? e.message : String(e)]);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <h2>Raw JSON</h2>
          <p>Edits replace the entire mcp_servers.json. Use with care.</p>
        </div>
      </div>
      <div className="toolkit-raw-editor">
        <div className="warning"><AlertTriangle size={14} /> Saving here disconnects affected servers and rewrites the config file.</div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        {parseError && <div className="toolkit-error-banner"><span>JSON parse: {parseError}</span></div>}
        {validateErrors.length > 0 && (
          <div className="toolkit-error-banner">
            {validateErrors.map((e, i) => <span key={i}>{e}</span>)}
          </div>
        )}
        {savedMsg && <div className="meta" style={{ color: 'rgb(34, 197, 94)' }}>{savedMsg}</div>}
        <div className="actions">
          <button className="toolkit-button" onClick={() => setText(initialJson)} disabled={saving}>
            Reset to current
          </button>
          <button className="toolkit-button primary" onClick={save} disabled={!valid || saving}>
            {saving ? <Loader2 size={14} className="spin" /> : <Save size={14} />} Save
          </button>
        </div>
      </div>
    </>
  );
}

/* ── helpers ──────────────────────────────────────────────── */

function uniqueAcross(lists: string[][]): string[] {
  const set = new Set<string>();
  for (const l of lists) for (const v of l) set.add(v);
  return Array.from(set).sort();
}

function toConfigInput(s: MCPServer): MCPServerConfigInput {
  return {
    transport: s.transport ?? 'stdio',
    command: s.command ?? null,
    args: s.args ?? [],
    env: s.env ?? {},
    url: s.url ?? null,
    headers: s.headers ?? {},
    timeout: s.timeout ?? 30,
    auto_reconnect: s.auto_reconnect ?? true,
    tags: s.tags ?? [],
    groups: s.groups ?? [],
    allowed_agent_ids: s.allowed_agent_ids ?? null,
  };
}

function buildRawJson(servers: MCPServer[]): string {
  const out: { servers: Record<string, Record<string, unknown>> } = { servers: {} };
  for (const s of servers) {
    const cfg: Record<string, unknown> = { transport: s.transport ?? 'stdio' };
    if (s.command) cfg.command = s.command;
    if (s.args && s.args.length) cfg.args = s.args;
    if (s.env && Object.keys(s.env).length) cfg.env = s.env;
    if (s.url) cfg.url = s.url;
    if (s.headers && Object.keys(s.headers).length) cfg.headers = s.headers;
    if (s.timeout != null && s.timeout !== 30) cfg.timeout = s.timeout;
    if (s.auto_reconnect === false) cfg.auto_reconnect = false;
    if (s.tags && s.tags.length) cfg.tags = s.tags;
    if (s.groups && s.groups.length) cfg.groups = s.groups;
    if (s.allowed_agent_ids != null) cfg.allowed_agent_ids = s.allowed_agent_ids;
    out.servers[s.name] = cfg;
  }
  return JSON.stringify(out, null, 2);
}

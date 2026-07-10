/**
 * ToolkitPage — Phase 18.2 immersive Tools menu replacement.
 *
 * Sub-views: Servers, Tools Browser, Groups & Tags, Access, Raw JSON.
 * Modeled on UnifiedSettings (animations + glass shell).
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import * as Accordion from '@radix-ui/react-accordion';
import {
  X, Server, Wrench, Tag, Shield, Code, Plus, Pencil, Trash2,
  Plug, Unplug, RefreshCw, Search, Loader2, AlertTriangle, Save, ChevronDown, KeyRound,
  ExternalLink, GraduationCap,
} from 'lucide-react';
import { useMCPServers, useMCPTools, useSkills } from '../../lib/hooks';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { needsAuth, sessionExpired } from '../../lib/connectors';
import { ParallaxBackground } from '../unified-settings/animations/ParallaxBackground';
import { backdropVariants, containerVariants } from '../unified-settings/animations/transitions';
import type { MCPServer, MCPServerConfigInput } from '../../lib/api';
import { api } from '../../lib/api';
import { useConfirm } from '../ui/ConfirmDialog';
import {
  Button, IconButton, Input, Textarea, StatusDot, CopyChip, Tooltip,
  Dialog, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription,
} from '../ui';
import { openExternal } from '../../lib/openExternal';
import { ServerForm } from './ServerForm';
import { ConnectorCatalogView } from './ConnectorCatalog';
import { SkillsSection } from './SkillsSection';
import './ToolkitPage.css';

interface ToolkitPageProps {
  isOpen: boolean;
  onClose: () => void;
}

/** A collapsible Toolkit section (Radix Accordion item), styled like the glass cards. */
function ToolkitSection({
  value, icon, title, subtitle, children,
}: {
  value: string;
  icon: ReactNode;
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <Accordion.Item value={value} className="toolkit-section-card">
      <Accordion.Header>
        <Accordion.Trigger className="toolkit-accordion-trigger">
          <span className="toolkit-accordion-titlewrap">
            <span className="toolkit-accordion-title">{icon}<span>{title}</span></span>
            {subtitle && <span className="toolkit-accordion-subtitle">{subtitle}</span>}
          </span>
          <ChevronDown size={16} className="toolkit-accordion-chevron" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="toolkit-accordion-content">
        <div className="toolkit-accordion-content-inner">{children}</div>
      </Accordion.Content>
    </Accordion.Item>
  );
}

export function ToolkitPage({ isOpen, onClose }: ToolkitPageProps) {
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
                <h1>Connectors &amp; Tools</h1>
                <p className="toolkit-tagline">Your agents' interface to the real world</p>
              </div>
              <IconButton aria-label="Close Connectors & Tools" onClick={onClose}>
                <X size={16} />
              </IconButton>
            </div>
            <div className="toolkit-content toolkit-single-view">
              <StatsStrip />

              {/* Primary section — always visible. */}
              <ServersView />

              {/* Everything else folds into collapsible cards in the same scroll. */}
              <Accordion.Root
                type="multiple"
                defaultValue={['connectors', 'skills', 'meta', 'access', 'catalog']}
                className="toolkit-accordion-root"
              >
                <ToolkitSection
                  value="connectors"
                  icon={<Plug size={15} />}
                  title="Connector Catalog"
                  subtitle="Add known-good connectors, or search the MCP registry"
                >
                  <ConnectorCatalogView />
                </ToolkitSection>
                <ToolkitSection
                  value="skills"
                  icon={<GraduationCap size={15} />}
                  title="Skills"
                  subtitle="Instruction packs agents load on demand with use_skill"
                >
                  <SkillsSection />
                </ToolkitSection>
                <ToolkitSection
                  value="meta"
                  icon={<Tag size={15} />}
                  title="Groups & Tags"
                  subtitle="Toggle each server's tag / group membership"
                >
                  <MetaView />
                </ToolkitSection>
                <ToolkitSection
                  value="access"
                  icon={<Shield size={15} />}
                  title="Agent Access"
                  subtitle="Whitelist which agents may use each server"
                >
                  <AccessView />
                </ToolkitSection>
                <ToolkitSection
                  value="catalog"
                  icon={<Wrench size={15} />}
                  title="Tool Catalog"
                  subtitle="Discovered tools across connected servers"
                >
                  <ToolsBrowserView />
                </ToolkitSection>
                <ToolkitSection
                  value="raw"
                  icon={<Code size={15} />}
                  title="Raw JSON"
                  subtitle="Advanced — edit mcp_servers.json directly"
                >
                  <RawJsonView />
                </ToolkitSection>
              </Accordion.Root>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

/* ── Stats strip ──────────────────────────────────────────── */

/** Control-center vitals: servers/tools/skills counts + a sign-in warning. */
function StatsStrip() {
  const { servers } = useMCPServers();
  const { tools } = useMCPTools();
  const { skills } = useSkills();
  // Whole-fleet view (agent-agnostic): every connector needing any sign-in.
  const needAuth = servers.filter(needsAuth).length;
  const connected = servers.filter(s => s.status === 'connected').length;
  const enabledSkills = skills.filter(s => s.enabled).length;

  return (
    <div className="toolkit-stats">
      <span className="toolkit-stat"><Server size={13} /> {servers.length} server{servers.length === 1 ? '' : 's'} · {connected} connected</span>
      <span className="toolkit-stat"><Wrench size={13} /> {tools.length} tool{tools.length === 1 ? '' : 's'}</span>
      <span className="toolkit-stat"><GraduationCap size={13} /> {enabledSkills} skill{enabledSkills === 1 ? '' : 's'}</span>
      {needAuth > 0 && (
        <span className="toolkit-stat warn"><KeyRound size={13} /> {needAuth} need{needAuth === 1 ? 's' : ''} sign-in</span>
      )}
    </div>
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
  const confirm = useConfirm();
  const [editing, setEditing] = useState<MCPServer | null | undefined>(undefined); // undefined = closed; null = creating
  const [busy, setBusy] = useState<string | null>(null);
  // Server currently waiting on a browser OAuth consent (connect continues
  // server-side; we poll until it flips to connected or errors). `authUrl` is
  // the authorization URL for the manual-open fallback in the sign-in dialog.
  const [authWait, setAuthWait] = useState<string | null>(null);
  const [authUrl, setAuthUrl] = useState<string | null>(null);

  const clearAuthWait = () => { setAuthWait(null); setAuthUrl(null); };

  const handleSubmit = async (name: string, cfg: MCPServerConfigInput, rename?: string) => {
    if (editing && editing.name) {
      await updateServer(editing.name, cfg, rename);
    } else {
      await createServer(name, cfg);
    }
  };

  const handleConnect = async (s: MCPServer) => {
    const result = await connectServer(s.name);
    if (result?.status === 'auth_required' && result.authorization_url) {
      setAuthWait(s.name);
      setAuthUrl(result.authorization_url);
      // Fire the browser open up front (Tauri-aware); the dialog stays up as a
      // manual fallback if it didn't open.
      void openExternal(result.authorization_url);
    }
  };

  // Abort an in-flight sign-in server-side so a late completion can't flip the
  // server to "signed in" after the user backed out.
  const cancelAuth = async (name: string) => {
    try { await api.cancelMCPServerAuth(name); } catch { /* best-effort */ }
    clearAuthWait();
    void refresh();
  };

  // Poll while the sign-in dialog is open; stop on connected / auth error / timeout.
  useEffect(() => {
    if (!authWait) return;
    const started = Date.now();
    const t = window.setInterval(() => {
      void refresh();
      if (Date.now() - started > 5 * 60_000) clearAuthWait();
    }, 2500);
    return () => window.clearInterval(t);
  }, [authWait, refresh]);
  useEffect(() => {
    if (!authWait) return;
    const s = servers.find(x => x.name === authWait);
    if (!s || s.status === 'connected' || s.auth_state?.error) clearAuthWait();
  }, [servers, authWait]);

  const resetAuth = async (s: MCPServer) => {
    const ok = await confirm({
      title: `Reset sign-in for "${s.name}"?`,
      body: 'Forgets the stored tokens and registration — the next connect asks you to sign in again.',
      confirmLabel: 'Reset',
      danger: true,
    });
    if (!ok) return;
    setBusy(s.name);
    try { await api.resetMCPServerAuth(s.name); await refresh(); } finally { setBusy(null); }
  };

  const onDelete = async (s: MCPServer) => {
    const ok = await confirm({
      title: `Delete server "${s.name}"?`,
      body: 'This rewrites mcp_servers.json.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
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
          <Button variant="secondary" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw size={14} /> Refresh
          </Button>
          <Button variant="secondary" size="sm" onClick={() => connectAll()} disabled={loading}>
            <Plug size={14} /> Connect all
          </Button>
          <Button variant="primary" size="sm" onClick={() => setEditing(null)}>
            <Plus size={14} /> Add server
          </Button>
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
                <StatusDot
                  tone={
                    s.status === 'connected' ? 'online'
                      : s.auth_state?.error ? 'error'
                        : (authWait === s.name || s.auth_state?.pending || sessionExpired(s)) ? 'warning'
                          : 'inactive'
                  }
                  pulse={authWait === s.name || !!s.auth_state?.pending}
                  title={
                    s.status === 'connected' ? 'Connected'
                      : sessionExpired(s) ? 'Session expired'
                        : 'Disconnected'
                  }
                />
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
              {s.auth?.type === 'oauth' && (
                <div className="meta">
                  {authWait === s.name || s.auth_state?.pending
                    ? 'OAuth · waiting for authorization…'
                    : s.auth_state?.error
                      ? `OAuth · sign-in failed: ${s.auth_state.error}`
                      : sessionExpired(s)
                        ? 'OAuth · session expired — sign in again on connect'
                        : s.auth_state?.authorized
                          ? s.auth_state.expired
                            ? 'OAuth · signed in (refreshes on connect)'
                            : 'OAuth · signed in'
                          : 'OAuth · sign-in required on connect'}
                </div>
              )}
              <div className="toolkit-card-actions">
                {authWait === s.name ? (
                  <Button variant="secondary" size="sm" disabled>
                    <Loader2 size={14} className="spin" /> Waiting…
                  </Button>
                ) : s.status === 'connected' ? (
                  <Button variant="secondary" size="sm" onClick={() => disconnectServer(s.name)} disabled={busy !== null}>
                    <Unplug size={14} /> Disconnect
                  </Button>
                ) : (
                  <Button variant="primary" size="sm" onClick={() => void handleConnect(s)} disabled={busy !== null}>
                    <Plug size={14} /> Connect
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={() => setEditing(s)} disabled={busy !== null}>
                  <Pencil size={14} /> Edit
                </Button>
                {s.auth?.type === 'oauth' && s.auth_state?.authorized && (
                  <Tooltip content="Forget stored tokens (sign in again on next connect)">
                    <Button variant="ghost" size="sm" onClick={() => void resetAuth(s)} disabled={busy !== null}>
                      <KeyRound size={14} /> Reset auth
                    </Button>
                  </Tooltip>
                )}
                <Tooltip content="Delete server">
                  <IconButton
                    aria-label={`Delete ${s.name}`}
                    size="sm"
                    tone="danger"
                    onClick={() => onDelete(s)}
                    disabled={busy === s.name}
                  >
                    {busy === s.name ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />}
                  </IconButton>
                </Tooltip>
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

      {/* OAuth sign-in — auto-opens the browser and stays up as the manual
          fallback. `open` is derived from authWait, so the poll effect that
          clears authWait on connect auto-closes it. Dismissing (Esc/backdrop)
          cancels the server-side flow. */}
      <Dialog
        open={authWait != null}
        onOpenChange={(o) => { if (!o && authWait) void cancelAuth(authWait); }}
      >
        <DialogContent showClose={false} className="max-w-md">
          <DialogHeader>
            <DialogTitle>Sign in to “{authWait}”</DialogTitle>
            <DialogDescription>
              We opened your browser to finish signing in. Didn’t open? Use the link below.
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 py-3">
            {authUrl && <CopyChip value={authUrl} label="Copy sign-in link" />}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => authWait && void cancelAuth(authWait)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => authUrl && void openExternal(authUrl)}>
              <ExternalLink size={14} /> Open sign-in page
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
      <p className="toolkit-section-note">
        Reference catalog of discovered tools. Per-agent tool access is set per profile —
        Agent Profiles → Tools.
      </p>
      <Input
        icon={<Search size={16} />}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search tools…"
      />
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
      <p className="toolkit-section-note">Define new tags/groups in a server's form; toggle membership here.</p>
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
      <p className="toolkit-section-note">"All agents" means no restriction on this server.</p>
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
      <div className="toolkit-raw-editor">
        <div className="warning"><AlertTriangle size={14} /> Saving here disconnects affected servers and rewrites the config file.</div>
        <Textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        {parseError && <div className="toolkit-error-banner"><span>JSON parse: {parseError}</span></div>}
        {validateErrors.length > 0 && (
          <div className="toolkit-error-banner">
            {validateErrors.map((e, i) => <span key={i}>{e}</span>)}
          </div>
        )}
        {savedMsg && <div className="meta" style={{ color: 'var(--feedback-success)' }}>{savedMsg}</div>}
        <div className="actions">
          <Button variant="secondary" size="sm" onClick={() => setText(initialJson)} disabled={saving}>
            Reset to current
          </Button>
          <Button variant="primary" size="sm" onClick={save} disabled={!valid || saving}>
            {saving ? <Loader2 size={14} className="spin" /> : <Save size={14} />} Save
          </Button>
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
    auto_connect: s.auto_connect ?? false,
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
    if (s.auto_connect) cfg.auto_connect = true;
    if (s.tags && s.tags.length) cfg.tags = s.tags;
    if (s.groups && s.groups.length) cfg.groups = s.groups;
    if (s.allowed_agent_ids != null) cfg.allowed_agent_ids = s.allowed_agent_ids;
    out.servers[s.name] = cfg;
  }
  return JSON.stringify(out, null, 2);
}

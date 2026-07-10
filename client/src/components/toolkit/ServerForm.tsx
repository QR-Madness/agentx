import { useEffect, useMemo, useState } from 'react';
import { Plus, Trash2, HelpCircle, ChevronDown, Globe, TerminalSquare, ExternalLink, Lightbulb } from 'lucide-react';
import type { MCPServer, MCPServerConfigInput, AgentProfile } from '../../lib/api';
import type { ServerDraft } from '../../lib/connectorCatalog';
import { openExternal } from '../../lib/openExternal';
import {
  Input, Textarea, Button, IconButton, Switch, Tooltip, SegmentedControl,
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from '../ui';

interface ServerFormProps {
  initial?: MCPServer | null;
  /** Prefill for a NEW server (catalog entry / registry result) — ignored when
   *  editing. Its guidance renders as a setup panel above the fields. */
  initialDraft?: ServerDraft | null;
  agentProfiles: AgentProfile[];
  onCancel: () => void;
  onSubmit: (name: string, config: MCPServerConfigInput, rename?: string) => Promise<void>;
  onValidate: (name: string, config: MCPServerConfigInput) => Promise<{ valid: boolean; errors: string[] }>;
}

type KV = { key: string; value: string };
type Mode = 'remote' | 'local';
/** Remote sub-transports — the specific wire protocol behind an MCP URL. */
type RemoteTransport = 'streamable_http' | 'sse' | 'websocket';

function dictToList(d?: Record<string, string>): KV[] {
  return d ? Object.entries(d).map(([key, value]) => ({ key, value })) : [];
}
function listToDict(rows: KV[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const r of rows) if (r.key.trim()) out[r.key.trim()] = r.value;
  return out;
}

/** Small ⓘ affordance that reveals guidance on hover/focus — keeps labels terse. */
function Hint({ text }: { text: string }) {
  return (
    <Tooltip content={text}>
      <span className="toolkit-hint" tabIndex={0} role="note" aria-label={text}>
        <HelpCircle size={13} />
      </span>
    </Tooltip>
  );
}

export function ServerForm({ initial, initialDraft, agentProfiles, onCancel, onSubmit, onValidate }: ServerFormProps) {
  const isEdit = !!initial;
  // Prefill seed for NEW servers only — editing always reflects the saved server.
  const draft = isEdit ? undefined : initialDraft?.config;
  const [name, setName] = useState(initial?.name ?? (isEdit ? '' : initialDraft?.name ?? ''));
  // `stdio` = local command; anything else = a remote MCP URL. Default new
  // servers to remote streamable_http (the overwhelming common case).
  const [transport, setTransport] = useState(initial?.transport ?? draft?.transport ?? 'streamable_http');
  const [command, setCommand] = useState(initial?.command ?? draft?.command ?? '');
  const [argsText, setArgsText] = useState((initial?.args ?? draft?.args ?? []).join('\n'));
  const [url, setUrl] = useState(initial?.url ?? draft?.url ?? '');
  const [env, setEnv] = useState<KV[]>(dictToList(initial?.env ?? draft?.env));
  const [headers, setHeaders] = useState<KV[]>(dictToList(initial?.headers ?? draft?.headers));
  const seedAuth = initial?.auth ?? draft?.auth;
  const [authMode, setAuthMode] = useState<'none' | 'oauth'>(seedAuth?.type === 'oauth' ? 'oauth' : 'none');
  const [authScope, setAuthScope] = useState(seedAuth?.scope ?? '');
  const [authClientId, setAuthClientId] = useState(seedAuth?.client_id ?? '');
  const [authClientSecret, setAuthClientSecret] = useState(seedAuth?.client_secret ?? '');
  const [timeout, setTimeoutVal] = useState<number>(initial?.timeout ?? draft?.timeout ?? 30);
  const [autoReconnect, setAutoReconnect] = useState<boolean>(initial?.auto_reconnect ?? draft?.auto_reconnect ?? true);
  const [tags, setTags] = useState<string>((initial?.tags ?? []).join(', '));
  const [groups, setGroups] = useState<string>((initial?.groups ?? []).join(', '));
  const [whitelistAll, setWhitelistAll] = useState<boolean>(initial?.allowed_agent_ids == null);
  const [allowedAgents, setAllowedAgents] = useState<string[]>(initial?.allowed_agent_ids ?? []);
  // Existing servers open Advanced so their configured settings are visible;
  // a new server seeded with a pre-registered client id opens it too (the
  // client id/secret live in Advanced now). New blank servers start collapsed.
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(isEdit || !!seedAuth?.client_id);

  const [errors, setErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const mode: Mode = transport === 'stdio' ? 'local' : 'remote';
  const setMode = (m: Mode) => {
    if (m === 'local') setTransport('stdio');
    else if (transport === 'stdio') setTransport('streamable_http');
  };

  const buildConfig = (): MCPServerConfigInput => ({
    transport,
    command: transport === 'stdio' ? (command || null) : null,
    args: transport === 'stdio'
      ? argsText.split('\n').map(s => s.trim()).filter(Boolean)
      : [],
    env: transport === 'stdio' ? listToDict(env) : {},
    url: transport !== 'stdio' ? (url || null) : null,
    headers: transport !== 'stdio' ? listToDict(headers) : {},
    auth: transport !== 'stdio' && authMode === 'oauth'
      ? {
          type: 'oauth' as const,
          ...(authScope.trim() ? { scope: authScope.trim() } : {}),
          ...(authClientId.trim() ? { client_id: authClientId.trim() } : {}),
          ...(authClientSecret.trim() ? { client_secret: authClientSecret.trim() } : {}),
        }
      : null,
    timeout: Number.isFinite(timeout) ? timeout : 30,
    auto_reconnect: autoReconnect,
    // Preserve the auto-managed connected-state flag across edits (not a form
    // field — it's set/cleared by connect/disconnect).
    auto_connect: initial?.auto_connect ?? false,
    tags: tags.split(',').map(s => s.trim()).filter(Boolean),
    groups: groups.split(',').map(s => s.trim()).filter(Boolean),
    allowed_agent_ids: whitelistAll ? null : allowedAgents,
  });

  // Live validation (debounced)
  const candidateName = name.trim() || '__candidate__';
  const config = useMemo(buildConfig, [
    transport, command, argsText, url, env, headers, timeout, autoReconnect,
    tags, groups, whitelistAll, allowedAgents,
    authMode, authScope, authClientId, authClientSecret,
  ]);
  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const r = await onValidate(candidateName, config);
        if (!cancelled) setErrors(r.valid ? [] : r.errors);
      } catch { /* ignore validate transport errors */ }
    }, 300);
    return () => { cancelled = true; window.clearTimeout(t); };
  }, [candidateName, config, onValidate]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) { setErrors(['Name is required']); return; }
    setSubmitting(true);
    try {
      const rename = isEdit && initial && initial.name !== name.trim() ? name.trim() : undefined;
      await onSubmit(isEdit ? (initial!.name) : name.trim(), buildConfig(), rename);
      onCancel(); // close on success
    } catch (err) {
      setErrors([err instanceof Error ? err.message : String(err)]);
      setSubmitting(false);
    }
  };

  const updateKV = (rows: KV[], setRows: (r: KV[]) => void, idx: number, patch: Partial<KV>) => {
    setRows(rows.map((r, i) => i === idx ? { ...r, ...patch } : r));
  };

  const renderKV = (rows: KV[], setRows: (r: KV[]) => void, label: string, hint?: string) => (
    <label>
      <span>{label}{hint && <Hint text={hint} />}</span>
      {rows.map((r, i) => (
        <div key={i} className="toolkit-kv-row">
          <Input value={r.key} placeholder="KEY" onChange={(e) => updateKV(rows, setRows, i, { key: e.target.value })} />
          <Input value={r.value} placeholder="value (use ${VAR})" onChange={(e) => updateKV(rows, setRows, i, { value: e.target.value })} />
          <IconButton aria-label="Remove row" size="sm" tone="danger" onClick={() => setRows(rows.filter((_, j) => j !== i))}>
            <Trash2 size={14} />
          </IconButton>
        </div>
      ))}
      <Button type="button" variant="ghost" size="sm" onClick={() => setRows([...rows, { key: '', value: '' }])}>
        <Plus size={14} /> Add row
      </Button>
    </label>
  );

  const guidance = !isEdit ? initialDraft?.guidance : undefined;

  return (
    <form className="toolkit-form" onSubmit={submit}>
      {guidance && (guidance.note || guidance.docsUrl) && (
        <div className="toolkit-guidance">
          <div className="toolkit-guidance-title">
            <Lightbulb size={14} />
            <span>{guidance.title ?? 'Setup'}</span>
            {guidance.docsUrl && (
              <button
                type="button"
                className="toolkit-guidance-link"
                onClick={() => void openExternal(guidance.docsUrl!)}
              >
                Docs <ExternalLink size={12} />
              </button>
            )}
          </div>
          {guidance.note && <div className="toolkit-guidance-note">{guidance.note}</div>}
        </div>
      )}
      <label>
        <span>Server name</span>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="filesystem" required disabled={submitting} />
      </label>

      <label>
        <span>Connection</span>
        <SegmentedControl<Mode>
          value={mode}
          onChange={setMode}
          ariaLabel="Connection type"
          options={[
            { value: 'remote', label: 'Remote (URL)', icon: <Globe size={14} /> },
            { value: 'local', label: 'Local (command)', icon: <TerminalSquare size={14} /> },
          ]}
        />
      </label>

      {mode === 'remote' ? (
        <>
          <label>
            <span>MCP server URL</span>
            <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://mcp.example.com/sse" disabled={submitting} />
          </label>
          <label>
            <span>Authorization</span>
            <Select value={authMode} onValueChange={(v) => setAuthMode(v as 'none' | 'oauth')} disabled={submitting}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None / static headers</SelectItem>
                <SelectItem value="oauth">OAuth 2.1 (sign in via browser)</SelectItem>
              </SelectContent>
            </Select>
          </label>
          {authMode === 'oauth' && (
            <>
              <label>
                <span>Scope<Hint text="Space-separated OAuth scopes to request. Leave blank to use the provider's defaults." /></span>
                <Input value={authScope} onChange={(e) => setAuthScope(e.target.value)} placeholder="mcp:tools offline_access" disabled={submitting} />
              </label>
              <p className="toolkit-section-note" style={{ margin: 0 }}>
                Most servers register automatically — just connect to sign in via your browser.
                {' '}Providers that need a pre-registered app (client ID/secret) live under Advanced.
              </p>
            </>
          )}
        </>
      ) : (
        <>
          <label>
            <span>Command</span>
            <Input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx" disabled={submitting} />
          </label>
          <label>
            <span>Args<Hint text="One argument per line." /></span>
            <Textarea value={argsText} onChange={(e) => setArgsText(e.target.value)} rows={4} placeholder="-y&#10;@modelcontextprotocol/server-filesystem&#10;/tmp" />
          </label>
        </>
      )}

      {/* Advanced — every remaining setting, gated behind a disclosure so the
          common path stays a URL + auth. Nothing is dropped. */}
      <button
        type="button"
        className="toolkit-advanced-toggle"
        aria-expanded={advancedOpen}
        onClick={() => setAdvancedOpen(o => !o)}
      >
        <ChevronDown size={15} className={advancedOpen ? 'open' : ''} />
        Advanced settings
      </button>

      {advancedOpen && (
        <div className="toolkit-advanced-body">
          {mode === 'remote' ? (
            <>
              <label>
                <span>Transport<Hint text="The wire protocol behind the URL. streamable_http suits most modern MCP servers; pick sse or websocket only if the server requires it." /></span>
                <Select value={transport} onValueChange={(v) => setTransport(v as RemoteTransport)} disabled={submitting}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="streamable_http">streamable_http</SelectItem>
                    <SelectItem value="sse">sse</SelectItem>
                    <SelectItem value="websocket">websocket</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              {authMode === 'oauth' && (
                <>
                  <label>
                    <span>Pre-registered client ID<Hint text="Leave blank for automatic dynamic client registration (RFC 7591) — the common case. Set it only for providers that require a pre-registered OAuth app." /></span>
                    <Input value={authClientId} onChange={(e) => setAuthClientId(e.target.value)} placeholder="blank = automatic registration (use ${VAR})" disabled={submitting} />
                  </label>
                  {authClientId.trim() && (
                    <label>
                      <span>Client secret<Hint text="Only for confidential pre-registered clients. Prefer ${VAR} so the secret isn't stored in mcp_servers.json." /></span>
                      <Input type="password" value={authClientSecret} onChange={(e) => setAuthClientSecret(e.target.value)} placeholder="${MY_CLIENT_SECRET}" disabled={submitting} />
                    </label>
                  )}
                </>
              )}
              {renderKV(headers, setHeaders, 'Headers', 'Static request headers sent on every call (e.g. a static bearer token). Use ${VAR} for env expansion.')}
            </>
          ) : (
            renderKV(env, setEnv, 'Environment variables', 'Passed to the launched command. Use ${VAR} to expand from the API server\'s environment.')
          )}

          <div className="toolkit-form-row">
            <label>
              <span>Timeout (s)</span>
              <Input type="number" min={1} step={0.5} value={timeout} onChange={(e) => setTimeoutVal(parseFloat(e.target.value) || 30)} />
            </label>
          </div>

          <div className="toolkit-switch-row" onClick={() => setAutoReconnect(v => !v)} role="presentation">
            <span>Auto-reconnect<Hint text="Reconnect automatically if the session drops or on app start." /></span>
            <Switch checked={autoReconnect} onCheckedChange={setAutoReconnect} onClick={(e) => e.stopPropagation()} />
          </div>

          <label>
            <span>Tags (comma-separated)</span>
            <Input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="filesystem, local" />
          </label>
          <label>
            <span>Groups (comma-separated)</span>
            <Input value={groups} onChange={(e) => setGroups(e.target.value)} placeholder="research, dev" />
          </label>

          <div className="toolkit-switch-row" onClick={() => setWhitelistAll(v => !v)} role="presentation">
            <span>Allow all agents<Hint text="When on, every agent may use this server. Turn off to whitelist specific agents." /></span>
            <Switch checked={whitelistAll} onCheckedChange={setWhitelistAll} onClick={(e) => e.stopPropagation()} />
          </div>
          {!whitelistAll && (
            <label>
              <span>Allowed agents</span>
              <div className="toolkit-chips">
                {agentProfiles.map(p => {
                  const on = allowedAgents.includes(p.agentId);
                  return (
                    <button
                      type="button"
                      key={p.agentId}
                      className={`toolkit-chip ${on ? 'solid' : ''}`}
                      onClick={() => setAllowedAgents(on
                        ? allowedAgents.filter(a => a !== p.agentId)
                        : [...allowedAgents, p.agentId])}
                    >
                      {p.name} <span style={{ opacity: 0.6 }}>({p.agentId})</span>
                    </button>
                  );
                })}
                {agentProfiles.length === 0 && <span className="meta">No agent profiles defined</span>}
              </div>
            </label>
          )}
        </div>
      )}

      {errors.length > 0 && (
        <div className="toolkit-error-banner">
          {errors.map((e, i) => <span key={i}>{e}</span>)}
        </div>
      )}

      <div className="toolkit-modal-actions">
        <Button type="button" variant="ghost" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" disabled={submitting || errors.length > 0}>
          {isEdit ? 'Save changes' : 'Create server'}
        </Button>
      </div>
    </form>
  );
}

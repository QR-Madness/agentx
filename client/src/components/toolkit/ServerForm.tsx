import { useEffect, useMemo, useState } from 'react';
import { Plus, Trash2, X } from 'lucide-react';
import type { MCPServer, MCPServerConfigInput, AgentProfile } from '../../lib/api';

interface ServerFormProps {
  initial?: MCPServer | null;
  agentProfiles: AgentProfile[];
  onCancel: () => void;
  onSubmit: (name: string, config: MCPServerConfigInput, rename?: string) => Promise<void>;
  onValidate: (name: string, config: MCPServerConfigInput) => Promise<{ valid: boolean; errors: string[] }>;
}

type KV = { key: string; value: string };

function dictToList(d?: Record<string, string>): KV[] {
  return d ? Object.entries(d).map(([key, value]) => ({ key, value })) : [];
}
function listToDict(rows: KV[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const r of rows) if (r.key.trim()) out[r.key.trim()] = r.value;
  return out;
}

export function ServerForm({ initial, agentProfiles, onCancel, onSubmit, onValidate }: ServerFormProps) {
  const isEdit = !!initial;
  const [name, setName] = useState(initial?.name ?? '');
  const [transport, setTransport] = useState(initial?.transport ?? 'stdio');
  const [command, setCommand] = useState(initial?.command ?? '');
  const [argsText, setArgsText] = useState((initial?.args ?? []).join('\n'));
  const [url, setUrl] = useState(initial?.url ?? '');
  const [env, setEnv] = useState<KV[]>(dictToList(initial?.env));
  const [headers, setHeaders] = useState<KV[]>(dictToList(initial?.headers));
  const [timeout, setTimeoutVal] = useState<number>(initial?.timeout ?? 30);
  const [autoReconnect, setAutoReconnect] = useState<boolean>(initial?.auto_reconnect ?? true);
  const [tags, setTags] = useState<string>((initial?.tags ?? []).join(', '));
  const [groups, setGroups] = useState<string>((initial?.groups ?? []).join(', '));
  const [whitelistAll, setWhitelistAll] = useState<boolean>(initial?.allowed_agent_ids == null);
  const [allowedAgents, setAllowedAgents] = useState<string[]>(initial?.allowed_agent_ids ?? []);

  const [errors, setErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const buildConfig = (): MCPServerConfigInput => ({
    transport,
    command: transport === 'stdio' ? (command || null) : null,
    args: transport === 'stdio'
      ? argsText.split('\n').map(s => s.trim()).filter(Boolean)
      : [],
    env: transport === 'stdio' ? listToDict(env) : {},
    url: transport !== 'stdio' ? (url || null) : null,
    headers: transport !== 'stdio' ? listToDict(headers) : {},
    timeout: Number.isFinite(timeout) ? timeout : 30,
    auto_reconnect: autoReconnect,
    tags: tags.split(',').map(s => s.trim()).filter(Boolean),
    groups: groups.split(',').map(s => s.trim()).filter(Boolean),
    allowed_agent_ids: whitelistAll ? null : allowedAgents,
  });

  // Live validation (debounced)
  const candidateName = name.trim() || '__candidate__';
  const config = useMemo(buildConfig, [
    transport, command, argsText, url, env, headers, timeout, autoReconnect,
    tags, groups, whitelistAll, allowedAgents,
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
    const next = rows.map((r, i) => i === idx ? { ...r, ...patch } : r);
    setRows(next);
  };

  const renderKV = (rows: KV[], setRows: (r: KV[]) => void, label: string) => (
    <label>
      <span>{label}</span>
      {rows.map((r, i) => (
        <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
          <input value={r.key} placeholder="KEY" onChange={(e) => updateKV(rows, setRows, i, { key: e.target.value })} />
          <input value={r.value} placeholder="value (use ${VAR} for env)" onChange={(e) => updateKV(rows, setRows, i, { value: e.target.value })} />
          <button type="button" className="toolkit-button" onClick={() => setRows(rows.filter((_, j) => j !== i))} title="Remove">
            <Trash2 size={14} />
          </button>
        </div>
      ))}
      <button type="button" className="toolkit-button" onClick={() => setRows([...rows, { key: '', value: '' }])}>
        <Plus size={14} /> Add row
      </button>
    </label>
  );

  return (
    <form className="toolkit-form" onSubmit={submit}>
      <label>
        <span>Server name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="filesystem"
          required
          disabled={submitting}
        />
      </label>

      <label>
        <span>Transport</span>
        <select value={transport} onChange={(e) => setTransport(e.target.value)} disabled={submitting}>
          <option value="stdio">stdio</option>
          <option value="sse">sse</option>
          <option value="streamable_http">streamable_http</option>
          <option value="websocket">websocket</option>
        </select>
      </label>

      {transport === 'stdio' ? (
        <>
          <label>
            <span>Command</span>
            <input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx" disabled={submitting} />
          </label>
          <label>
            <span>Args (one per line)</span>
            <textarea value={argsText} onChange={(e) => setArgsText(e.target.value)} rows={4} placeholder="-y&#10;@modelcontextprotocol/server-filesystem&#10;/tmp" />
          </label>
          {renderKV(env, setEnv, 'Environment variables')}
        </>
      ) : (
        <>
          <label>
            <span>URL</span>
            <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://..." disabled={submitting} />
          </label>
          {renderKV(headers, setHeaders, 'Headers')}
        </>
      )}

      <label>
        <span>Timeout (s)</span>
        <input
          type="number"
          min={1}
          step={0.5}
          value={timeout}
          onChange={(e) => setTimeoutVal(parseFloat(e.target.value) || 30)}
        />
      </label>
      <label className="checkbox-row">
        <input type="checkbox" checked={autoReconnect} onChange={(e) => setAutoReconnect(e.target.checked)} />
        <span>Auto-reconnect</span>
      </label>

      <label>
        <span>Tags (comma-separated)</span>
        <input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="filesystem, local" />
      </label>
      <label>
        <span>Groups (comma-separated)</span>
        <input value={groups} onChange={(e) => setGroups(e.target.value)} placeholder="research, dev" />
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={whitelistAll}
          onChange={(e) => setWhitelistAll(e.target.checked)}
        />
        <span>Allow all agents</span>
      </label>
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

      {errors.length > 0 && (
        <div className="toolkit-error-banner">
          {errors.map((e, i) => <span key={i}>{e}</span>)}
        </div>
      )}

      <div className="toolkit-modal-actions">
        <button type="button" className="toolkit-button" onClick={onCancel} disabled={submitting}>
          <X size={14} /> Cancel
        </button>
        <button type="submit" className="toolkit-button primary" disabled={submitting || errors.length > 0}>
          {isEdit ? 'Save changes' : 'Create server'}
        </button>
      </div>
    </form>
  );
}

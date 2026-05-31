import { useMemo, useState } from 'react';
import { Lock, Search, RotateCcw } from 'lucide-react';
import { useMCPTools } from '../../lib/hooks';

interface ToolAccessSectionProps {
  /**
   * `null` ⇔ all tools on (minus blocked); a non-null array is an allow-list.
   * The UI never emits `[]` — clearing the last Allow collapses back to `null`.
   */
  allowedTools: string[] | null;
  setAllowedTools: (next: string[] | null) => void;
  /** Always an array (server defaults to []). Wins over allowedTools. */
  blockedTools: string[];
  setBlockedTools: (next: string[]) => void;
}

type ToolState = 'default' | 'allow' | 'block';

const SEGMENTS: { value: ToolState; label: string }[] = [
  { value: 'default', label: 'Default' },
  { value: 'allow', label: 'Allow' },
  { value: 'block', label: 'Block' },
];

/**
 * Phase 18.9.x — per-profile tool gating UI.
 *
 * One flat, server-grouped list; each tool carries a Default · Allow · Block
 * segmented control. The two backend levers (`allowed_tools` allow-list +
 * `blocked_tools` denylist, matched on fully-qualified `server.tool` keys in
 * `Agent._get_tools_for_provider`) are derived from the per-tool states:
 *   - any tool set to Allow ⇒ allow-list mode (others are off)
 *   - no Allow set          ⇒ all tools on, minus Blocked
 * A per-row effective-state dim makes the "Default means off now" flip visible
 * once an allow-list is active.
 */
export function ToolAccessSection({
  allowedTools,
  setAllowedTools,
  blockedTools,
  setBlockedTools,
}: ToolAccessSectionProps) {
  const { tools, loading } = useMCPTools();
  const [search, setSearch] = useState('');

  // Fully-qualified `server.tool` keys grouped by server, `_internal` first.
  const groups = useMemo(() => {
    const byServer = new Map<string, { fq: string; name: string; description: string }[]>();
    for (const t of tools) {
      const server = t.server || '_unknown';
      const fq = `${server}.${t.name}`;
      const arr = byServer.get(server) ?? [];
      arr.push({ fq, name: t.name, description: t.description });
      byServer.set(server, arr);
    }
    return Array.from(byServer.entries()).sort(([a], [b]) => {
      if (a === '_internal') return -1;
      if (b === '_internal') return 1;
      return a.localeCompare(b);
    });
  }, [tools]);

  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groups;
    const needle = search.trim().toLowerCase();
    return groups
      .map(([server, items]) => [
        server,
        items.filter(
          i =>
            i.fq.toLowerCase().includes(needle) ||
            i.name.toLowerCase().includes(needle) ||
            i.description.toLowerCase().includes(needle),
        ),
      ] as const)
      .filter(([, items]) => items.length > 0);
  }, [groups, search]);

  const allowedSet = useMemo(() => new Set(allowedTools ?? []), [allowedTools]);
  const blockedSet = useMemo(() => new Set(blockedTools), [blockedTools]);
  const allowlistActive = allowedTools !== null;

  const toolState = (fq: string): ToolState => {
    if (blockedSet.has(fq)) return 'block';
    if (allowedSet.has(fq)) return 'allow';
    return 'default';
  };

  // Mirrors the backend gate: blocked wins, then allow-list, else on.
  const effectiveOn = (fq: string): boolean => {
    if (blockedSet.has(fq)) return false;
    if (allowlistActive && !allowedSet.has(fq)) return false;
    return true;
  };

  const commit = (nextAllowed: Set<string>, nextBlocked: Set<string>) => {
    // Never emit []; an empty allow-list collapses to null (= all on).
    setAllowedTools(nextAllowed.size > 0 ? Array.from(nextAllowed) : null);
    setBlockedTools(Array.from(nextBlocked));
  };

  const setToolState = (fq: string, next: ToolState) => {
    const nextAllowed = new Set(allowedSet);
    const nextBlocked = new Set(blockedSet);
    nextAllowed.delete(fq);
    nextBlocked.delete(fq);
    if (next === 'allow') nextAllowed.add(fq);
    else if (next === 'block') nextBlocked.add(fq);
    commit(nextAllowed, nextBlocked);
  };

  // Batch-apply a state to every tool in a group (per-group bulk action).
  const setGroupState = (fqs: string[], next: ToolState) => {
    const nextAllowed = new Set(allowedSet);
    const nextBlocked = new Set(blockedSet);
    for (const fq of fqs) {
      nextAllowed.delete(fq);
      nextBlocked.delete(fq);
      if (next === 'allow') nextAllowed.add(fq);
      else if (next === 'block') nextBlocked.add(fq);
    }
    commit(nextAllowed, nextBlocked);
  };

  const resetAll = () => {
    setAllowedTools(null);
    setBlockedTools([]);
  };

  const allowCount = allowedTools?.length ?? 0;
  const blockCount = blockedTools.length;
  const hasOverrides = allowlistActive || blockCount > 0;

  return (
    <div className="profile-form-group profile-nested profile-tool-access">
      <div className="profile-tool-access-head">
        <span className="profile-tool-access-title">
          <Lock size={14} />
          Tool Access
        </span>
        <span className="profile-tool-access-status">
          {allowlistActive
            ? 'Allow-list active — only Allowed tools run.'
            : 'All tools run except Blocked.'}
        </span>
        {hasOverrides && (
          <button
            type="button"
            className="profile-tool-reset"
            onClick={resetAll}
            title="Clear all Allow/Block overrides"
          >
            <RotateCcw size={12} />
            Reset
          </button>
        )}
      </div>

      <div className="profile-tool-list">
        <div className="profile-tool-search">
          <Search size={13} />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter tools…"
          />
        </div>

        {loading && <div className="profile-tool-empty">Loading tools…</div>}
        {!loading && filteredGroups.length === 0 && (
          <div className="profile-tool-empty">
            {tools.length === 0
              ? 'No tools available. Connect an MCP server to surface tools here.'
              : 'No tools match your search.'}
          </div>
        )}

        {!loading &&
          filteredGroups.map(([server, items]) => (
            <div key={server} className="profile-tool-group">
              <div className="profile-tool-group-header">
                <span className="profile-tool-group-name">{server}</span>
                <span className="profile-tool-group-count">{items.length}</span>
                {server === '_internal' && (
                  <span className="profile-tool-group-hint">built-in</span>
                )}
                <div className="profile-tool-group-actions" aria-label={`Set all ${server} tools`}>
                  {SEGMENTS.map(seg => (
                    <button
                      key={seg.value}
                      type="button"
                      className={`profile-tool-group-btn ${seg.value}`}
                      onClick={() => setGroupState(items.map(i => i.fq), seg.value)}
                      title={`Set all to ${seg.label}`}
                    >
                      {seg.label}
                    </button>
                  ))}
                </div>
              </div>
              {items.map(t => {
                const state = toolState(t.fq);
                const off = !effectiveOn(t.fq);
                return (
                  <div
                    key={t.fq}
                    className={`profile-tool-row ${off ? 'is-off' : ''}`}
                  >
                    <div className="profile-tool-info">
                      <span className="profile-tool-name" title={t.fq}>
                        {t.name}
                        {off && <span className="profile-tool-off">off</span>}
                      </span>
                      {t.description && (
                        <span className="profile-tool-desc" title={t.description}>
                          {t.description}
                        </span>
                      )}
                    </div>
                    <div
                      className="profile-tool-seg"
                      role="radiogroup"
                      aria-label={`Access for ${t.name}`}
                    >
                      {SEGMENTS.map(seg => (
                        <button
                          key={seg.value}
                          type="button"
                          role="radio"
                          aria-checked={state === seg.value}
                          className={`profile-tool-seg-btn ${seg.value} ${
                            state === seg.value ? 'active' : ''
                          }`}
                          onClick={() => setToolState(t.fq, seg.value)}
                        >
                          {seg.label}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
      </div>

      <span className="profile-form-hint">
        {allowCount} allowed · {blockCount} blocked
      </span>
    </div>
  );
}

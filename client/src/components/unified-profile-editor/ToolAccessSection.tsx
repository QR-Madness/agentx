import { useMemo, useState } from 'react';
import { ShieldOff, Lock, Search, X } from 'lucide-react';
import { useMCPTools } from '../../lib/hooks';

interface ToolAccessSectionProps {
  /**
   * `null` ⇔ "Allow all tools"; an array switches to whitelist mode. The
   * empty array is a valid whitelist (block everything) but the UI nudges
   * users toward picking at least one.
   */
  allowedTools: string[] | null;
  setAllowedTools: (next: string[] | null) => void;
  /** Always an array (server defaults to []). Wins over allowedTools. */
  blockedTools: string[];
  setBlockedTools: (next: string[]) => void;
}

/**
 * Phase 18.9.x — per-profile tool gating UI.
 *
 * Sits inside the Profile Editor's Capabilities section, rendered only when
 * `enableTools` is on. Mirrors the chip-toggle pattern from the Toolkit's
 * Access tab (`ToolkitPage.tsx::AccessView`) so the two surfaces feel like
 * one feature: that one gates whole servers per agent, this one refines
 * within a single agent down to individual tools.
 */
export function ToolAccessSection({
  allowedTools,
  setAllowedTools,
  blockedTools,
  setBlockedTools,
}: ToolAccessSectionProps) {
  const { tools, loading } = useMCPTools();
  const [search, setSearch] = useState('');

  // Fully-qualified `server.tool` keys — what the backend matches against
  // (`Agent._get_tools_for_provider`). Built-in tools surface as
  // `_internal.<name>`. We group by server for the checklist.
  const groups = useMemo(() => {
    const byServer = new Map<string, { fq: string; name: string; description: string }[]>();
    for (const t of tools) {
      const server = t.server || '_unknown';
      const fq = `${server}.${t.name}`;
      const arr = byServer.get(server) ?? [];
      arr.push({ fq, name: t.name, description: t.description });
      byServer.set(server, arr);
    }
    // Internal tools first so users notice them; rest alphabetical.
    const ordered = Array.from(byServer.entries()).sort(([a], [b]) => {
      if (a === '_internal') return -1;
      if (b === '_internal') return 1;
      return a.localeCompare(b);
    });
    return ordered;
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
            i.description.toLowerCase().includes(needle),
        ),
      ] as const)
      .filter(([, items]) => items.length > 0);
  }, [groups, search]);

  const allowAllMode = allowedTools === null;
  const allowedSet = useMemo(() => new Set(allowedTools ?? []), [allowedTools]);
  const blockedSet = useMemo(() => new Set(blockedTools), [blockedTools]);

  const toggleAllowed = (fq: string) => {
    const next = new Set(allowedSet);
    if (next.has(fq)) next.delete(fq);
    else next.add(fq);
    setAllowedTools(Array.from(next));
  };

  const toggleBlocked = (fq: string) => {
    const next = new Set(blockedSet);
    if (next.has(fq)) next.delete(fq);
    else next.add(fq);
    setBlockedTools(Array.from(next));
  };

  const setMode = (mode: 'all' | 'whitelist') => {
    if (mode === 'all') setAllowedTools(null);
    else setAllowedTools(allowedTools ?? []);
  };

  return (
    <div className="profile-form-group profile-nested profile-tool-access">
      <label className="profile-tool-access-title">
        <Lock size={14} />
        Tool Access
      </label>

      {/* Mode toggle — mirrors the "All agents / per-agent" pattern from
          ToolkitPage::AccessView (chip-style segmented control). */}
      <div className="profile-tool-access-mode" role="radiogroup" aria-label="Tool access mode">
        <button
          type="button"
          role="radio"
          aria-checked={allowAllMode}
          className={`profile-tool-mode-chip ${allowAllMode ? 'solid' : ''}`}
          onClick={() => setMode('all')}
        >
          Allow all tools
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={!allowAllMode}
          className={`profile-tool-mode-chip ${!allowAllMode ? 'solid' : ''}`}
          onClick={() => setMode('whitelist')}
        >
          Limit to selected
        </button>
      </div>
      <span className="profile-form-hint">
        Blocked tools always win over allowed tools.
      </span>

      {/* Allowed-tools checklist (whitelist mode only) */}
      {!allowAllMode && (
        <div className="profile-tool-checklist">
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
                  {server === '_internal' && (
                    <span className="profile-tool-group-hint">
                      AgentX built-in tools
                    </span>
                  )}
                </div>
                <ul className="profile-tool-list">
                  {items.map(t => {
                    const checked = allowedSet.has(t.fq);
                    const isBlocked = blockedSet.has(t.fq);
                    return (
                      <li key={t.fq} className="profile-tool-row">
                        <label className="profile-tool-checkbox">
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={isBlocked}
                            onChange={() => toggleAllowed(t.fq)}
                          />
                          <span className="profile-tool-name">{t.name}</span>
                          {isBlocked && (
                            <span className="profile-tool-blocked-tag">blocked</span>
                          )}
                        </label>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
        </div>
      )}

      {/* Blocked-tools list — always visible. Block wins so users always have
          a denylist regardless of mode. */}
      <div className="profile-tool-blocked">
        <div className="profile-tool-blocked-header">
          <ShieldOff size={13} />
          <span>Blocked tools</span>
          <span className="profile-form-hint">({blockedTools.length})</span>
        </div>
        {blockedTools.length === 0 ? (
          <div className="profile-tool-empty profile-tool-blocked-empty">
            Nothing blocked. Click a tool below to add it.
          </div>
        ) : (
          <div className="profile-tool-blocked-chips">
            {blockedTools.map(fq => (
              <button
                key={fq}
                type="button"
                className="profile-tool-blocked-chip"
                onClick={() => toggleBlocked(fq)}
                title="Click to unblock"
              >
                {fq}
                <X size={11} />
              </button>
            ))}
          </div>
        )}

        {/* Inline picker: every tool, click to toggle block. Reuses the same
            groups list so users don't need a second search box. */}
        <details className="profile-tool-block-picker">
          <summary>Add a tool to block</summary>
          <div className="profile-tool-block-picker-list">
            {groups.map(([server, items]) => (
              <div key={server} className="profile-tool-group">
                <div className="profile-tool-group-header">
                  <span className="profile-tool-group-name">{server}</span>
                </div>
                <ul className="profile-tool-list">
                  {items.map(t => {
                    const isBlocked = blockedSet.has(t.fq);
                    return (
                      <li key={t.fq} className="profile-tool-row">
                        <label className="profile-tool-checkbox">
                          <input
                            type="checkbox"
                            checked={isBlocked}
                            onChange={() => toggleBlocked(t.fq)}
                          />
                          <span className="profile-tool-name">{t.name}</span>
                        </label>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}

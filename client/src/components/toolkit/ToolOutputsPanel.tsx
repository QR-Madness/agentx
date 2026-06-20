/**
 * ToolOutputsPanel — a debug browser over the persisted tool-output store.
 *
 * Oversized tool results (>~12KB) are stashed in Redis with a TTL so the agent
 * (and the chat's inline "View Output") can page back into them. This panel is
 * the *global* view of that store: list every stashed output, filter by name,
 * read the full body, and prune. Backend: GET/DELETE /api/tool-outputs[/{key}].
 */

import { useCallback, useMemo, useState } from 'react';
import { RefreshCw, Trash2, Copy, Check, Search, FileStack, FileText } from 'lucide-react';
import { api } from '../../lib/api';
import { useApi } from '../../lib/hooks';
import { useNotify } from '../../contexts/NotificationContext';
import { useConfirm } from '../ui/ConfirmDialog';
import { Button } from '../ui';

type OutputMeta = {
  key: string;
  tool_name: string;
  tool_call_id: string;
  size_chars: number;
  stored_at: string;
};

function humanizeBytes(chars: number): string {
  if (chars < 1024) return `${chars} B`;
  if (chars < 1024 * 1024) return `${(chars / 1024).toFixed(1)} KB`;
  return `${(chars / (1024 * 1024)).toFixed(1)} MB`;
}

function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

function tryFormatJson(content: string): { formatted: string; isJson: boolean } {
  try {
    return { formatted: JSON.stringify(JSON.parse(content), null, 2), isJson: true };
  } catch {
    return { formatted: content, isJson: false };
  }
}

export function ToolOutputsPanel() {
  const notify = useNotify();
  const confirm = useConfirm();
  const [filter, setFilter] = useState('');
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // The store keys look like `tool_output:<tool>:<id>`; matching the raw filter
  // text as a substring pattern is enough for a debug list.
  const pattern = filter.trim() ? `*${filter.trim()}*` : undefined;

  const list = useApi(() => api.listToolOutputs(pattern), [pattern]);
  const detail = useApi(
    () => (selectedKey ? api.getToolOutput(selectedKey) : Promise.resolve(null)),
    [selectedKey],
    { enabled: !!selectedKey },
  );

  const outputs = useMemo<OutputMeta[]>(() => list.data?.outputs ?? [], [list.data]);
  const selected = detail.data;
  const { formatted, isJson } = useMemo(
    () => tryFormatJson(selected?.content ?? ''),
    [selected?.content],
  );

  const handleCopy = useCallback(async () => {
    if (!selected?.content) return;
    try {
      await navigator.clipboard.writeText(selected.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      notify.notifyError(err, 'Copy failed');
    }
  }, [selected?.content, notify]);

  const handleDelete = useCallback(
    async (key: string, toolName: string) => {
      const ok = await confirm({
        title: 'Delete stored output?',
        body: `Remove the stashed "${toolName}" output. This only clears the debug cache — it can't be undone.`,
        confirmLabel: 'Delete',
        danger: true,
      });
      if (!ok) return;
      try {
        await api.deleteToolOutput(key);
        if (selectedKey === key) setSelectedKey(null);
        list.refresh();
      } catch (err) {
        notify.notifyError(err, 'Delete failed');
      }
    },
    [confirm, list, notify, selectedKey],
  );

  const handleClearShown = useCallback(async () => {
    if (!outputs.length) return;
    const ok = await confirm({
      title: `Clear ${outputs.length} stored output${outputs.length === 1 ? '' : 's'}?`,
      body: 'Removes every output currently listed (respecting the filter). This only clears the debug cache.',
      confirmLabel: 'Clear all',
      danger: true,
    });
    if (!ok) return;
    try {
      await Promise.all(outputs.map((o) => api.deleteToolOutput(o.key)));
      setSelectedKey(null);
      list.refresh();
      notify.notifySuccess(`Cleared ${outputs.length} stored output${outputs.length === 1 ? '' : 's'}.`);
    } catch (err) {
      notify.notifyError(err, 'Clear failed');
    }
  }, [outputs, confirm, list, notify]);

  return (
    <div className="flex h-full flex-col bg-surface-base text-fg">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <FileStack size={18} className="text-accent" />
        <h2 className="text-base font-semibold">Tool Outputs</h2>
        <span className="text-xs text-fg-muted">{list.data?.count ?? 0} stored</span>
        <div className="ml-auto flex items-center gap-1.5">
          <button
            className="rounded-md p-1.5 text-fg-secondary hover:bg-surface-hover hover:text-fg"
            onClick={() => list.refresh()}
            title="Refresh"
            aria-label="Refresh"
          >
            <RefreshCw size={15} className={list.loading ? 'animate-spin' : ''} />
          </button>
          <Button variant="ghost" onClick={handleClearShown} disabled={!outputs.length}>
            <Trash2 size={14} />
            <span className="ml-1">Clear shown</span>
          </Button>
        </div>
      </div>

      {/* Filter */}
      <div className="border-b border-line px-4 py-2">
        <div className="flex items-center gap-2 rounded-md border border-line bg-surface-sunken px-2.5 py-1.5">
          <Search size={14} className="text-fg-muted" />
          <input
            className="w-full bg-transparent text-sm text-fg placeholder:text-fg-muted focus:outline-none"
            placeholder="Filter by tool name…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>

      {/* Body: list + viewer */}
      <div className="flex min-h-0 flex-1">
        {/* List rail */}
        <div className="w-2/5 min-w-[14rem] max-w-sm overflow-y-auto border-r border-line">
          {list.loading && !outputs.length ? (
            <p className="p-4 text-sm text-fg-muted">Loading…</p>
          ) : list.error ? (
            <p className="p-4 text-sm text-error">Failed to load stored outputs.</p>
          ) : !outputs.length ? (
            <div className="flex flex-col items-center gap-2 p-8 text-center text-fg-muted">
              <FileText size={28} className="opacity-40" />
              <p className="text-sm">No stored outputs.</p>
              <p className="text-xs">Outputs are stashed only when a tool result is large, and expire on a TTL.</p>
            </div>
          ) : (
            <ul className="divide-y divide-line">
              {outputs.map((o) => (
                <li key={o.key}>
                  <button
                    className={`group flex w-full items-start gap-2 px-3 py-2.5 text-left hover:bg-surface-hover ${
                      selectedKey === o.key ? 'bg-surface-raised' : ''
                    }`}
                    onClick={() => setSelectedKey(o.key)}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-fg">{o.tool_name}</div>
                      <div className="mt-0.5 flex items-center gap-2 text-xs text-fg-muted">
                        <span>{humanizeBytes(o.size_chars)}</span>
                        <span>·</span>
                        <span>{relativeTime(o.stored_at)}</span>
                      </div>
                    </div>
                    <span
                      role="button"
                      tabIndex={0}
                      className="rounded p-1 text-fg-muted opacity-0 hover:bg-surface-overlay hover:text-error group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(o.key, o.tool_name);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.stopPropagation();
                          e.preventDefault();
                          handleDelete(o.key, o.tool_name);
                        }
                      }}
                      title="Delete this output"
                      aria-label={`Delete ${o.tool_name} output`}
                    >
                      <Trash2 size={14} />
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Viewer */}
        <div className="flex min-w-0 flex-1 flex-col">
          {!selectedKey ? (
            <div className="flex flex-1 items-center justify-center p-8 text-sm text-fg-muted">
              Select an output to inspect it.
            </div>
          ) : detail.loading ? (
            <p className="p-4 text-sm text-fg-muted">Loading output…</p>
          ) : detail.error ? (
            <p className="p-4 text-sm text-error">Failed to load this output (it may have expired).</p>
          ) : (
            <>
              <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
                <span className="truncate text-sm font-medium text-fg">{selected?.tool_name}</span>
                {selected?.size_chars != null && (
                  <span className="text-xs text-fg-muted">{humanizeBytes(selected.size_chars)}</span>
                )}
                <div className="ml-auto flex items-center gap-1.5">
                  <Button variant="ghost" onClick={handleCopy}>
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    <span className="ml-1">{copied ? 'Copied' : 'Copy'}</span>
                  </Button>
                  {selectedKey && (
                    <Button
                      variant="ghost"
                      onClick={() => handleDelete(selectedKey, selected?.tool_name ?? 'output')}
                    >
                      <Trash2 size={14} />
                    </Button>
                  )}
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-auto p-4">
                <pre
                  className={`whitespace-pre-wrap break-words rounded-lg bg-surface-sunken p-4 font-mono text-[0.8125rem] leading-relaxed ${
                    isJson ? 'text-fg' : 'text-fg-secondary'
                  }`}
                >
                  {formatted || '(empty)'}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

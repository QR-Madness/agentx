/**
 * LogsPanel — live server log viewer (the Log panel).
 *
 * Streams the API's in-memory ring buffer over SSE, color-coded by category and
 * level. Built for volume: incoming lines are batched on requestAnimationFrame
 * (one render per frame, not per line) and the rendered set is capped to a
 * sliding window. Filters (level / category / run-id / search) apply client-side
 * over the buffered window; clicking a row's run tag isolates that turn.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, Trash2, X, Search, Archive, Download, ChevronRight, ChevronDown, Lock, ShieldCheck, ShieldOff } from 'lucide-react';
import { api, type LogRecord, type LogArchiveSegment, type LogArchiveStatus } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { categoryMeta, levelColor, LOG_LEVELS } from '../../lib/logCategories';
import './LogsPanel.css';

const MAX_ROWS = 600; // sliding DOM window; older history lives in the archive
const ARCHIVE_PAGE = 10; // day-chunks revealed per scroll step

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString(undefined, { hour12: false }) + '.' +
    String(d.getMilliseconds()).padStart(3, '0');
}

/** Friendly day label for a daily chunk (`agentx-YYYY-MM-DD.log.gz[.enc]`).
 *  Legacy size-based segments (`agentx.log.N.gz`) fall back to their mtime. */
function segmentDayLabel(s: LogArchiveSegment): string {
  const m = s.name.match(/agentx-(\d{4})-(\d{2})-(\d{2})\.log/);
  const d = m ? new Date(`${m[1]}-${m[2]}-${m[3]}T00:00:00`) : new Date(s.modified * 1000);
  if (Number.isNaN(d.getTime())) return s.name;
  return d.toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}

function shortRun(runId?: string | null): string {
  if (!runId) return '';
  const tail = runId.split('_').pop() || runId;
  return tail.slice(0, 6);
}

/** Compact encryption/vault state for the top of the archive drawer. */
function ArchiveVaultStrip({ status }: { status: LogArchiveStatus }) {
  const { encryption_enabled, keyring_present, unlocked, sealed_segments, pending_segments, retention_days } = status;

  let icon = <ShieldOff size={13} />;
  let tone = 'neutral';
  let label = 'Plaintext archives';
  let detail = 'Encryption is off';

  if (!encryption_enabled) {
    detail = 'AGENTX_LOG_ARCHIVE_ENCRYPT=false';
  } else if (!keyring_present) {
    label = 'Not encrypted yet';
    detail = 'Set a password to start sealing logs';
  } else if (unlocked) {
    icon = <ShieldCheck size={13} />;
    tone = 'ok';
    label = 'Encrypted · unlocked';
    detail = `${sealed_segments} sealed${pending_segments ? `, ${pending_segments} pending` : ''} · ${retention_days}d retention`;
  } else {
    icon = <Lock size={13} />;
    tone = 'locked';
    label = 'Encrypted · locked';
    detail = 'Re-authenticate to download sealed days';
  }

  return (
    <div className={`logs-vault logs-vault--${tone}`} title={detail}>
      {icon}
      <span className="logs-vault-label">{label}</span>
      <span className="logs-vault-detail">{detail}</span>
    </div>
  );
}

export function LogsPanel() {
  const notify = useNotify();
  const [logs, setLogs] = useState<LogRecord[]>([]);
  const [available, setAvailable] = useState(true);
  const [live, setLive] = useState(true);
  const [paused, setPaused] = useState(false); // auto-scroll paused (user scrolled up)

  const [level, setLevel] = useState('');
  const [category, setCategory] = useState('');
  const [runId, setRunId] = useState('');
  const [search, setSearch] = useState('');

  const [showArchive, setShowArchive] = useState(false);
  const [segments, setSegments] = useState<LogArchiveSegment[]>([]);
  const [archiveStatus, setArchiveStatus] = useState<LogArchiveStatus | null>(null);
  const [visibleDays, setVisibleDays] = useState(ARCHIVE_PAGE);
  const archiveSentinel = useRef<HTMLDivElement | null>(null);
  const archiveScrollRef = useRef<HTMLDivElement | null>(null);

  // Rows carrying an oversized `detail` payload (e.g. a full LLM request) start
  // collapsed; the user expands the ones they care about.
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set());
  const toggleExpanded = useCallback((id: number) => {
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const incoming = useRef<LogRecord[]>([]);
  const rafId = useRef<number | null>(null);
  const lastId = useRef(0);

  // Flush the rAF batch into state, dedup by id, cap to the window.
  const flush = useCallback(() => {
    rafId.current = null;
    if (incoming.current.length === 0) return;
    const batch = incoming.current;
    incoming.current = [];
    setLogs((prev) => {
      const merged = prev.concat(batch.filter((r) => r.id > lastId.current));
      for (const r of batch) lastId.current = Math.max(lastId.current, r.id);
      return merged.length > MAX_ROWS ? merged.slice(merged.length - MAX_ROWS) : merged;
    });
  }, []);

  const schedule = useCallback(() => {
    if (rafId.current == null) rafId.current = requestAnimationFrame(flush);
  }, [flush]);

  // Initial snapshot.
  useEffect(() => {
    let cancelled = false;
    api
      .listLogs({ limit: 500 })
      .then((res) => {
        if (cancelled) return;
        setAvailable(res.available);
        setLogs(res.logs);
        lastId.current = res.logs.length ? res.logs[res.logs.length - 1].id : 0;
      })
      .catch((err) => {
        if (!cancelled) notify.notifyError(err);
      });
    return () => {
      cancelled = true;
    };
  }, [notify]);

  // Live stream (toggle).
  useEffect(() => {
    if (!live || !available) return;
    const handle = api.streamLogs({
      onLog: (rec) => {
        incoming.current.push(rec);
        schedule();
      },
      onError: () => {
        /* transient; the toggle lets the user retry */
      },
    });
    return () => {
      handle.abort();
      if (rafId.current != null) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    };
  }, [live, available, schedule]);

  // Auto-scroll to bottom unless the user scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && !paused) el.scrollTop = el.scrollHeight;
  }, [logs, paused]);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setPaused(!atBottom);
  }, []);

  const filtered = useMemo(() => {
    const lv = level.toUpperCase();
    const needle = search.trim().toLowerCase();
    return logs.filter((r) => {
      if (lv && r.level !== lv) return false;
      if (category && r.category !== category) return false;
      if (runId && r.run_id !== runId) return false;
      if (needle && !r.message.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [logs, level, category, runId, search]);

  const categories = useMemo(() => {
    const set = new Set<string>();
    for (const r of logs) set.add(r.category);
    return Array.from(set).sort();
  }, [logs]);

  const clearView = () => {
    setLogs([]);
    lastId.current = 0;
  };

  const toggleArchive = async () => {
    const next = !showArchive;
    setShowArchive(next);
    if (next) {
      setVisibleDays(ARCHIVE_PAGE);
      try {
        const [list, status] = await Promise.all([
          api.listLogArchive(),
          api.getLogArchiveStatus().catch(() => null),
        ]);
        setSegments(list.segments);
        setArchiveStatus(status);
      } catch (err) {
        notify.notifyError(err);
      }
    }
  };

  // Scroll-to-load: reveal another page of older day-chunks when the sentinel
  // at the bottom of the archive list scrolls into view.
  useEffect(() => {
    if (!showArchive) return;
    const node = archiveSentinel.current;
    if (!node) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setVisibleDays((n) => (n < segments.length ? n + ARCHIVE_PAGE : n));
        }
      },
      { root: archiveScrollRef.current, threshold: 0.1 }
    );
    io.observe(node);
    return () => io.disconnect();
  }, [showArchive, segments.length]);

  if (!available) {
    return (
      <div className="logs-panel logs-panel--empty">
        <p>The log API is disabled on this server.</p>
        <p className="logs-empty-hint">Set <code>AGENTX_LOG_API_ENABLED=true</code> to enable it.</p>
      </div>
    );
  }

  return (
    <div className="logs-panel">
      <div className="logs-toolbar">
        <button
          className={`logs-btn ${live ? 'logs-btn--active' : ''}`}
          onClick={() => setLive((v) => !v)}
          title={live ? 'Pause live tail' : 'Resume live tail'}
        >
          {live ? <Pause size={14} /> : <Play size={14} />}
          <span>{live ? 'Live' : 'Paused'}</span>
        </button>

        <select className="logs-select" value={level} onChange={(e) => setLevel(e.target.value)} title="Level">
          <option value="">All levels</option>
          {LOG_LEVELS.map((l) => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>

        <select className="logs-select" value={category} onChange={(e) => setCategory(e.target.value)} title="Category">
          <option value="">All categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>{categoryMeta(c).label}</option>
          ))}
        </select>

        <div className="logs-search">
          <Search size={13} />
          <input
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {runId && (
          <button className="logs-chip" onClick={() => setRunId('')} title="Clear run filter">
            run:{shortRun(runId)} <X size={11} />
          </button>
        )}

        <div className="logs-toolbar-spacer" />

        <button className="logs-btn" onClick={toggleArchive} title="Archive">
          <Archive size={14} />
        </button>
        <button className="logs-btn" onClick={clearView} title="Clear view">
          <Trash2 size={14} />
        </button>
      </div>

      {showArchive && (
        <div className="logs-archive" ref={archiveScrollRef}>
          {archiveStatus && <ArchiveVaultStrip status={archiveStatus} />}
          {segments.length === 0 ? (
            <span className="logs-archive-empty">No archived segments yet.</span>
          ) : (
            <>
              {segments.slice(0, visibleDays).map((s) => {
                const locked = !!s.encrypted && !archiveStatus?.unlocked;
                const size = `${(s.size / 1024).toFixed(1)} KB`;
                return s.encrypted ? (
                  <button
                    key={s.name}
                    type="button"
                    className="logs-archive-item"
                    title={`${s.name} — ${locked ? 'encrypted, locked (re-authenticate to download)' : 'encrypted, decrypted on download'}`}
                    onClick={async () => {
                      try {
                        await api.downloadLogArchive(s);
                      } catch (err) {
                        notify.notifyError(err);
                      }
                    }}
                  >
                    <Lock size={12} className={locked ? 'logs-archive-lock--locked' : 'logs-archive-lock'} />
                    <span className="logs-archive-name">{segmentDayLabel(s)}</span>
                    <span className="logs-archive-size">{size}</span>
                  </button>
                ) : (
                  <a key={s.name} className="logs-archive-item" href={api.logArchiveUrl(s.name)} download title={s.name}>
                    <Download size={12} />
                    <span className="logs-archive-name">{segmentDayLabel(s)}</span>
                    <span className="logs-archive-size">{size}</span>
                  </a>
                );
              })}
              {visibleDays < segments.length && (
                <div ref={archiveSentinel} className="logs-archive-more">
                  Loading earlier days… ({segments.length - visibleDays} more)
                </div>
              )}
            </>
          )}
        </div>
      )}

      <div className="logs-stream" ref={scrollRef} onScroll={onScroll}>
        {filtered.length === 0 ? (
          <div className="logs-empty-rows">No log lines{logs.length ? ' match the filters' : ' yet'}.</div>
        ) : (
          filtered.map((r) => {
            const cat = categoryMeta(r.category);
            return (
              <div key={r.id} className={`logs-row logs-row--${r.level.toLowerCase()}`}>
                <span className="logs-time">{formatTime(r.ts)}</span>
                <span className="logs-level" style={{ color: levelColor(r.level) }}>{r.level}</span>
                <span className="logs-cat" style={{ color: cat.color }} title={r.logger}>
                  {cat.emoji} {cat.label}
                </span>
                {r.run_id ? (
                  <button
                    className="logs-runtag"
                    onClick={() => setRunId(r.run_id || '')}
                    title="Filter to this run"
                  >
                    run:{shortRun(r.run_id)}
                  </button>
                ) : (
                  <span className="logs-runtag logs-runtag--none" />
                )}
                <span className="logs-msg">
                  {r.detail ? (
                    <button
                      type="button"
                      className="logs-detail-toggle"
                      onClick={() => toggleExpanded(r.id)}
                      aria-expanded={expanded.has(r.id)}
                      title={expanded.has(r.id) ? 'Hide payload' : 'Show full payload'}
                    >
                      {expanded.has(r.id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                      {r.message}
                    </button>
                  ) : (
                    r.message
                  )}
                  {r.exc && <pre className="logs-exc">{r.exc}</pre>}
                  {r.detail && expanded.has(r.id) && <pre className="logs-detail">{r.detail}</pre>}
                </span>
              </div>
            );
          })
        )}
      </div>

      <div className="logs-footer">
        <span>{filtered.length} / {logs.length} lines{paused ? ' · scroll paused' : ''}</span>
      </div>
    </div>
  );
}

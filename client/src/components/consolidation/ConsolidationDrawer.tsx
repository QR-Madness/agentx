/**
 * ConsolidationDrawer — the ⚡ memory-consolidation surface (right drawer).
 *
 * Idle → Start + last-run summary. Running → a genuinely moving progress bar
 * (per-conversation "N of M" + trickle), live metric chips, rotating
 * memory-flavored messages, a live issues pip, and pipeline step-pills. Done →
 * either a completion flourish (work happened) or a clean "Memory's up to date"
 * (the common idempotent no-op). No fake Stop: closing runs it in the
 * background (the ⚡ keeps pulsing; the provider re-attaches on reopen).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Zap, CheckCircle, XCircle, Loader2, AlertTriangle, ChevronRight } from 'lucide-react';
import { api } from '../../lib/api';
import type { ConsolidationProgressEvent, JobHistory } from '../../lib/api/types';
import { useConsolidation } from '../../contexts/ConsolidationContext';
import { ProgressBar } from '../ui/ProgressBar';
import { nextMessage } from './consolidation-messages';
import { parseNofM, compactNumber } from './progress';

const JOB_LABELS: Record<string, string> = {
  consolidate: 'Extract',
  patterns: 'Patterns',
  promote: 'Promote',
};
const PIPELINE = ['consolidate', 'patterns', 'promote'];

const STAGE_LABELS: Record<string, string> = {
  discovery: 'Finding new conversations…',
  processing: 'Reading turns…',
  storing: 'Storing to memory…',
  complete: 'Wrapping up…',
  starting: 'Starting…',
  done: 'Done',
};

function relativeTime(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return '';
  const s = Math.max(0, (Date.now() - d) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

/** Ease a displayed value toward `target`, and gently creep forward between real
 *  updates so a slow per-conversation gap never looks frozen. */
function useTrickle(target: number, active: boolean): number {
  const [display, setDisplay] = useState(target);
  const targetRef = useRef(target);
  targetRef.current = target;
  useEffect(() => {
    if (!active) { setDisplay(target); return; }
    // Snap up to any real progress immediately.
    setDisplay((d) => (target > d ? target : d));
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    const id = window.setInterval(() => {
      setDisplay((d) => {
        const t = targetRef.current;
        if (d < t) return t; // real progress overtook the creep
        const ceil = Math.min(t + 0.08, 0.97); // never reach the next real step
        return d < ceil ? d + (ceil - d) * 0.06 : d;
      });
    }, 250);
    return () => window.clearInterval(id);
  }, [active, target]);
  return Math.max(display, target);
}

export function ConsolidationDrawer(_props: { onClose: () => void }) {
  const c = useConsolidation();
  const progress = c.progress as ConsolidationProgressEvent | null;

  const [message, setMessage] = useState(() => nextMessage());
  const [lastRun, setLastRun] = useState<JobHistory | null>(null);
  const [errorsOpen, setErrorsOpen] = useState(false);

  // Rotate the hero message while streaming.
  useEffect(() => {
    if (!c.isStreaming) return;
    const id = window.setInterval(() => setMessage((m) => nextMessage(m)), 2500);
    return () => window.clearInterval(id);
  }, [c.isStreaming]);

  // Fetch the last run for the idle summary (and refresh after a run completes).
  useEffect(() => {
    if (c.isStreaming) return;
    let cancelled = false;
    api.getJob('consolidate')
      .then((d) => { if (!cancelled) setLastRun(d.history?.[0] ?? null); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [c.isStreaming, c.result]);

  // Determinate fraction from "N of M" (floor at the completed conversations).
  const base = useMemo(() => {
    const f = parseNofM(progress?.conversation);
    if (f == null) return c.stage === 'discovery' ? 0 : 0;
    return f;
  }, [progress?.conversation, c.stage]);
  const indeterminate = c.isStreaming && (c.stage === 'discovery' || c.stage === 'starting' || base === 0);
  const value = useTrickle(base, c.isStreaming && !indeterminate);

  const turns = progress?.turns_processed ?? 0;
  const llm = progress?.llm_calls ?? 0;
  const tokens = progress?.tokens ?? 0;
  const errorsCount = progress?.errors_count ?? 0;

  // The `done` event nests each job's result under `results.<job>`; the consolidate
  // job carries the counts + metrics. Deriving "did work" from the result (not from
  // observed stage transitions) is robust when the drawer opens after a run finished.
  const result = c.result as Record<string, unknown> | null;
  const num = (v: unknown) => (typeof v === 'number' ? v : 0);
  const cr = ((result?.results as Record<string, unknown> | undefined)?.consolidate ??
    null) as Record<string, unknown> | null;
  const crMetrics = (cr?.metrics ?? {}) as Record<string, unknown>;
  const doneEntities = num(cr?.entities);
  const doneFacts = num(cr?.facts);
  const doneRels = num(cr?.relationships);
  const workTurns = num(crMetrics.turns_total) + num(crMetrics.assistant_turns_total);
  const didWork = workTurns > 0 || doneEntities > 0 || doneFacts > 0 || doneRels > 0;
  const resultErrors = Array.isArray(result?.errors) ? (result!.errors as string[]) : [];
  const success = result?.success !== false;
  const upToDate = !!result && !c.isStreaming && !didWork;

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center gap-2">
        <Zap size={18} className="text-accent" />
        <h2 className="text-base font-semibold text-fg">Memory Consolidation</h2>
        {c.isStreaming && <Loader2 size={14} className="ml-auto animate-spin text-fg-muted" />}
      </div>

      {c.isStreaming ? (
        <div className="flex flex-col gap-4">
          {/* Hero message + real stage/detail */}
          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium text-fg">{message}…</div>
            <div className="flex items-center gap-1 text-xs text-fg-muted">
              <ChevronRight size={12} className="shrink-0" />
              <span>{STAGE_LABELS[c.stage ?? ''] ?? c.stage}</span>
              {progress?.conversation && <span>· {progress.conversation}</span>}
            </div>
          </div>

          <ProgressBar value={value} indeterminate={indeterminate} aria-label="Consolidation progress" />

          {/* Pipeline step-pills */}
          <div className="flex flex-wrap gap-1.5">
            {PIPELINE.map((job) => {
              const isCurrent = c.currentJob === job;
              const idx = PIPELINE.indexOf(job);
              const curIdx = c.currentJob ? PIPELINE.indexOf(c.currentJob) : -1;
              const done = curIdx > idx;
              return (
                <span
                  key={job}
                  className={
                    'rounded-pill px-2 py-0.5 text-2xs font-medium ' +
                    (done
                      ? 'bg-success/15 text-success'
                      : isCurrent
                        ? 'bg-accent/15 text-accent'
                        : 'bg-surface-sunken text-fg-muted')
                  }
                >
                  {JOB_LABELS[job] ?? job}
                </span>
              );
            })}
          </div>

          {/* Live metric chips */}
          <div className="grid grid-cols-3 gap-2">
            <Stat label="turns" value={compactNumber(turns)} />
            <Stat label="LLM calls" value={compactNumber(llm)} />
            <Stat label="tokens" value={compactNumber(tokens)} />
          </div>

          {errorsCount > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-warning">
              <AlertTriangle size={13} className="shrink-0" />
              <span>{errorsCount} issue{errorsCount === 1 ? '' : 's'} (will retry)</span>
            </div>
          )}

          <p className="text-2xs text-fg-muted">
            You can close this — consolidation keeps running in the background.
          </p>
        </div>
      ) : upToDate ? (
        <div className="flex flex-col items-start gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-success">
            <CheckCircle size={18} className="shrink-0" />
            <span>Memory's up to date</span>
          </div>
          <p className="text-xs text-fg-muted">Nothing new to consolidate since the last run.</p>
          <LastRun lastRun={lastRun} />
          <StartButton onClick={() => c.trigger()} label="Run anyway" />
        </div>
      ) : result ? (
        <div className="flex flex-col gap-3">
          <div className={`flex items-center gap-2 text-sm font-medium ${success ? 'text-success' : 'text-error'}`}>
            {success ? <CheckCircle size={18} /> : <XCircle size={18} />}
            <span>{success ? 'Consolidation complete' : 'Consolidation failed'}</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="entities" value={compactNumber(doneEntities)} />
            <Stat label="facts" value={compactNumber(doneFacts)} />
            <Stat label="relationships" value={compactNumber(doneRels)} />
          </div>
          {resultErrors.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <button
                type="button"
                onClick={() => setErrorsOpen((v) => !v)}
                className="flex items-center gap-1.5 text-left text-xs font-medium text-warning hover:text-fg"
              >
                <AlertTriangle size={13} className="shrink-0" />
                {resultErrors.length} issue{resultErrors.length === 1 ? '' : 's'}
                <ChevronRight size={12} className={errorsOpen ? 'rotate-90 transition-transform' : 'transition-transform'} />
              </button>
              {errorsOpen && (
                <ul className="flex flex-col gap-1 rounded-md border border-line bg-surface-sunken p-2">
                  {resultErrors.slice(0, 10).map((e, i) => (
                    <li key={i} className="truncate font-mono text-2xs text-fg-muted" title={e}>{e}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
          <StartButton onClick={() => c.trigger()} label="Run again" />
        </div>
      ) : c.error ? (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-error">
            <XCircle size={18} />
            <span className="min-w-0 truncate" title={c.error}>{c.error}</span>
          </div>
          <StartButton onClick={() => c.trigger()} label="Retry" />
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <p className="text-sm text-fg-secondary">
            Extract entities, facts, and relationships from your recent conversations into memory.
          </p>
          <LastRun lastRun={lastRun} />
          <StartButton onClick={() => c.trigger()} label="Start consolidation" />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 rounded-md bg-surface-sunken px-2 py-2">
      <span className="font-mono text-sm font-semibold tabular-nums text-fg">{value}</span>
      <span className="text-2xs uppercase tracking-caps text-fg-muted">{label}</span>
    </div>
  );
}

function LastRun({ lastRun }: { lastRun: JobHistory | null }) {
  if (!lastRun) return null;
  const facts = Number(lastRun.metrics?.facts_stored ?? 0);
  return (
    <div className="text-xs text-fg-muted">
      Last run · {relativeTime(lastRun.timestamp)}
      {` · ${facts} fact${facts === 1 ? '' : 's'}`}
      {` · ${(lastRun.duration_ms / 1000).toFixed(1)}s`}
    </div>
  );
}

function StartButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center justify-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-fg-inverse transition-colors hover:bg-accent-secondary"
    >
      <Zap size={14} />
      {label}
    </button>
  );
}

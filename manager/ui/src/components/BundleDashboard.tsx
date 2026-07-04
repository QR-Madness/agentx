/* Bundle mode: one deployment → a full dashboard in plain language.
 *
 * Bundle users downloaded a tarball and shouldn't need docker vocabulary —
 * no "cluster", "compose", "containers", "orphans". Repo mode keeps the
 * multi-cluster ClusterCard grid (those users prefer the precise terms).
 */

import { useCallback, useState } from "react";
import { api, type Cluster, type ClusterUsage, type LifecycleAction, type Phase } from "../api";
import { usePolling } from "../hooks";
import { formatBytes } from "./ui";

const PHASE_COPY: Record<Phase, { title: string; blurb: string; pill: string }> = {
  down: {
    title: "Stopped",
    blurb: "AgentX isn't running. Start it to bring everything online.",
    pill: "bg-fg-muted/10 text-fg-muted border-line",
  },
  initializing: {
    title: "Starting up…",
    blurb:
      "Preparing databases. The very first start also downloads AI models (a few GB) — that can take several minutes.",
    pill: "bg-amber-500/15 text-amber-300 border-amber-400/30 animate-pulse",
  },
  up: {
    title: "Running",
    blurb: "All components are healthy.",
    pill: "bg-emerald-500/15 text-emerald-300 border-emerald-400/30",
  },
  degraded: {
    title: "Needs attention",
    blurb:
      "Some components aren't healthy. Right after a start this is usually temporary — check the activity log if it persists.",
    pill: "bg-orange-500/15 text-orange-300 border-orange-400/30",
  },
};

const BUSY_VERB: Record<string, string> = {
  up: "Starting…",
  down: "Stopping…",
  restart: "Restarting…",
  destroy: "Resetting…",
};

const SERVICE_INFO: Record<string, { label: string; desc: string }> = {
  api: { label: "AgentX Server", desc: "the agent platform itself" },
  neo4j: { label: "Knowledge Graph", desc: "long-term memory · Neo4j" },
  postgres: { label: "Memory Store", desc: "vector memory · PostgreSQL" },
  redis: { label: "Cache", desc: "working memory · Redis" },
  nginx: { label: "Gateway", desc: "secure access proxy" },
  cloudflared: { label: "Tunnel", desc: "Cloudflare connection" },
};

const PORT_INFO: [key: keyof Cluster["ports"], label: string][] = [
  ["api", "AgentX API"],
  ["neo4j_http", "Neo4j browser"],
  ["neo4j_bolt", "Neo4j bolt"],
  ["postgres", "PostgreSQL"],
  ["redis", "Redis"],
];

function serviceState(state: string, health: string): { dot: string; word: string } {
  if (state === "running" && (health === "" || health === "healthy"))
    return { dot: "bg-emerald-400", word: "running" };
  if (health === "starting" || state === "restarting" || state === "created")
    return { dot: "bg-amber-400", word: "starting" };
  if (state === "exited") return { dot: "bg-fg-muted", word: "stopped" };
  return { dot: "bg-orange-400", word: health || state };
}

function StatTile({
  label,
  value,
  sub,
  gauge,
}: {
  label: string;
  value: string;
  sub?: string;
  gauge?: number;
}) {
  return (
    <div className="rounded-xl border border-line bg-raised p-4">
      <p className="text-[11px] uppercase tracking-wide text-fg-muted">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-fg">{value}</p>
      {sub && <p className="mt-0.5 text-xs tabular-nums text-fg-muted">{sub}</p>}
      {gauge !== undefined && (
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-sunken">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              gauge > 90 ? "bg-orange-400" : gauge > 70 ? "bg-amber-400" : "bg-accent-2"
            }`}
            style={{ width: `${Math.max(0, Math.min(100, gauge))}%` }}
          />
        </div>
      )}
    </div>
  );
}

export function BundleDashboard({
  cluster,
  busy,
  onAction,
  onDestroy,
  onLogs,
}: {
  cluster: Cluster;
  busy: string | null;
  onAction: (action: LifecycleAction) => void;
  onDestroy: () => void;
  onLogs: () => void;
}) {
  const [usage, setUsage] = useState<ClusterUsage | null>(null);
  const running = cluster.phase !== "down";
  const copy = PHASE_COPY[cluster.phase];

  const pollUsage = useCallback(async () => {
    try {
      setUsage(await api.usage(cluster.name));
    } catch {
      /* best-effort — the status pill carries the real signal */
    }
  }, [cluster.name]);
  usePolling(pollUsage, 5000, running);
  if (!running && usage !== null) setUsage(null);

  const downloading =
    cluster.phase === "initializing" && (usage?.net_rx_rate ?? 0) > 500_000;

  return (
    <div className="flex flex-col gap-4">
      {/* Hero: status + primary actions */}
      <div className="rounded-xl border border-line bg-raised p-5 shadow-md sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2.5">
              <h2 className="text-lg font-semibold tracking-tight text-fg">AgentX</h2>
              <span
                className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${copy.pill}`}
              >
                {copy.title}
              </span>
              {busy && (
                <span className="flex items-center gap-1.5 text-xs text-accent">
                  <span className="size-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                  {BUSY_VERB[busy] ?? `${busy}…`}
                </span>
              )}
            </div>
            <p className="mt-1.5 max-w-xl text-sm text-fg-secondary">{copy.blurb}</p>
            {downloading && usage && (
              <p className="mt-1.5 text-sm tabular-nums text-fg-secondary">
                Downloading AI models — {formatBytes(usage.net_rx_rate)}/s
                <span className="text-fg-muted"> · {formatBytes(usage.net_rx_bytes)} so far</span>
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            {cluster.phase === "up" && (
              <a
                href={cluster.url}
                target="_blank"
                rel="noreferrer"
                className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-4 py-1.5 text-sm font-medium text-fg hover:bg-accent/25"
              >
                Open AgentX ↗
              </a>
            )}
            {!running && (
              <button
                disabled={busy !== null}
                onClick={() => onAction("up")}
                className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-4 text-sm font-medium text-fg hover:bg-accent/25 disabled:opacity-40"
              >
                Start AgentX
              </button>
            )}
            {running && (
              <>
                <button
                  disabled={busy !== null}
                  onClick={() => onAction("restart")}
                  className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg disabled:opacity-40"
                >
                  Restart
                </button>
                <button
                  disabled={busy !== null}
                  onClick={() => onAction("down")}
                  className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg disabled:opacity-40"
                >
                  Stop
                </button>
              </>
            )}
            <button
              onClick={onLogs}
              className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg"
            >
              Activity Log
            </button>
          </div>
        </div>
      </div>

      {/* Resource tiles */}
      {running && usage && (
        <div className="grid gap-4 sm:grid-cols-3">
          <StatTile
            label="Processor"
            value={`${usage.cpu_percent.toFixed(1)}%`}
            gauge={usage.cpu_percent}
          />
          <StatTile
            label="Memory"
            value={formatBytes(usage.mem_used_bytes)}
            sub={usage.mem_limit_bytes > 0 ? `of ${formatBytes(usage.mem_limit_bytes)}` : undefined}
            gauge={usage.mem_percent}
          />
          <StatTile
            label="Network"
            value={`↓ ${formatBytes(usage.net_rx_rate)}/s`}
            sub={`↑ ${formatBytes(usage.net_tx_rate)}/s · total ↓ ${formatBytes(usage.net_rx_bytes)}`}
          />
        </div>
      )}

      {/* Components */}
      {cluster.services.length > 0 && (
        <div className="rounded-xl border border-line bg-raised p-5 sm:p-6">
          <h3 className="mb-3 text-[11px] uppercase tracking-wide text-fg-muted">Components</h3>
          <div className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {cluster.services.map((svc) => {
              const info = SERVICE_INFO[svc.service] ?? {
                label: svc.service,
                desc: "",
              };
              const state = serviceState(svc.state, svc.health);
              const perService = usage?.services.find(
                (u) => u.service.endsWith(`-${svc.service}`) || u.service === svc.service,
              );
              return (
                <div
                  key={svc.service}
                  className="flex min-h-11 items-center gap-3 border-t border-line/60 py-1.5"
                >
                  <span className={`size-2 shrink-0 rounded-full ${state.dot}`} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-fg">{info.label}</p>
                    {info.desc && <p className="truncate text-[11px] text-fg-muted">{info.desc}</p>}
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="text-xs text-fg-secondary">{state.word}</p>
                    {perService && (
                      <p className="text-[11px] tabular-nums text-fg-muted">
                        {perService.cpu_percent.toFixed(1)}% · {formatBytes(perService.mem_used_bytes)}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Connection details + reset */}
      <div className="rounded-xl border border-line bg-raised p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-[11px] uppercase tracking-wide text-fg-muted">Connection</h3>
            <a
              href={cluster.url}
              target="_blank"
              rel="noreferrer"
              className={`mt-1 block truncate font-mono text-xs hover:text-fg ${
                running ? "text-accent-2" : "text-fg-muted"
              }`}
            >
              {cluster.url}
            </a>
            <p className="mt-1 text-[11px] text-fg-muted">
              Point the AgentX desktop app at this address.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {PORT_INFO.map(([key, label]) => (
              <span
                key={key}
                className="rounded-md border border-line bg-sunken px-1.5 py-0.5 text-[11px] text-fg-muted"
              >
                {label} <span className="font-mono">:{cluster.ports[key]}</span>
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="text-[11px] text-fg-muted">
          This dashboard manages the single AgentX deployment in{" "}
          <span className="font-mono">{cluster.dir}</span>. Multi-deployment management is
          available when running the manager from a source checkout.
        </p>
        <button
          disabled={busy !== null}
          onClick={onDestroy}
          className="min-h-9 shrink-0 rounded-lg border border-red-900/60 bg-red-950/40 px-3 text-sm text-red-300 hover:border-red-700 disabled:opacity-40"
        >
          Reset…
        </button>
      </div>
    </div>
  );
}

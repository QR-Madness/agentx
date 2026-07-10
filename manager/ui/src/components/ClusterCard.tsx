/* One deployment: phase, spec tags, live resource gauges, services, actions. */

import { useCallback, useState } from "react";
import { api, type Cluster, type ClusterUsage, type LifecycleAction } from "../api";
import { usePolling } from "../hooks";
import { formatBytes, PhaseBadge, UsageGauge } from "./ui";

function SpecTags({ cluster }: { cluster: Cluster }) {
  const { spec } = cluster;
  const tags: string[] = [spec.kind];
  if (spec.gateway) tags.push("gateway");
  if (spec.tunnel !== "none") tags.push(`tunnel:${spec.tunnel}`);
  if (spec.expose) tags.push("expose");
  if (spec.gpu) tags.push("gpu");
  if (spec.shell) tags.push("shell");
  return (
    <div className="flex flex-wrap gap-1.5">
      {tags.map((tag) => (
        <span
          key={tag}
          className="rounded-md border border-line bg-sunken px-1.5 py-0.5 text-[11px] text-fg-muted"
        >
          {tag}
        </span>
      ))}
    </div>
  );
}

const ACTION_LABEL: Record<LifecycleAction, string> = {
  up: "Up",
  down: "Down",
  restart: "Restart",
  rebuild: "Rebuild",
  adopt: "Adopt",
};

const PORT_LABELS: [key: keyof Cluster["ports"], label: string][] = [
  ["api", "api"],
  ["neo4j_http", "neo4j-http"],
  ["neo4j_bolt", "neo4j-bolt"],
  ["postgres", "postgres"],
  ["redis", "redis"],
];

export function ClusterCard({
  cluster,
  repoVersion,
  busy,
  onAction,
  onDestroy,
  onLogs,
  onShare,
  onEnableGateway,
}: {
  cluster: Cluster;
  repoVersion: string | null; // checkout version (repo mode), for drift hints
  busy: string | null; // running action name, if any
  onAction: (action: LifecycleAction) => void;
  onDestroy: () => void;
  onLogs: () => void;
  onShare: () => void;
  onEnableGateway: () => void;
}) {
  const [usage, setUsage] = useState<ClusterUsage | null>(null);
  const [expanded, setExpanded] = useState(false);
  const running = cluster.phase !== "down";

  const poll = useCallback(async () => {
    try {
      setUsage(await api.usage(cluster.name));
    } catch {
      /* usage is best-effort; the phase badge carries the real signal */
    }
  }, [cluster.name]);
  // Visibility-aware: no usage requests from background tabs.
  usePolling(poll, 5000, running);
  if (!running && usage !== null) setUsage(null);

  const actions: LifecycleAction[] = running
    ? ["down", "restart"]
    : ["up"];
  if (cluster.spec.kind === "source") actions.push("rebuild");

  return (
    <div className="rounded-xl border border-line bg-raised p-5 shadow-md sm:p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2.5">
              <h3 className="text-[15px] font-semibold tracking-tight text-fg">{cluster.name}</h3>
              <PhaseBadge phase={cluster.phase} />
              {cluster.api_version && (
                <span className="rounded-md border border-line bg-sunken px-1.5 py-0.5 font-mono text-[11px] text-fg-muted">
                  api v{cluster.api_version}
                </span>
              )}
              {cluster.api_version && repoVersion && cluster.api_version !== repoVersion && (
                <span
                  title="Running image differs from the checkout — Rebuild, then Up to roll it out"
                  className="rounded-md border border-amber-400/30 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[11px] text-amber-300"
                >
                  checkout v{repoVersion}
                </span>
              )}
              {busy && (
                <span className="flex items-center gap-1.5 text-xs text-accent">
                  <span className="size-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                  {busy}…
                </span>
              )}
            </div>
            <p className="mt-1.5 truncate font-mono text-[11px] text-fg-muted" title={cluster.dir}>
              {cluster.dir}
            </p>
            <a
              href={cluster.url}
              target="_blank"
              rel="noreferrer"
              className={`mt-1 block truncate font-mono text-[11px] hover:text-fg ${
                running ? "text-accent-2" : "text-fg-muted"
              }`}
            >
              {cluster.url}
            </a>
          </div>
          <SpecTags cluster={cluster} />
        </div>

        {usage && (
          <div className="space-y-1.5 border-t border-line/60 pt-4">
            <UsageGauge label="cpu" percent={usage.cpu_percent} />
            <UsageGauge
              label="mem"
              percent={usage.mem_percent}
              detail={
                usage.mem_limit_bytes > 0
                  ? `${formatBytes(usage.mem_used_bytes)} / ${formatBytes(usage.mem_limit_bytes)}`
                  : formatBytes(usage.mem_used_bytes)
              }
            />
            <div className="flex items-center gap-2 text-xs">
              <span className="w-8 shrink-0 text-fg-muted">net</span>
              <span className="flex-1 tabular-nums text-fg-secondary">
                ↓ {formatBytes(usage.net_rx_rate)}/s · ↑ {formatBytes(usage.net_tx_rate)}/s
              </span>
              <span className="shrink-0 text-right text-[11px] tabular-nums text-fg-muted">
                total ↓ {formatBytes(usage.net_rx_bytes)} · ↑ {formatBytes(usage.net_tx_bytes)}
              </span>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-1.5 border-t border-line/60 pt-4">
          <span className="text-[11px] text-fg-muted">ports</span>
          {PORT_LABELS.map(([key, label]) => (
            <span
              key={key}
              className="rounded-md border border-line bg-sunken px-1.5 py-0.5 font-mono text-[11px] text-fg-muted"
            >
              {label}:{cluster.ports[key]}
            </span>
          ))}
        </div>

        {cluster.services.length > 0 && (
          <div className="border-t border-line/60 pt-3">
            <button
              onClick={() => setExpanded((current) => !current)}
              className="text-xs text-fg-muted hover:text-fg-secondary"
            >
              {expanded ? "▾" : "▸"} {cluster.services.length} services
            </button>
            {expanded && (
              <table className="mt-2 w-full text-xs">
                <tbody>
                  {cluster.services.map((svc) => {
                    const perService = usage?.services.find((u) =>
                      u.service.endsWith(`-${svc.service}`) || u.service === svc.service,
                    );
                    return (
                      <tr key={svc.service} className="border-t border-line/60">
                        <td className="py-1 pr-2 font-mono">{svc.service}</td>
                        <td className="py-1 pr-2 text-fg-secondary">
                          {svc.state}
                          {svc.health ? ` (${svc.health})` : ""}
                        </td>
                        <td className="py-1 text-right tabular-nums text-fg-muted">
                          {perService
                            ? `${perService.cpu_percent.toFixed(1)}% · ${formatBytes(perService.mem_used_bytes)} · ↓ ${formatBytes(perService.net_rx_bytes)}`
                            : ""}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2.5 border-t border-line/60 pt-4">
          {actions.map((action) => (
            <button
              key={action}
              disabled={busy !== null}
              onClick={() => onAction(action)}
              className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg disabled:opacity-40"
            >
              {ACTION_LABEL[action]}
            </button>
          ))}
          <button
            onClick={onLogs}
            className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg"
          >
            Logs
          </button>
          <button
            onClick={onShare}
            className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg"
          >
            Share
          </button>
          {!cluster.spec.gateway && (
            <button
              disabled={busy !== null}
              onClick={onEnableGateway}
              className="min-h-9 rounded-lg border border-line bg-overlay px-3 text-sm text-fg-secondary hover:border-line-strong hover:text-fg disabled:opacity-40"
            >
              Enable gateway…
            </button>
          )}
          <button
            disabled={busy !== null}
            onClick={onDestroy}
            className="min-h-9 rounded-lg border border-red-900/60 bg-red-950/40 px-3 text-sm text-red-300 hover:border-red-700 disabled:opacity-40"
          >
            Destroy
          </button>
        </div>
      </div>
    </div>
  );
}

/* One deployment: phase, spec tags, live resource gauges, services, actions. */

import { useEffect, useState } from "react";
import { api, type Cluster, type ClusterUsage, type LifecycleAction } from "../api";
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

export function ClusterCard({
  cluster,
  busy,
  onAction,
  onDestroy,
  onLogs,
}: {
  cluster: Cluster;
  busy: string | null; // running action name, if any
  onAction: (action: LifecycleAction) => void;
  onDestroy: () => void;
  onLogs: () => void;
}) {
  const [usage, setUsage] = useState<ClusterUsage | null>(null);
  const [expanded, setExpanded] = useState(false);
  const running = cluster.phase !== "down";

  useEffect(() => {
    if (!running) {
      setUsage(null);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await api.usage(cluster.name);
        if (!cancelled) setUsage(next);
      } catch {
        /* usage is best-effort; the phase badge carries the real signal */
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [cluster.name, running]);

  const actions: LifecycleAction[] = running
    ? ["down", "restart"]
    : ["up"];
  if (cluster.spec.kind === "source") actions.push("rebuild");

  return (
    <div className="rounded-xl border border-line bg-raised p-4 shadow-md">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2.5">
            <h3 className="text-base font-semibold">{cluster.name}</h3>
            <PhaseBadge phase={cluster.phase} />
            {busy && (
              <span className="flex items-center gap-1.5 text-xs text-accent">
                <span className="size-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                {busy}…
              </span>
            )}
          </div>
          <p className="mt-1 truncate font-mono text-[11px] text-fg-muted" title={cluster.dir}>
            {cluster.dir}
          </p>
        </div>
        <SpecTags cluster={cluster} />
      </div>

      {usage && (
        <div className="mb-3 space-y-1.5">
          <UsageGauge label="cpu" percent={usage.cpu_percent} />
          <UsageGauge
            label="mem"
            percent={usage.mem_percent}
            detail={formatBytes(usage.mem_used_bytes)}
          />
        </div>
      )}

      {cluster.services.length > 0 && (
        <button
          onClick={() => setExpanded((current) => !current)}
          className="mb-2 text-xs text-fg-muted hover:text-fg-secondary"
        >
          {expanded ? "▾" : "▸"} {cluster.services.length} services
        </button>
      )}
      {expanded && (
        <table className="mb-3 w-full text-xs">
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
                      ? `${perService.cpu_percent.toFixed(1)}% · ${formatBytes(perService.mem_used_bytes)}`
                      : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      <div className="flex flex-wrap gap-2">
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
          disabled={busy !== null}
          onClick={onDestroy}
          className="min-h-9 rounded-lg border border-red-900/60 bg-red-950/40 px-3 text-sm text-red-300 hover:border-red-700 disabled:opacity-40"
        >
          Destroy
        </button>
      </div>
    </div>
  );
}

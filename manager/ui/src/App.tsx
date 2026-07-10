/* AgentX Manager dashboard: token gate → cluster grid, polling every 5s. */

import { useCallback, useEffect, useState } from "react";
import {
  api,
  ApiError,
  clearToken,
  getToken,
  setToken,
  waitForJob,
  type Cluster,
  type LifecycleAction,
  type Meta,
} from "./api";
import { BundleDashboard } from "./components/BundleDashboard";
import { ClusterCard } from "./components/ClusterCard";
import { LogsPanel } from "./components/LogsPanel";
import {
  DestroyModal,
  EnableGatewayModal,
  NewClusterModal,
  ShareModal,
} from "./components/modals";
import { ToastProvider, useToast } from "./components/ui";
import { usePolling } from "./hooks";

function TokenGate({ onUnlocked }: { onUnlocked: () => void }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  const submit = async () => {
    setToken(value.trim());
    try {
      await api.meta();
      onUnlocked();
    } catch (err) {
      clearToken();
      setError(err instanceof ApiError && err.status === 401
        ? "That token was rejected."
        : `Can't reach the manager: ${err instanceof Error ? err.message : err}`);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <div className="w-full max-w-sm rounded-xl border border-line bg-raised p-6 shadow-xl">
        <h1 className="mb-1 text-lg font-semibold text-fg">AgentX Manager</h1>
        <p className="mb-4 text-sm text-fg-secondary">
          Paste the access token — it's printed by{" "}
          <span className="font-mono text-xs">agentx-manager serve</span> (or{" "}
          <span className="font-mono text-xs">docker compose logs manager</span>)
          and stored in <span className="font-mono text-xs">.manager-token</span>{" "}
          next to your compose files.
        </p>
        <input
          autoFocus
          type="password"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void submit();
          }}
          placeholder="manager token"
          className="mb-3 w-full rounded-lg border border-line bg-sunken px-3 py-2 font-mono text-sm outline-none focus:border-line-strong"
        />
        {error && <p className="mb-3 text-sm text-red-300">{error}</p>}
        <button
          disabled={!value.trim()}
          onClick={() => void submit()}
          className="min-h-10 w-full rounded-lg border border-accent/60 bg-accent/15 text-sm font-medium disabled:opacity-40"
        >
          Unlock
        </button>
      </div>
    </div>
  );
}

function Dashboard({ onLocked }: { onLocked: () => void }) {
  const { toast } = useToast();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [busy, setBusy] = useState<Record<string, string | null>>({});
  const [logsFor, setLogsFor] = useState<Cluster | null>(null);
  const [destroyFor, setDestroyFor] = useState<Cluster | null>(null);
  const [shareFor, setShareFor] = useState<Cluster | null>(null);
  const [gatewayFor, setGatewayFor] = useState<Cluster | null>(null);
  const [creating, setCreating] = useState(false);

  const bail = useCallback(
    (err: unknown) => {
      if (err instanceof ApiError && err.status === 401) {
        clearToken();
        onLocked();
        return true;
      }
      return false;
    },
    [onLocked],
  );

  const refresh = useCallback(async () => {
    try {
      setClusters(await api.clusters());
    } catch (err) {
      if (!bail(err)) {
        /* transient poll failure — keep the last snapshot */
      }
    }
  }, [bail]);

  useEffect(() => {
    void api.meta().then(setMeta).catch(bail);
  }, [bail]);
  // Poll only while the tab is visible — this page is often left open in a
  // background tab for hours.
  usePolling(refresh, 5000);

  const runAction = async (
    cluster: Cluster,
    action: LifecycleAction | "destroy",
    start: () => Promise<{ job: string }>,
  ) => {
    setBusy((current) => ({ ...current, [cluster.name]: action }));
    try {
      const { job } = await start();
      const finished = await waitForJob(job);
      if (finished.status === "done") {
        toast("ok", `${cluster.name}: ${finished.detail || action}`);
      } else {
        toast("error", `${cluster.name} ${action} failed: ${finished.detail}`);
      }
    } catch (err) {
      if (!bail(err)) {
        toast("error", `${cluster.name} ${action}: ${err instanceof Error ? err.message : err}`);
      }
    } finally {
      setBusy((current) => ({ ...current, [cluster.name]: null }));
      void refresh();
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold text-fg">AgentX Manager</h1>
        {meta && meta.mode === "repo" && (
          <span className="rounded-md border border-line bg-sunken px-2 py-0.5 font-mono text-[11px] text-fg-muted">
            repo · {meta.root}
            {meta.repo_version ? ` · v${meta.repo_version}` : ""}
          </span>
        )}
        <span className="text-[11px] text-fg-muted">
          {meta ? `manager v${meta.version} · ` : ""}live · pauses in background tabs
        </span>
        <div className="ml-auto flex gap-2">
          {meta?.mode === "repo" && (
            <button
              onClick={() => setCreating(true)}
              className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-3 text-sm font-medium"
            >
              + New cluster
            </button>
          )}
          <button
            onClick={() => {
              clearToken();
              onLocked();
            }}
            className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-muted hover:text-fg-secondary"
          >
            Lock
          </button>
        </div>
      </header>

      {clusters.length === 0 ? (
        <p className="text-sm text-fg-muted">
          No deployments found under this root.
          {meta?.mode === "repo" ? " Create one with “New cluster”." : ""}
        </p>
      ) : meta?.mode === "bundle" && clusters.length === 1 ? (
        <BundleDashboard
          cluster={clusters[0]}
          busy={busy[clusters[0].name] ?? null}
          onAction={(action) =>
            void runAction(clusters[0], action, () => api.action(clusters[0].name, action))
          }
          onDestroy={() => setDestroyFor(clusters[0])}
          onLogs={() => setLogsFor(clusters[0])}
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">
          {clusters.map((cluster) => (
            <ClusterCard
              key={cluster.name}
              cluster={cluster}
              repoVersion={meta?.repo_version ?? null}
              busy={busy[cluster.name] ?? null}
              onAction={(action) =>
                void runAction(cluster, action, () => api.action(cluster.name, action))
              }
              onDestroy={() => setDestroyFor(cluster)}
              onLogs={() => setLogsFor(cluster)}
              onShare={() => setShareFor(cluster)}
              onEnableGateway={() => setGatewayFor(cluster)}
            />
          ))}
        </div>
      )}

      {logsFor && (
        <LogsPanel
          cluster={logsFor}
          title={meta?.mode === "bundle" ? "Activity Log" : undefined}
          onClose={() => setLogsFor(null)}
        />
      )}
      {shareFor && <ShareModal cluster={shareFor} onClose={() => setShareFor(null)} />}
      {gatewayFor && (
        <EnableGatewayModal
          name={gatewayFor.name}
          onEnable={(tunnel) => api.enableGateway(gatewayFor.name, tunnel)}
          onClose={() => {
            setGatewayFor(null);
            void refresh(); // spec.gateway may have flipped
          }}
        />
      )}
      {destroyFor && (
        <DestroyModal
          name={destroyFor.name}
          flavor={meta?.mode === "bundle" ? "bundle" : "cluster"}
          onClose={() => setDestroyFor(null)}
          onConfirm={(keepData) => {
            const target = destroyFor;
            setDestroyFor(null);
            void runAction(target, "destroy", () =>
              api.destroy(target.name, target.name, keepData),
            );
          }}
        />
      )}
      {creating && (
        <NewClusterModal
          onClose={() => setCreating(false)}
          onCreate={async (payload) => {
            const result = await api.createCluster(payload);
            void refresh();
            return result;
          }}
        />
      )}
    </div>
  );
}

export default function App() {
  const [unlocked, setUnlocked] = useState(() => getToken() !== null);
  return (
    <ToastProvider>
      {unlocked ? (
        <Dashboard onLocked={() => setUnlocked(false)} />
      ) : (
        <TokenGate onUnlocked={() => setUnlocked(true)} />
      )}
    </ToastProvider>
  );
}

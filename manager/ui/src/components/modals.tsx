/* Destroy (typed confirmation), New Cluster, Share link, and Enable Gateway modals. */

import { useCallback, useEffect, useState } from "react";
import {
  api,
  type Cluster,
  type ConnectionInfo,
  type GatewayResult,
  type Kind,
  type NewClusterPayload,
  type ScaffoldResult,
  type Tunnel,
} from "../api";
import { buildConnectUrl } from "../lib/connect";
import { Modal, useToast } from "./ui";

export function DestroyModal({
  name,
  flavor = "cluster",
  onConfirm,
  onClose,
}: {
  name: string;
  flavor?: "cluster" | "bundle";
  onConfirm: (keepData: boolean) => void;
  onClose: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [keepData, setKeepData] = useState(false);
  const armed = typed === name;

  return (
    <Modal
      title={flavor === "bundle" ? "Reset AgentX" : `Destroy ${name}`}
      onClose={onClose}
    >
      <p className="mb-3 text-sm text-fg-secondary">
        {flavor === "bundle" ? (
          <>
            This stops AgentX
            {keepData
              ? ""
              : " and permanently deletes everything it has stored — databases, agent memory, and downloaded AI models"}
            . It cannot be undone. Type{" "}
            <span className="font-mono text-fg">{name}</span> to confirm.
          </>
        ) : (
          <>
            This stops every container, removes volumes
            {keepData ? "" : ", and deletes the cluster's data directories"}. It
            cannot be undone. Type <span className="font-mono text-fg">{name}</span>{" "}
            to confirm.
          </>
        )}
      </p>
      <input
        autoFocus
        value={typed}
        onChange={(event) => setTyped(event.target.value)}
        placeholder={name}
        className="mb-3 w-full rounded-lg border border-line bg-sunken px-3 py-2 font-mono text-sm outline-none focus:border-line-strong"
      />
      <label className="mb-4 flex min-h-9 items-center gap-2 text-sm text-fg-secondary">
        <input
          type="checkbox"
          checked={keepData}
          onChange={(event) => setKeepData(event.target.checked)}
        />
        {flavor === "bundle"
          ? "Keep stored data (just stop and remove the runtime)"
          : "Keep data directories (containers + volumes only)"}
      </label>
      <div className="flex justify-end gap-2">
        <button
          onClick={onClose}
          className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-secondary hover:text-fg"
        >
          Cancel
        </button>
        <button
          disabled={!armed}
          onClick={() => onConfirm(keepData)}
          className="min-h-9 rounded-lg border border-red-700 bg-red-900/60 px-4 text-sm font-medium text-red-100 disabled:opacity-40"
        >
          {flavor === "bundle" ? "Reset" : "Destroy"}
        </button>
      </div>
    </Modal>
  );
}

export function NewClusterModal({
  onCreate,
  onClose,
}: {
  onCreate: (payload: NewClusterPayload) => Promise<ScaffoldResult>;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [kind, setKind] = useState<Kind>("source");
  const [gateway, setGateway] = useState(false);
  const [tunnel, setTunnel] = useState<Tunnel>("none");
  const [gpu, setGpu] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ScaffoldResult | null>(null);

  const submit = async () => {
    setPending(true);
    setError("");
    try {
      setResult(await onCreate({ name: name.trim(), kind, gateway, tunnel, gpu }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(false);
    }
  };

  if (result) {
    return (
      <Modal title={`Created ${name}`} onClose={onClose}>
        <p className="mb-2 font-mono text-xs text-fg-muted">{result.dir}</p>
        {Object.keys(result.generated).length > 0 && (
          <>
            <p className="mb-1 text-sm text-fg-secondary">
              Generated secrets — shown once (already written to .env):
            </p>
            <pre className="mb-3 select-all overflow-x-auto rounded-lg border border-line bg-sunken p-2 font-mono text-[11px] leading-relaxed">
              {Object.entries(result.generated)
                .map(([key, value]) => `${key}=${value}`)
                .join("\n")}
            </pre>
          </>
        )}
        <ul className="mb-4 list-disc pl-5 text-sm text-fg-secondary">
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="min-h-9 rounded-lg border border-line-strong bg-overlay px-4 text-sm hover:text-fg"
          >
            Done
          </button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal title="New cluster" onClose={onClose}>
      <div className="space-y-3">
        <input
          autoFocus
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="cluster name (e.g. prod)"
          className="w-full rounded-lg border border-line bg-sunken px-3 py-2 font-mono text-sm outline-none focus:border-line-strong"
        />
        <div className="flex gap-2">
          {(["source", "image"] as Kind[]).map((option) => (
            <button
              key={option}
              onClick={() => setKind(option)}
              className={`min-h-9 flex-1 rounded-lg border px-3 text-sm ${
                kind === option
                  ? "border-accent/60 bg-accent/10 text-fg"
                  : "border-line text-fg-muted hover:text-fg-secondary"
              }`}
            >
              {option === "source" ? "source (build from repo)" : "image (published)"}
            </button>
          ))}
        </div>
        <label className="flex min-h-9 items-center gap-2 text-sm text-fg-secondary">
          <input
            type="checkbox"
            checked={gateway}
            onChange={(event) => {
              setGateway(event.target.checked);
              if (!event.target.checked) setTunnel("none");
            }}
          />
          Token gateway (shared secret + rate limiting)
        </label>
        {gateway && (
          <select
            value={tunnel}
            onChange={(event) => setTunnel(event.target.value as Tunnel)}
            className="w-full rounded-lg border border-line bg-sunken px-3 py-2 text-sm"
          >
            <option value="none">no tunnel (expose or private)</option>
            <option value="token">Cloudflare token tunnel (dashboard)</option>
            <option value="named">Cloudflare named tunnel (credentials file)</option>
          </select>
        )}
        <label className="flex min-h-9 items-center gap-2 text-sm text-fg-secondary">
          <input
            type="checkbox"
            checked={gpu}
            onChange={(event) => setGpu(event.target.checked)}
          />
          NVIDIA GPU overlay
        </label>
        {error && <p className="text-sm text-red-300">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-secondary hover:text-fg"
          >
            Cancel
          </button>
          <button
            disabled={pending || !name.trim()}
            onClick={() => void submit()}
            className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-4 text-sm font-medium text-fg disabled:opacity-40"
          >
            {pending ? "Creating…" : "Create"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Share (connection link)

const APP_BASE_KEY = "agentx-manager:app-base";
const LOCAL_URL_RE = /^https?:\/\/(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|\[::1\])(:|\/|$)/i;

function isHttpUrl(raw: string): boolean {
  try {
    const parsed = new URL(raw.trim());
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function originOf(raw: string): string | null {
  try {
    return new URL(raw.trim()).origin;
  } catch {
    return null;
  }
}

function Notice({ tone, children }: { tone: "amber" | "red"; children: React.ReactNode }) {
  const style =
    tone === "red"
      ? "border-red-400/40 bg-red-950/40 text-red-300"
      : "border-amber-400/30 bg-amber-500/10 text-amber-300";
  return <p className={`rounded-lg border px-3 py-2 text-xs leading-relaxed ${style}`}>{children}</p>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-fg-muted">{label}</span>
      {children}
    </label>
  );
}

const FIELD_CLASS =
  "w-full rounded-lg border border-line bg-sunken px-3 py-2 font-mono text-sm outline-none focus:border-line-strong";

export function ShareModal({ cluster, onClose }: { cluster: Cluster; onClose: () => void }) {
  const { toast } = useToast();
  const [info, setInfo] = useState<ConnectionInfo | null>(null);
  const [loadError, setLoadError] = useState("");
  const [url, setUrl] = useState("");
  const [name, setName] = useState(cluster.name);
  const [appBase, setAppBase] = useState(() => localStorage.getItem(APP_BASE_KEY) ?? "");

  const load = useCallback(async () => {
    setLoadError("");
    try {
      const data = await api.connection(cluster.name);
      setInfo(data);
      setUrl(data.url);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
  }, [cluster.name]);
  useEffect(() => {
    void load();
  }, [load]);

  const serverOk = isHttpUrl(url);
  const appOrigin = originOf(appBase);
  const link =
    info && serverOk && appOrigin
      ? buildConnectUrl(
          {
            url: url.trim(),
            gatewayToken: info.gateway_token || undefined,
            name: name.trim() || undefined,
          },
          appBase.trim(),
        )
      : "";

  // The public host's own origin is whitelisted automatically by the API
  // (AGENTX_PUBLIC_HOST), so only warn about a *different* app origin that the
  // cluster's CORS list doesn't cover.
  const corsMiss =
    info !== null &&
    appOrigin !== null &&
    serverOk &&
    !LOCAL_URL_RE.test(url.trim()) &&
    appOrigin !== originOf(url) &&
    !info.cors_origins.some((origin) => origin.replace(/\/+$/, "") === appOrigin);

  const copy = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast("ok", `${label} copied`);
    } catch {
      toast("error", "Clipboard unavailable here — select the text and copy it manually");
    }
  };

  return (
    <Modal title={`Share ${cluster.name}`} onClose={onClose}>
      {!info && !loadError && <p className="text-sm text-fg-muted">Loading connection info…</p>}
      {loadError && (
        <div className="space-y-3">
          <p className="text-sm text-red-300">{loadError}</p>
          <button
            onClick={() => void load()}
            className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-secondary hover:text-fg"
          >
            Retry
          </button>
        </div>
      )}
      {info && (
        <div className="space-y-3">
          <Field label="Server URL (what the recipient's app connects to)">
            <input value={url} onChange={(e) => setUrl(e.target.value)} className={FIELD_CLASS} />
          </Field>
          <Field label="Display name (shown in their server list)">
            <input value={name} onChange={(e) => setName(e.target.value)} className={FIELD_CLASS} />
          </Field>
          <Field label="Web app base (where the deployed client lives)">
            <input
              value={appBase}
              placeholder="https://app.example.com"
              onChange={(e) => {
                setAppBase(e.target.value);
                localStorage.setItem(APP_BASE_KEY, e.target.value);
              }}
              className={FIELD_CLASS}
            />
          </Field>

          {LOCAL_URL_RE.test(url.trim()) && (
            <Notice tone="amber">
              This URL is only reachable on the manager's machine. For remote access, share the
              tunnel hostname instead (set <span className="font-mono">AGENTX_PUBLIC_HOST</span> in
              the cluster .env, or type the hostname above).
            </Notice>
          )}
          {info.gateway_enabled && !info.gateway_token && (
            <Notice tone="amber">
              The gateway is enabled but <span className="font-mono">AGENTX_GATEWAY_TOKEN</span> is
              empty in .env — the gateway won't start and this link carries no access token.
            </Notice>
          )}
          {!info.gateway_enabled && (
            <Notice tone="amber">
              No token gateway — the link carries no access token. Fine on a private LAN; before
              exposing this cluster publicly, use <em>Enable gateway…</em> on its card.
            </Notice>
          )}
          {corsMiss && (
            <Notice tone="amber">
              <span className="font-mono">{appOrigin}</span> isn't in this cluster's{" "}
              <span className="font-mono">CORS_ALLOWED_ORIGINS</span> — browsers will block the web
              app's API calls. Add it to the cluster .env, then run <em>Up</em> (Restart doesn't
              re-read env).
            </Notice>
          )}
          {!info.auth_enabled && (
            <Notice tone="red">
              Authentication is OFF (<span className="font-mono">AGENTX_AUTH_ENABLED=false</span>)
              — anyone who gets this link has full access to the agent.
            </Notice>
          )}

          <Field label="Connection link">
            <input
              readOnly
              value={link}
              placeholder={
                appOrigin ? "fix the server URL above" : "enter the web app base above"
              }
              onFocus={(e) => e.currentTarget.select()}
              className={FIELD_CLASS}
            />
          </Field>
          <p className="text-xs leading-relaxed text-fg-muted">
            The link embeds the server address{info.gateway_token ? " and the gateway token" : ""} —
            treat it like a secret and share it privately. The recipient's password is never
            included; they sign in after connecting.
          </p>
          <div className="flex justify-end gap-2">
            {info.gateway_token && (
              <button
                onClick={() => void copy(info.gateway_token, "Gateway token")}
                className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-secondary hover:text-fg"
              >
                Copy token
              </button>
            )}
            <button
              disabled={!link}
              onClick={() => void copy(link, "Connection link")}
              className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-4 text-sm font-medium text-fg disabled:opacity-40"
            >
              Copy link
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Enable gateway (surfaces POST /api/clusters/{name}/gateway)

export function EnableGatewayModal({
  name,
  onEnable,
  onClose,
}: {
  name: string;
  onEnable: (tunnel: Tunnel) => Promise<GatewayResult>;
  onClose: () => void;
}) {
  const [tunnel, setTunnel] = useState<Tunnel>("none");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<GatewayResult | null>(null);

  const submit = async () => {
    setPending(true);
    setError("");
    try {
      setResult(await onEnable(tunnel));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(false);
    }
  };

  if (result) {
    return (
      <Modal title={`Gateway enabled for ${name}`} onClose={onClose}>
        {Object.keys(result.generated).length > 0 && (
          <>
            <p className="mb-1 text-sm text-fg-secondary">
              Generated secrets — shown once (already written to .env):
            </p>
            <pre className="mb-3 select-all overflow-x-auto rounded-lg border border-line bg-sunken p-2 font-mono text-[11px] leading-relaxed">
              {Object.entries(result.generated)
                .map(([key, value]) => `${key}=${value}`)
                .join("\n")}
            </pre>
          </>
        )}
        {result.notes.length > 0 && (
          <ul className="mb-3 list-disc pl-5 text-sm text-fg-secondary">
            {result.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        )}
        <p className="mb-4 text-sm text-fg-secondary">
          Run <em>Up</em> to apply — Restart won't create the gateway service or re-read .env.
          Public traffic additionally needs an exposure overlay (tunnel or expose) — see the{" "}
          <em>Going public</em> docs.
        </p>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="min-h-9 rounded-lg border border-line-strong bg-overlay px-4 text-sm hover:text-fg"
          >
            Done
          </button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal title={`Enable gateway for ${name}`} onClose={onClose}>
      <div className="space-y-3">
        <p className="text-sm text-fg-secondary">
          Puts an Nginx token gateway in front of the API: every request must carry a shared-secret
          header, with per-IP rate limiting. Generates{" "}
          <span className="font-mono text-xs">AGENTX_GATEWAY_TOKEN</span> into .env — share links
          then carry that token for you.
        </p>
        <select
          value={tunnel}
          onChange={(event) => setTunnel(event.target.value as Tunnel)}
          className="w-full rounded-lg border border-line bg-sunken px-3 py-2 text-sm"
        >
          <option value="none">no tunnel (expose or private)</option>
          <option value="token">Cloudflare token tunnel (dashboard)</option>
          <option value="named">Cloudflare named tunnel (credentials file)</option>
        </select>
        {error && <p className="text-sm text-red-300">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="min-h-9 rounded-lg border border-line px-3 text-sm text-fg-secondary hover:text-fg"
          >
            Cancel
          </button>
          <button
            disabled={pending}
            onClick={() => void submit()}
            className="min-h-9 rounded-lg border border-accent/60 bg-accent/15 px-4 text-sm font-medium text-fg disabled:opacity-40"
          >
            {pending ? "Enabling…" : "Enable"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

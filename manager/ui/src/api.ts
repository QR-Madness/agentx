/* Typed fetch wrapper for the manager REST API (agentx_manager/server.py).
 *
 * Every /api request carries the bearer token from localStorage. A 401 is
 * surfaced as ApiError(status=401) so the app can drop the token and show
 * the token gate. The log stream is SSE-over-fetch (EventSource can't send
 * headers), parsed manually from `data: <json>\n\n` frames.
 */

const TOKEN_KEY = "agentx-manager:token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ---------------------------------------------------------------------------
// Response shapes (mirroring server.py / spec.py / health.py)

export type Mode = "repo" | "bundle";
export type Kind = "source" | "image";
export type Tunnel = "none" | "token" | "named";
export type Phase = "down" | "initializing" | "degraded" | "up";
export type JobStatus = "running" | "done" | "failed";

export interface Meta {
  version: string;
  root: string;
  mode: Mode;
}

export interface ClusterSpec {
  name: string;
  kind: Kind;
  gateway: boolean;
  tunnel: Tunnel;
  expose: boolean;
  gpu: boolean;
  shell: boolean;
}

export interface ServiceStatus {
  service: string;
  state: string;
  health: string;
}

export interface Cluster {
  name: string;
  spec: ClusterSpec;
  phase: Phase;
  services: ServiceStatus[];
  dir: string;
}

export interface ServiceUsage {
  service: string;
  cpu_percent: number;
  mem_used_bytes: number;
  mem_limit_bytes: number;
  mem_percent: number;
}

export interface ClusterUsage {
  name: string;
  cpu_percent: number;
  mem_used_bytes: number;
  mem_limit_bytes: number;
  mem_percent: number;
  services: ServiceUsage[];
}

export interface Job {
  id: string;
  cluster: string;
  action: string;
  status: JobStatus;
  detail: string;
}

export type LifecycleAction = "up" | "down" | "restart" | "rebuild" | "adopt";

export interface NewClusterPayload {
  name: string;
  kind: Kind;
  gateway: boolean;
  tunnel: Tunnel;
  gpu: boolean;
}

export interface ScaffoldResult {
  dir: string;
  generated: Record<string, string>;
  notes: string[];
}

// ---------------------------------------------------------------------------
// Core request helper

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body !== undefined) headers.set("Content-Type", "application/json");

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch (err) {
    throw new ApiError(0, err instanceof Error ? err.message : "network error");
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body: unknown = await res.json();
      if (
        body &&
        typeof body === "object" &&
        typeof (body as { detail?: unknown }).detail === "string"
      ) {
        message = (body as { detail: string }).detail;
      }
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Endpoints

export const api = {
  meta: (): Promise<Meta> => request("/api/meta"),

  clusters: (): Promise<Cluster[]> => request("/api/clusters"),

  usage: (name: string): Promise<ClusterUsage> =>
    request(`/api/clusters/${encodeURIComponent(name)}/usage`),

  action: (name: string, action: LifecycleAction): Promise<{ job: string }> =>
    request(`/api/clusters/${encodeURIComponent(name)}/${action}`, {
      method: "POST",
      body: JSON.stringify({}),
    }),

  destroy: (
    name: string,
    confirm: string,
    keepData: boolean,
  ): Promise<{ job: string }> =>
    request(`/api/clusters/${encodeURIComponent(name)}/destroy`, {
      method: "POST",
      body: JSON.stringify({ confirm, keep_data: keepData }),
    }),

  createCluster: (payload: NewClusterPayload): Promise<ScaffoldResult> =>
    request("/api/clusters", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  job: (id: string): Promise<Job> =>
    request(`/api/jobs/${encodeURIComponent(id)}`),
};

/** Poll a job every 2s until it leaves the running state. */
export async function waitForJob(id: string): Promise<Job> {
  for (;;) {
    const job = await api.job(id);
    if (job.status !== "running") return job;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

// ---------------------------------------------------------------------------
// SSE-over-fetch log stream

export interface LogStreamOptions {
  service?: string;
  tail?: number;
  signal: AbortSignal;
  onLine: (line: string) => void;
}

/**
 * Stream `docker compose logs -f` for a cluster. Resolves when the server
 * closes the stream; rejects with ApiError on a bad status. Abort the signal
 * to terminate (an abort is swallowed, not thrown).
 */
export async function streamLogs(
  name: string,
  opts: LogStreamOptions,
): Promise<void> {
  const params = new URLSearchParams();
  if (opts.service) params.set("service", opts.service);
  params.set("tail", String(opts.tail ?? 200));

  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(
      `/api/clusters/${encodeURIComponent(name)}/logs?${params.toString()}`,
      { headers, signal: opts.signal },
    );
  } catch (err) {
    if (opts.signal.aborted) return;
    throw new ApiError(0, err instanceof Error ? err.message : "network error");
  }
  if (!res.ok) throw new ApiError(res.status, `log stream failed (${res.status})`);
  if (!res.body) throw new ApiError(0, "log stream has no body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const raw of frame.split("\n")) {
          if (!raw.startsWith("data: ")) continue;
          const payload = raw.slice(6);
          try {
            opts.onLine(String(JSON.parse(payload)));
          } catch {
            opts.onLine(payload);
          }
        }
      }
    }
  } catch (err) {
    if (!opts.signal.aborted) throw err;
  }
}

/* Slide-over live log stream (SSE-over-fetch), service filter, pause. */

import { useEffect, useRef, useState } from "react";
import { streamLogs, type Cluster } from "../api";

export function LogsPanel({
  cluster,
  title,
  onClose,
}: {
  cluster: Cluster;
  title?: string;
  onClose: () => void;
}) {
  const [service, setService] = useState("");
  const [lines, setLines] = useState<string[]>([]);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const scrollRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    setLines([]);
    const controller = new AbortController();
    void streamLogs(cluster.name, {
      service: service || undefined,
      tail: 200,
      signal: controller.signal,
      onLine: (line) => {
        if (pausedRef.current) return;
        setLines((current) => {
          const next = [...current, line];
          return next.length > 2000 ? next.slice(-2000) : next;
        });
      },
    }).catch(() => {
      /* stream errors surface as an abrupt end; the panel stays usable */
    });
    return () => controller.abort();
  }, [cluster.name, service]);

  useEffect(() => {
    if (!paused && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, paused]);

  return (
    <div className="fixed inset-y-0 right-0 z-30 flex w-full max-w-2xl flex-col border-l border-line bg-overlay shadow-2xl">
      <div className="flex items-center gap-3 border-b border-line px-4 py-3">
        <h2 className="text-sm font-semibold">
          {title ?? (
            <>
              Logs — <span className="font-mono">{cluster.name}</span>
            </>
          )}
        </h2>
        <select
          value={service}
          onChange={(event) => setService(event.target.value)}
          className="rounded-md border border-line bg-sunken px-2 py-1 text-xs"
        >
          <option value="">all services</option>
          {cluster.services.map((svc) => (
            <option key={svc.service} value={svc.service}>
              {svc.service}
            </option>
          ))}
        </select>
        <button
          onClick={() => setPaused((current) => !current)}
          className={`min-h-8 rounded-md border px-2.5 text-xs ${
            paused
              ? "border-amber-400/40 text-amber-300"
              : "border-line text-fg-muted hover:text-fg-secondary"
          }`}
        >
          {paused ? "Resume" : "Pause"}
        </button>
        <button
          onClick={onClose}
          className="ml-auto min-h-8 rounded-md px-2.5 text-fg-muted hover:bg-hover hover:text-fg"
          aria-label="Close logs"
        >
          ✕
        </button>
      </div>
      <pre
        ref={scrollRef}
        className="flex-1 overflow-auto whitespace-pre-wrap break-all px-4 py-3 font-mono text-[11px] leading-relaxed text-fg-secondary"
      >
        {lines.length ? lines.join("\n") : "waiting for log lines…"}
      </pre>
    </div>
  );
}

/* Destroy (typed confirmation) and New Cluster modals. */

import { useState } from "react";
import type { Kind, NewClusterPayload, ScaffoldResult, Tunnel } from "../api";
import { Modal } from "./ui";

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

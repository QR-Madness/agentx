/**
 * ConnectGate — confirmation shown when the app is opened via a `#connect=…`
 * share link (see lib/connectionString.ts). Rather than silently trusting a link
 * (a phishing vector), we ask the user to confirm the host before adding it.
 *
 * On confirm we add/select the server and hard-reload so AuthContext re-probes
 * the new host cleanly and lands the user on its password screen — they only
 * ever enter a password, never the URL or gateway token.
 */

import { useState } from 'react';
import { ShieldCheck, X, PlugZap } from 'lucide-react';
import { Button } from './ui/Button';
import {
  getPendingConnect,
  clearPendingConnect,
  type ConnectionPayload,
} from '../lib/connectionString';
import { getServers, addServer, updateServer, setActiveServerId } from '../lib/storage';

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export function ConnectGate() {
  const [payload, setPayload] = useState<ConnectionPayload | null>(() => getPendingConnect());

  if (!payload) return null;

  const host = hostOf(payload.url);
  const displayName = payload.name?.trim() || host;

  const handleConnect = () => {
    const normalized = payload.url.replace(/\/+$/, '');
    const existing = getServers().find((s) => s.url.replace(/\/+$/, '') === normalized);

    let serverId: string;
    if (existing) {
      // Refresh the gateway token if the link carries one (it may have rotated).
      if (payload.gatewayToken && payload.gatewayToken !== existing.gatewayToken) {
        updateServer(existing.id, { gatewayToken: payload.gatewayToken });
      }
      serverId = existing.id;
    } else {
      serverId = addServer(displayName, payload.url, payload.gatewayToken).id;
    }

    setActiveServerId(serverId);
    clearPendingConnect();
    // Fresh boot re-probes the now-active server and shows its Connect/password screen.
    window.location.reload();
  };

  const handleCancel = () => {
    clearPendingConnect();
    setPayload(null);
  };

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="connect-gate-title"
    >
      <div className="w-full max-w-md rounded-2xl border border-line bg-surface-raised p-6 shadow-[var(--shadow-md)]">
        <div className="mb-4 flex items-center gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/15 text-accent">
            <ShieldCheck size={20} />
          </span>
          <div>
            <h2 id="connect-gate-title" className="text-base font-semibold text-fg">
              Connect to a shared server?
            </h2>
            <p className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
              From a connection link
            </p>
          </div>
        </div>

        <p className="mb-3 text-sm text-fg-secondary">
          This link wants to add the following AgentX server. You'll sign in with a password after
          connecting.
        </p>

        <dl className="mb-4 space-y-2 rounded-xl border border-line-subtle bg-surface-sunken p-3 text-sm">
          <div className="flex items-baseline justify-between gap-3">
            <dt className="text-fg-muted">Name</dt>
            <dd className="truncate font-medium text-fg">{displayName}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-3">
            <dt className="text-fg-muted">Address</dt>
            <dd className="truncate font-mono text-2xs text-fg-secondary">{payload.url}</dd>
          </div>
          {payload.gatewayToken && (
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-fg-muted">Gateway</dt>
              <dd className="text-2xs text-fg-secondary">Access token included</dd>
            </div>
          )}
        </dl>

        <p className="mb-5 text-2xs text-fg-muted">
          Only connect to servers you trust — a connection link grants access to whoever holds it.
        </p>

        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={handleCancel}>
            <X size={15} /> Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={handleConnect}>
            <PlugZap size={15} /> Connect
          </Button>
        </div>
      </div>
    </div>
  );
}

/**
 * ProviderConnectionCard — one backend, and whether it actually works.
 *
 * Replaces the old static tile (icon + name + a password box, identical whether
 * a provider was linked or empty). The card leads with **state** — reachable,
 * how many models, key on file — and keeps the credential fields behind an
 * expand, because reading the page is the common case and editing a key is not.
 *
 * The key field never displays a secret. Server-side state arrives as a
 * fingerprint (`····3f21`) from the catalog; typing into the field is an
 * explicit *replacement*, which is why the input starts empty with the
 * fingerprint shown beside it rather than pre-filled with dots.
 */

import { useState, type ReactNode } from 'react';
import { Check, ChevronDown, Eye, EyeOff, Trash2, Wifi, X } from 'lucide-react';
import type { ProviderCatalogEntry, ProviderHealthEntry, ProviderTestResult } from '../../../lib/api';
import { Badge, Button, Card, IconButton, Input, StatusDot } from '../../ui';
import type { BadgeProps, StatusDotProps } from '../../ui';

export interface ProviderPresentation {
  /** Marketing-free one-liner: what this backend is good for. */
  description: string;
  icon: ReactNode;
  /** Icon-tile variant (chip gradient only). */
  tile: 'local' | 'cloud' | 'experimental';
  badge?: { label: string; variant: BadgeProps['variant'] };
  note?: string;
  placeholder: string;
}

type ConnectionState = 'connected' | 'unconfigured' | 'error' | 'checking' | 'blocked';

const STATE_TONE: Record<ConnectionState, StatusDotProps['tone']> = {
  connected: 'online',
  unconfigured: 'inactive',
  error: 'error',
  checking: 'warning',
  blocked: 'inactive',
};

function connectionState(
  entry: ProviderCatalogEntry,
  health: ProviderHealthEntry | undefined,
  healthLoading: boolean,
  blocked: boolean
): ConnectionState {
  if (blocked) return 'blocked';
  if (!entry.configured) return 'unconfigured';
  if (healthLoading && !health) return 'checking';
  if (!health) return 'checking';
  return health.status === 'healthy' ? 'connected' : 'error';
}

/** One line of plain status. Says what happened, never apologizes. */
function stateSummary(
  state: ConnectionState,
  entry: ProviderCatalogEntry,
  health: ProviderHealthEntry | undefined
): string {
  switch (state) {
    case 'blocked':
      return 'Local connection only — not reachable from this cluster.';
    case 'unconfigured':
      return entry.credential === 'base_url'
        ? 'Not connected. Add a server URL to use local models.'
        : 'Not connected. Add a key to use these models.';
    case 'checking':
      return 'Checking…';
    case 'error':
      return health?.error ? `Not responding — ${health.error}` : 'Not responding.';
    case 'connected': {
      const count = health?.models_available;
      return count ? `${count.toLocaleString()} models available` : 'Connected';
    }
  }
}

export interface ProviderConnectionCardProps {
  entry: ProviderCatalogEntry;
  presentation: ProviderPresentation;
  health: ProviderHealthEntry | undefined;
  healthLoading: boolean;
  /** LM Studio on a remote cluster — configurable, but unreachable from there. */
  blocked?: boolean;
  /** Draft credential value, owned by the parent (secrets save explicitly). */
  draft: string;
  onDraftChange: (value: string) => void;
  onTest?: () => Promise<ProviderTestResult | null>;
  onDelete?: () => void;
  /** Rendered under the fields — the OpenRouter link controls land here later. */
  children?: ReactNode;
}

export function ProviderConnectionCard({
  entry,
  presentation,
  health,
  healthLoading,
  blocked = false,
  draft,
  onDraftChange,
  onTest,
  onDelete,
  children,
}: ProviderConnectionCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ProviderTestResult | null>(null);

  const state = connectionState(entry, health, healthLoading, blocked);
  const isUrlCredential = entry.credential === 'base_url';
  const storedValue = isUrlCredential ? entry.base_url : entry.key_fingerprint;

  const handleTest = async () => {
    if (!onTest) return;
    setTesting(true);
    try {
      setTestResult(await onTest());
    } finally {
      setTesting(false);
    }
  };

  return (
    <Card className={`connection-card${expanded ? ' is-expanded' : ''}`}>
      <button
        type="button"
        className="connection-summary"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
        // Explicit label: the provider name also appears in the Supply Line, so
        // the visible text alone doesn't uniquely name this control.
        aria-label={`${entry.label} connection settings`}
      >
        <span className={`provider-icon ${presentation.tile}`} aria-hidden>
          {presentation.icon}
        </span>

        <span className="connection-identity">
          <span className="connection-name">
            {entry.label}
            {presentation.badge && (
              <Badge variant={presentation.badge.variant} size="sm">
                {presentation.badge.label}
              </Badge>
            )}
            {!entry.builtin && (
              <Badge variant="neutral" size="sm">
                Custom
              </Badge>
            )}
          </span>
          <span className="connection-state">
            <StatusDot tone={STATE_TONE[state]} pulse={state === 'checking'} />
            {stateSummary(state, entry, health)}
          </span>
        </span>

        <ChevronDown size={16} className="connection-chevron" aria-hidden />
      </button>

      {expanded && (
        <div className="connection-body">
          <p className="connection-description">{presentation.description}</p>
          {blocked ? (
            <p className="provider-note warning">
              This server reaches AgentX through a cluster gateway, so it can&rsquo;t open a
              connection back to your local machine.
            </p>
          ) : (
            presentation.note && <p className="provider-note">{presentation.note}</p>
          )}

          {/* A registered endpoint's address is the thing that defines it, so
              state it plainly. Built-in cloud providers hardcode theirs. */}
          {!entry.builtin && entry.base_url && !isUrlCredential && (
            <p className="connection-endpoint">
              <span className="connection-field-label">Endpoint</span>
              <code>{entry.base_url}</code>
            </p>
          )}

          <label className="connection-field-label" htmlFor={`provider-cred-${entry.id}`}>
            {isUrlCredential ? 'Server URL' : 'API key'}
            {storedValue && (
              <span className="connection-stored" title="Stored on the server">
                {isUrlCredential ? storedValue : `on file ${storedValue}`}
              </span>
            )}
          </label>
          <div className="api-key-input">
            <Input
              id={`provider-cred-${entry.id}`}
              type={isUrlCredential || revealed ? 'text' : 'password'}
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              placeholder={
                storedValue && !isUrlCredential
                  ? 'Enter a new key to replace the stored one'
                  : presentation.placeholder
              }
              autoComplete="off"
              spellCheck={false}
              disabled={blocked}
            />
            {!isUrlCredential && (
              <IconButton
                className="visibility-toggle"
                onClick={() => setRevealed((shown) => !shown)}
                aria-label={revealed ? 'Hide value' : 'Show value'}
                disabled={blocked}
              >
                {revealed ? <EyeOff size={16} /> : <Eye size={16} />}
              </IconButton>
            )}
          </div>

          {children}

          {(onTest || onDelete) && (
            <div className="connection-actions">
              {onTest && (
                <Button variant="secondary" onClick={handleTest} loading={testing} disabled={blocked}>
                  <Wifi size={16} />
                  {testing ? 'Testing…' : 'Test connection'}
                </Button>
              )}
              {onDelete && (
                <Button variant="ghost" onClick={onDelete} className="connection-remove">
                  <Trash2 size={16} />
                  Remove
                </Button>
              )}
            </div>
          )}

          {testResult && (
            <p className={`connection-test ${testResult.reachable ? 'ok' : 'bad'}`}>
              {testResult.reachable ? (
                <Check size={14} aria-hidden />
              ) : (
                <X size={14} aria-hidden />
              )}
              {testResult.reachable
                ? `Reached ${entry.label} in ${testResult.elapsed_ms} ms — ${testResult.models_available.toLocaleString()} models.`
                : `Couldn't reach ${testResult.base_url} — ${testResult.error ?? 'no response'}.`}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

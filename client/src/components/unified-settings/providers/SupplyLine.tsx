/**
 * SupplyLine — where a turn actually goes, right now.
 *
 * The rail under the Providers header renders the live route for the active
 * agent's model: which provider resolves it, whether that provider is reachable,
 * the model's real context window and price, and the fallback waiting behind it.
 *
 * It exists because this section's failure mode is **silent**. When a model is
 * unavailable the turn is quietly demoted to another one, and when a provider
 * doesn't list a model id its capabilities fall back to a conservative default
 * that looks like a real number. Both surface much later as premature compaction
 * and "spotty memory". Every other integrations page shows a grid of logos; this
 * one shows where the tokens go.
 *
 * Everything here is server-resolved (`GET /api/providers/route` mirrors the
 * runtime fallback chain) — the client never guesses at resolution.
 */

import { useEffect, useState } from 'react';
import { ArrowRight, CornerDownRight, TriangleAlert } from 'lucide-react';
import { api } from '../../../lib/api';
import type { ProviderRouteResponse, RouteCandidate } from '../../../lib/api';
import { StatusDot } from '../../ui';

/** `1M` / `200k` / `8192` — context windows are read at a glance, not audited.
 *  A trailing `.0` is dropped so the common 1048576 reads as "1M", not "1.0M". */
function formatContext(tokens: number | null): string | null {
  if (!tokens) return null;
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
  }
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}k`;
  return String(tokens);
}

/** Per-1k costs are unreadably small; quote the per-million price everyone uses. */
function formatPrice(input: number | null, output: number | null): string | null {
  if (input == null && output == null) return null;
  const perMillion = (perThousand: number | null) =>
    perThousand == null ? '—' : `$${(perThousand * 1000).toFixed(2)}`;
  if (input === 0 && output === 0) return 'free';
  return `${perMillion(input)}/${perMillion(output)} per 1M`;
}

function candidateFacts(candidate: RouteCandidate): string[] {
  const facts: string[] = [];
  const context = formatContext(candidate.context_window);
  if (context) facts.push(`${context} context`);
  const price = formatPrice(candidate.cost_per_1k_input, candidate.cost_per_1k_output);
  if (price) facts.push(price);
  return facts;
}

export interface SupplyLineProps {
  /** The active agent's own model, when the profile pins one. */
  model: string | null;
  /** `preferences.default_model` — what an unpinned profile actually runs on. */
  fallbackModel: string | null;
  /** Agent display name, so the rail reads as *this agent's* supply. */
  agentName: string;
  /** Bumped by the parent to force a re-resolve (e.g. after saving a key). */
  refreshToken?: number;
}

export function SupplyLine({
  model: profileModel,
  fallbackModel,
  agentName,
  refreshToken = 0,
}: SupplyLineProps) {
  // A profile with no model set follows the global default — the single most
  // confusing resolution in the product, because nothing anywhere says so. The
  // rail resolves the same chain the server does and names which link applied.
  const model = profileModel || fallbackModel;
  const inheritsDefault = !profileModel && !!fallbackModel;
  const [route, setRoute] = useState<ProviderRouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!model) {
      setRoute(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    api
      .getProviderRoute(model)
      .then((result) => {
        if (!cancelled) setRoute(result);
      })
      .catch(() => {
        // A rail that can't resolve says nothing rather than guessing — it is
        // the one element on this page that must never state a wrong number.
        if (!cancelled) setFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model, refreshToken]);

  if (!model || failed) return null;

  const resolved = route?.resolved ?? null;
  const backup = route?.candidates.find(
    (candidate) => candidate.model !== resolved?.model && candidate.configured
  );

  return (
    <section className="supply-line" aria-label="Current model route">
      <div className="supply-line-eyebrow">Supply line</div>

      {loading && !route ? (
        <div className="supply-line-row supply-line-loading">Resolving route…</div>
      ) : !resolved ? (
        <div className="supply-line-row">
          <TriangleAlert size={14} className="text-warning shrink-0" aria-hidden />
          <span className="supply-line-text">
            Nothing configured can run <code className="supply-line-model">{model}</code>. Connect a
            provider below.
          </span>
        </div>
      ) : (
        <>
          <div className="supply-line-row">
            <span className="supply-line-node">
              <StatusDot tone={resolved.healthy ? 'online' : 'warning'} />
              {agentName}
              {inheritsDefault && <span className="supply-line-sub">default model</span>}
            </span>

            <ArrowRight size={14} className="supply-line-arrow" aria-hidden />

            <span className="supply-line-node">
              <span className="supply-line-provider">{resolved.provider_label}</span>
              <span className="supply-line-sub">
                {resolved.healthy ? 'reachable' : 'not responding'}
              </span>
            </span>

            <ArrowRight size={14} className="supply-line-arrow" aria-hidden />

            <span className="supply-line-node supply-line-node-model">
              <code className="supply-line-model">{resolved.model_id}</code>
              {candidateFacts(resolved).length > 0 && (
                <span className="supply-line-sub">{candidateFacts(resolved).join(' · ')}</span>
              )}
            </span>
          </div>

          {route?.substituted && (
            <p className="supply-line-note warning">
              <TriangleAlert size={13} aria-hidden />
              Your agent asks for <code>{route.requested}</code>, but that isn&rsquo;t available —
              turns run on the model above instead.
            </p>
          )}

          {resolved.known === false && (
            <p className="supply-line-note warning">
              <TriangleAlert size={13} aria-hidden />
              {resolved.provider_label} doesn&rsquo;t list this model, so its context window is
              unknown. Turns may be compacted far earlier than expected.
            </p>
          )}

          {backup && (
            <p className="supply-line-note">
              <CornerDownRight size={13} aria-hidden />
              If {resolved.provider_label} drops out:{' '}
              <code>{backup.model_id}</code> on {backup.provider_label}
            </p>
          )}
        </>
      )}
    </section>
  );
}

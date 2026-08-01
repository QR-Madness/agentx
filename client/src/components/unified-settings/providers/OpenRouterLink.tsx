/**
 * OpenRouterLink — connect an OpenRouter account in one click.
 *
 * OpenRouter mints a **user-controlled** key over OAuth PKCE, which removes the
 * whole "leave the app, make an account, find the keys page, paste it back"
 * detour that is otherwise the first thing a new user hits.
 *
 * The flow is server-driven: `POST …/oauth/start` returns a consent URL, we open
 * it in the user's *real* browser (a bare `window.open` no-ops inside the Tauri
 * webview), OpenRouter redirects to the API's callback, and the API exchanges
 * the code. The verifier and the minted key never touch this component — it only
 * polls whether the flow finished.
 *
 * Two pieces of honesty the UI owes the user:
 *  - On a local install the consent screen is titled `localhost:12319`, because
 *    OpenRouter names localhost apps by host:port. Said up front, that reads as
 *    a detail; discovered mid-flow, it reads as phishing.
 *  - Unlinking is **local**. Revoking a user-controlled key needs an OpenRouter
 *    management key, which AgentX doesn't hold — so the copy says the key still
 *    exists upstream and links to where it can actually be revoked.
 */

import { useEffect, useRef, useState } from 'react';
import { ExternalLink, Link2, Link2Off, Loader2 } from 'lucide-react';
import { api, apiErrorMessage, type ProviderLink } from '../../../lib/api';
import { openExternal } from '../../../lib/openExternal';
import { useNotify } from '../../../contexts/NotificationContext';
import { useConfirm } from '../../ui/ConfirmDialog';
import { Button } from '../../ui';
import { OpenRouterAccount } from './OpenRouterAccount';

/** Poll cadence while a consent tab is open, and the ceiling before we give up
 *  (matching the server's 10-minute flow TTL). */
const POLL_MS = 2000;
const MAX_POLLS = (10 * 60_000) / POLL_MS;

const REVOKE_URL = 'https://openrouter.ai/settings/keys';

export interface OpenRouterLinkProps {
  /** Link metadata from the catalog — present only when linked via OAuth. */
  link?: ProviderLink | null;
  /** Whether any key is stored (a pasted key counts, and has no link record). */
  hasKey: boolean;
  /** Re-read the catalog + health after linking or unlinking. */
  onChanged: () => void;
}

export function OpenRouterLink({ link, hasKey, onChanged }: OpenRouterLinkProps) {
  const { notifyError, notifySuccess } = useNotify();
  const confirm = useConfirm();
  const [pending, setPending] = useState(false);
  const [localCallback, setLocalCallback] = useState(false);
  // Bumped on link/unlink so the balance strip re-reads instead of showing a
  // cached figure for a key that just changed.
  const [accountToken, setAccountToken] = useState(0);
  const flowRef = useRef<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Abandon a pending flow if the user navigates away mid-consent, so a stale
  // nonce isn't left live on the server.
  useEffect(() => {
    return () => {
      stopPolling();
      if (flowRef.current) void api.cancelOpenRouterLink(flowRef.current).catch(() => {});
    };
  }, []);

  const handleLink = async () => {
    setPending(true);
    try {
      const start = await api.startOpenRouterLink();
      flowRef.current = start.flow_id;
      setLocalCallback(start.local_callback);
      await openExternal(start.authorization_url);

      let polls = 0;
      pollRef.current = window.setInterval(async () => {
        polls += 1;
        if (polls > MAX_POLLS) {
          stopPolling();
          flowRef.current = null;
          setPending(false);
          notifyError('The sign-in timed out. Try linking again.', 'OpenRouter');
          return;
        }
        try {
          const status = await api.getOpenRouterLinkStatus(start.flow_id);
          if (status.status === 'pending') return;

          stopPolling();
          flowRef.current = null;
          setPending(false);
          if (status.status === 'linked') {
            notifySuccess('Account linked', 'OpenRouter');
            setAccountToken((t) => t + 1);
            onChanged();
          } else if (status.status === 'error') {
            notifyError(status.error || 'The sign-in failed.', 'OpenRouter');
          } else {
            notifyError('The sign-in expired. Try linking again.', 'OpenRouter');
          }
        } catch {
          // A transient poll failure shouldn't kill the flow — the next tick
          // retries, and MAX_POLLS still bounds it.
        }
      }, POLL_MS);
    } catch (error) {
      setPending(false);
      flowRef.current = null;
      notifyError(apiErrorMessage(error), "Couldn't start the sign-in");
    }
  };

  const handleUnlink = async () => {
    const confirmed = await confirm({
      title: 'Forget this key?',
      body:
        'AgentX will forget the key. It still exists in your OpenRouter account — ' +
        'revoke it there to remove it completely.',
      confirmLabel: 'Forget key',
      danger: true,
    });
    if (!confirmed) return;

    try {
      await api.unlinkOpenRouter();
      notifySuccess('Key forgotten', 'OpenRouter');
      setAccountToken((t) => t + 1);
      onChanged();
    } catch (error) {
      notifyError(error, "Couldn't forget the key");
    }
  };

  if (link || hasKey) {
    return (
      <div className="openrouter-link">
        {link ? (
          <p className="openrouter-link-state">
            <Link2 size={14} aria-hidden />
            Linked account
            {link.user_id && <code className="openrouter-link-user">{link.user_id}</code>}
          </p>
        ) : (
          <p className="openrouter-link-state">
            <Link2 size={14} aria-hidden />
            Key added by hand
          </p>
        )}
        <OpenRouterAccount refreshToken={accountToken} />
        <div className="connection-actions">
          <Button variant="ghost" onClick={handleUnlink}>
            <Link2Off size={16} />
            Forget key
          </Button>
          <Button variant="ghost" onClick={() => void openExternal(REVOKE_URL)}>
            <ExternalLink size={16} />
            Manage keys
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="openrouter-link">
      <Button variant="primary" onClick={handleLink} loading={pending} disabled={pending}>
        {pending ? <Loader2 size={16} className="spin" /> : <Link2 size={16} />}
        {pending ? 'Waiting for OpenRouter…' : 'Link account'}
      </Button>
      <p className="provider-note">
        {pending
          ? 'Finish signing in on the OpenRouter tab. AgentX picks up the key automatically.'
          : 'Sign in once and AgentX gets its own key — no copying and pasting.'}
      </p>
      {pending && localCallback && (
        <p className="provider-note">
          The consent screen will name this app <code>localhost:12319</code> — that&rsquo;s your
          own AgentX server, which is where the key is delivered.
        </p>
      )}
    </div>
  );
}

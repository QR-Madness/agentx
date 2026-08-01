/**
 * OpenRouterAccount — what's left in the tank, and what the alias rename broke.
 *
 * Two supporting facts on the OpenRouter card, both of which the product could
 * previously only learn by leaving the app:
 *
 *  - **Balance and spend.** From `/key` (+ `/credits` when it answers). An
 *    uncapped account reports `limit: null` — that is "no cap", not zero, so it
 *    renders as usage without a gauge rather than an empty meter.
 *  - **Stale `*-latest` ids.** OpenRouter moved its alias family to a `~` prefix.
 *    Ids stored before that still run, but miss our catalog and report a
 *    conservative default context window — surfacing much later as premature
 *    compaction. The repair is **opt-in with the list shown**, because these are
 *    the user's own model choices.
 *
 * Both degrade silently: this is a supporting strip, never the reason the page
 * looks broken.
 */

import { useCallback, useEffect, useState } from 'react';
import { Check, TriangleAlert, Wrench } from 'lucide-react';
import { api, type AliasMigrationScan, type OpenRouterAccount as Account } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { useConfirm } from '../../ui/ConfirmDialog';
import { Button } from '../../ui';

/** Warn once the remaining balance drops below this share of the cap. */
const LOW_BALANCE_RATIO = 0.05;

const money = (value: number | null | undefined): string =>
  value == null ? '—' : `$${value.toFixed(2)}`;

/** Credits bought minus credits used — the number a user actually thinks in.
 *  Only meaningful when `/credits` answered. */
function remainingCredits(account: Account): number | null {
  if (account.total_credits == null || account.total_usage == null) return null;
  return account.total_credits - account.total_usage;
}

export interface OpenRouterAccountProps {
  /** Bumped by the card after link/unlink so the strip re-reads. */
  refreshToken?: number;
}

export function OpenRouterAccount({ refreshToken = 0 }: OpenRouterAccountProps) {
  const { notifyError, notifySuccess } = useNotify();
  const confirm = useConfirm();
  const [account, setAccount] = useState<Account | null>(null);
  const [scan, setScan] = useState<AliasMigrationScan | null>(null);
  const [repairing, setRepairing] = useState(false);

  const load = useCallback(() => {
    api.getOpenRouterAccount().then(setAccount).catch(() => setAccount(null));
    api.scanOpenRouterAliases().then(setScan).catch(() => setScan(null));
  }, []);

  useEffect(load, [load, refreshToken]);

  const handleRepair = async () => {
    if (!scan?.refs.length) return;
    const confirmed = await confirm({
      title: `Update ${scan.count} model reference${scan.count === 1 ? '' : 's'}?`,
      body: (
        <>
          <p>
            OpenRouter renamed these models. The stored names still work, but AgentX can&rsquo;t
            read their real context window — so turns get compacted far earlier than they need to be.
          </p>
          <ul className="alias-repair-list">
            {scan.refs.map((ref) => (
              <li key={`${ref.store}:${ref.location}`}>
                <b>{ref.label}</b>
                <code>{ref.current}</code>
                <span aria-hidden>→</span>
                <code>{ref.suggested}</code>
              </li>
            ))}
          </ul>
        </>
      ),
      confirmLabel: 'Update them',
    });
    if (!confirmed) return;

    setRepairing(true);
    try {
      const result = await api.repairOpenRouterAliases(scan.refs);
      notifySuccess(
        `Updated ${result.count} model reference${result.count === 1 ? '' : 's'}`,
        'OpenRouter'
      );
      if (result.failed.length) {
        notifyError(
          `${result.failed.length} couldn't be updated: ${result.failed
            .map((f) => f.location)
            .join(', ')}`,
          'OpenRouter'
        );
      }
      load();
    } catch (error) {
      notifyError(error, "Couldn't update the model references");
    } finally {
      setRepairing(false);
    }
  };

  const credits = account?.available ? remainingCredits(account) : null;
  const capped = account?.available && account.limit != null;
  const lowOnCap =
    capped && account.limit_remaining != null && account.limit
      ? account.limit_remaining / account.limit < LOW_BALANCE_RATIO
      : false;

  const showStrip = account?.available;
  const showRepair = (scan?.count ?? 0) > 0;
  if (!showStrip && !showRepair) return null;

  return (
    <div className="openrouter-account">
      {showStrip && account && (
        <dl className="openrouter-facts">
          {credits != null ? (
            <div className={lowOnCap ? 'is-low' : undefined}>
              <dt>Balance</dt>
              <dd>
                {money(credits)}
                {account.total_credits != null && (
                  <span className="openrouter-facts-sub"> of {money(account.total_credits)}</span>
                )}
              </dd>
            </div>
          ) : capped ? (
            <div className={lowOnCap ? 'is-low' : undefined}>
              <dt>Key limit left</dt>
              <dd>
                {money(account.limit_remaining)}
                <span className="openrouter-facts-sub"> of {money(account.limit)}</span>
              </dd>
            </div>
          ) : null}

          <div>
            <dt>This month</dt>
            <dd>{money(account.usage_monthly)}</dd>
          </div>

          {/* An uncapped key has no gauge to show — say so rather than
              rendering an empty meter that reads as "nothing left". */}
          {!capped && credits == null && (
            <div>
              <dt>Spend cap</dt>
              <dd>
                None
                <span className="openrouter-facts-sub"> on this key</span>
              </dd>
            </div>
          )}

          {account.is_free_tier && (
            <div>
              <dt>Tier</dt>
              <dd>Free</dd>
            </div>
          )}
        </dl>
      )}

      {lowOnCap && (
        <p className="supply-line-note warning">
          <TriangleAlert size={13} aria-hidden />
          This key is nearly at its spend limit — turns will start failing when it runs out.
        </p>
      )}

      {showRepair && scan && (
        <div className="alias-repair">
          <p className="supply-line-note warning">
            <TriangleAlert size={13} aria-hidden />
            {scan.count} model reference{scan.count === 1 ? '' : 's'} use{scan.count === 1 ? 's' : ''}{' '}
            a name OpenRouter has renamed. They still run, but their context window reads as
            unknown, so turns compact earlier than they should.
          </p>
          <Button variant="secondary" onClick={handleRepair} loading={repairing} disabled={repairing}>
            {repairing ? <Check size={16} /> : <Wrench size={16} />}
            {repairing ? 'Updating…' : `Fix ${scan.count} reference${scan.count === 1 ? '' : 's'}`}
          </Button>
        </div>
      )}
    </div>
  );
}

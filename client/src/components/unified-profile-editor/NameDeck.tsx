/**
 * NameDeck — the dice button beside the profile name input.
 *
 * Deals 10 names from the server pools (Random | Preferred), redeals
 * infinitely, and stars names into the preferred pool (a cheap server-side
 * array). The server never deals a name an existing profile already wears.
 * Picking a chip fills the name input; the deck stays open for browsing.
 */

import { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Dices, Loader2, Star } from 'lucide-react';
import { api, apiErrorMessage } from '../../lib/api';
import {
  IconButton, Popover, PopoverContent, PopoverTrigger, SegmentedControl,
} from '../ui';

type Pool = 'random' | 'preferred';

const nameKey = (name: string) => name.trim().toLowerCase();

interface NameDeckProps {
  onPick: (name: string) => void;
}

export function NameDeck({ onPick }: NameDeckProps) {
  const [open, setOpen] = useState(false);
  const [pool, setPool] = useState<Pool>('random');
  const [names, setNames] = useState<string[]>([]);
  const [preferred, setPreferred] = useState<Set<string>>(new Set());
  const [preferredLoaded, setPreferredLoaded] = useState(false);
  const [dealId, setDealId] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reduce = useReducedMotion();

  const deal = async (p: Pool = pool) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.dealNames(p, 10);
      setNames(res.names);
      setDealId(id => id + 1);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const openDeck = (o: boolean) => {
    setOpen(o);
    if (!o) return;
    void deal();
    if (!preferredLoaded) {
      void api.listPreferredNames()
        .then(res => {
          setPreferred(new Set(res.preferred.map(nameKey)));
          setPreferredLoaded(true);
        })
        .catch(() => { /* stars degrade to add-only */ });
    }
  };

  const switchPool = (p: Pool) => {
    setPool(p);
    void deal(p);
  };

  const toggleStar = async (name: string) => {
    const starred = preferred.has(nameKey(name));
    try {
      const res = await api.updatePreferredNames(starred ? { remove: name } : { add: name });
      setPreferred(new Set(res.preferred.map(nameKey)));
      if (starred && pool === 'preferred') {
        // Unstarred names leave the preferred pool — drop the chip too.
        setNames(ns => ns.filter(n => nameKey(n) !== nameKey(name)));
      }
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  };

  return (
    <Popover open={open} onOpenChange={openDeck}>
      <PopoverTrigger asChild>
        <IconButton size="sm" aria-label="Deal name ideas" title="Deal name ideas" active={open}>
          <Dices size={16} />
        </IconButton>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
            Name deck
          </span>
          <SegmentedControl
            size="sm"
            ariaLabel="Name pool"
            value={pool}
            onChange={switchPool}
            options={[
              { value: 'random', label: 'Random' },
              { value: 'preferred', label: 'Preferred' },
            ]}
          />
        </div>

        {error && <div className="mb-2 text-xs text-error">{error}</div>}

        {names.length === 0 && !loading ? (
          <div className="rounded-md border border-line bg-surface-overlay px-3 py-4 text-center text-xs text-fg-muted">
            {pool === 'preferred'
              ? 'No starred names yet — star a dealt name to keep it here.'
              : 'Every name in this pool is already in use.'}
          </div>
        ) : (
          <motion.div
            key={dealId}
            className="grid grid-cols-2 gap-1.5"
            initial={reduce ? false : 'hidden'}
            animate="show"
            variants={{ show: { transition: { staggerChildren: 0.04 } } }}
          >
            {names.map(n => {
              const starred = preferred.has(nameKey(n));
              return (
                <motion.div
                  key={n}
                  variants={reduce ? {} : { hidden: { opacity: 0, y: 4 }, show: { opacity: 1, y: 0 } }}
                  className="flex min-w-0 items-center rounded-md border border-line bg-surface-overlay transition-colors hover:border-line-strong"
                >
                  <button
                    type="button"
                    className="min-w-0 flex-1 truncate bg-transparent px-2.5 py-1.5 text-left text-sm text-fg"
                    onClick={() => onPick(n)}
                    title={`Use “${n}”`}
                  >
                    {n}
                  </button>
                  <IconButton
                    size="xs"
                    tone={starred ? 'accent' : undefined}
                    aria-label={starred ? `Unstar ${n}` : `Star ${n} as preferred`}
                    title={starred ? 'Remove from preferred' : 'Add to preferred'}
                    onClick={() => void toggleStar(n)}
                    className="mr-0.5 shrink-0"
                  >
                    <Star size={12} fill={starred ? 'currentColor' : 'none'} />
                  </IconButton>
                </motion.div>
              );
            })}
          </motion.div>
        )}

        <button
          type="button"
          onClick={() => void deal()}
          disabled={loading}
          className="mt-2 inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-md border border-line bg-surface-overlay text-sm text-fg transition-colors hover:border-line-strong disabled:opacity-40"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Dices size={14} />}
          Redeal
        </button>
      </PopoverContent>
    </Popover>
  );
}

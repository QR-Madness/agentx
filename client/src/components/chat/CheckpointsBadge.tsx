/**
 * CheckpointsBadge — header affordance surfacing the agent's model-authored
 * checkpoints for the active conversation.
 *
 * Checkpoints are anchors the model writes via the `checkpoint` tool; they live
 * in Redis (7-day TTL) and survive trajectory compression. This badge shows the
 * live count and opens a popover listing each entry with a Clear action. It
 * flashes when the model saves a checkpoint autonomously mid-stream
 * (`flashSignal` increments).
 */

import { useEffect, useRef, useState } from 'react';
import { Flag, X, Trash2 } from 'lucide-react';
import { api } from '../../lib/api';
import { useCheckpoints } from '../../lib/hooks';
import { useNotify } from '../../contexts/NotificationContext';
import { DropdownPortal } from '../ui/DropdownPortal';
import { formatTimestamp } from '../memory/formatTimestamp';
import './CheckpointsBadge.css';

interface CheckpointsBadgeProps {
  conversationId: string | null | undefined;
  /** Increments when the model autonomously saves a checkpoint — triggers a refetch + flash. */
  flashSignal?: number;
}

export function CheckpointsBadge({ conversationId, flashSignal = 0 }: CheckpointsBadgeProps) {
  const { checkpoints, count, refresh } = useCheckpoints(conversationId);
  const { notifySuccess, notifyError } = useNotify();
  const [open, setOpen] = useState(false);
  const [flashing, setFlashing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Refetch + flash when a checkpoint lands mid-stream. Skip the initial render
  // (flashSignal starts at 0) so we don't flash on mount.
  const lastSignal = useRef(flashSignal);
  useEffect(() => {
    if (flashSignal === lastSignal.current) return;
    lastSignal.current = flashSignal;
    refresh();
    setFlashing(true);
    const t = setTimeout(() => setFlashing(false), 1200);
    return () => clearTimeout(t);
  }, [flashSignal, refresh]);

  // Nothing to show until the agent has saved at least one anchor.
  if (!conversationId || count === 0) return null;

  const handleClear = async () => {
    if (!conversationId) return;
    setClearing(true);
    try {
      await api.clearCheckpoints(conversationId);
      await refresh();
      notifySuccess('Checkpoints cleared');
      setOpen(false);
    } catch (err) {
      notifyError(err, 'Failed to clear checkpoints');
    } finally {
      setClearing(false);
    }
  };

  return (
    <>
      <button
        ref={buttonRef}
        className={`checkpoints-badge ${open ? 'active' : ''} ${flashing ? 'flash' : ''}`}
        onClick={() => setOpen(v => !v)}
        title={`${count} checkpoint${count === 1 ? '' : 's'} for this conversation`}
        aria-label={`${count} checkpoints`}
      >
        <Flag size={12} />
        <span>{count}</span>
      </button>

      <DropdownPortal
        isOpen={open}
        onClose={() => setOpen(false)}
        anchorRef={buttonRef}
        preferredSide="bottom"
        align="end"
        estimatedHeight={420}
      >
        <div className="checkpoints-popover" role="dialog" aria-label="Checkpoints">
          <div className="checkpoints-popover-header">
            <Flag size={14} />
            <span>Checkpoints</span>
            <button
              className="checkpoints-clear"
              onClick={handleClear}
              disabled={clearing}
              title="Clear all checkpoints for this conversation"
            >
              <Trash2 size={12} />
              <span>Clear</span>
            </button>
            <button
              className="checkpoints-close"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              <X size={14} />
            </button>
          </div>

          <p className="checkpoints-hint">
            Anchors the agent saved to keep its bearings on long tasks. They
            persist across compression and reloads (7-day expiry).
          </p>

          <ol className="checkpoints-list">
            {checkpoints.map((cp, i) => (
              <li key={`${cp.created_at}-${i}`} className="checkpoint-item">
                <div className="checkpoint-item-head">
                  <span className="checkpoint-index">{i + 1}</span>
                  <span className="checkpoint-summary">{cp.summary}</span>
                </div>
                {cp.decisions.length > 0 && (
                  <ul className="checkpoint-decisions">
                    {cp.decisions.map((d, j) => (
                      <li key={j}>{d}</li>
                    ))}
                  </ul>
                )}
                {cp.next_step && (
                  <div className="checkpoint-next">
                    <span className="checkpoint-next-label">Next</span>
                    {cp.next_step}
                  </div>
                )}
                <div className="checkpoint-time">{formatTimestamp(cp.created_at)}</div>
              </li>
            ))}
          </ol>
        </div>
      </DropdownPortal>
    </>
  );
}

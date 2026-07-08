/**
 * ConversationStateBadge — compact composer indicator for the conversation's
 * structured working memory (goals/decisions/open threads/artifacts/narrative).
 *
 * Sits beside the context-usage chip in the composer's input-stats line. Shows a
 * one-line slot summary, flashes when the agent writes state mid-stream
 * (`flashSignal`), refetches when the drawer saves (a window event), and opens
 * the editable {@link ConversationStateDrawer} on click. Hidden until the
 * conversation has some state.
 */

import { useEffect, useRef, useState } from 'react';
import { useConversationState } from '../../lib/hooks';
import { useModal } from '../../contexts/ModalContext';
import { SURFACES } from '../../lib/surfaces';
import type { ConversationStateSlot } from '../../lib/api/types';
import './ConversationStateDrawer.css';

interface Props {
  conversationId: string | null | undefined;
  /** Increments when the agent writes state mid-stream — triggers a refetch + flash. */
  flashSignal?: number;
}

const SHORT: Record<ConversationStateSlot, [string, string]> = {
  decisions: ['decision', 'decisions'],
  open_threads: ['open thread', 'open threads'],
  goals: ['goal', 'goals'],
  artifacts: ['artifact', 'artifacts'],
  narrative: ['note', 'notes'],
};
const ORDER: ConversationStateSlot[] = ['decisions', 'open_threads', 'goals', 'artifacts', 'narrative'];

function summarize(counts: Record<ConversationStateSlot, number>): string {
  const parts = ORDER.filter((k) => counts[k] > 0).map((k) => {
    const [s, p] = SHORT[k];
    return `${counts[k]} ${counts[k] === 1 ? s : p}`;
  });
  if (parts.length <= 2) return parts.join(', ');
  return `${parts.slice(0, 2).join(', ')}…`;
}

export function ConversationStateBadge({ conversationId, flashSignal = 0 }: Props) {
  const { counts, total, refresh } = useConversationState(conversationId);
  const { openModal } = useModal();
  const [flashing, setFlashing] = useState(false);

  // Refetch + flash when the agent writes state mid-stream. Skip mount (starts 0).
  const lastSignal = useRef(flashSignal);
  useEffect(() => {
    if (flashSignal === lastSignal.current) return;
    lastSignal.current = flashSignal;
    refresh();
    setFlashing(true);
    const t = setTimeout(() => setFlashing(false), 1200);
    return () => clearTimeout(t);
  }, [flashSignal, refresh]);

  // The drawer (a separate instance) dispatches this after a user save.
  useEffect(() => {
    const onUpdate = (e: Event) => {
      const detail = (e as CustomEvent).detail as { conversationId?: string } | undefined;
      if (!conversationId || detail?.conversationId === conversationId) refresh();
    };
    window.addEventListener('conversationstate:updated', onUpdate);
    return () => window.removeEventListener('conversationstate:updated', onUpdate);
  }, [conversationId, refresh]);

  if (!conversationId || total === 0) return null;

  return (
    <button
      type="button"
      className={`conv-state-chip ${flashing ? 'flash' : ''}`}
      onClick={() => openModal({ ...SURFACES.conversationState, props: { conversationId } })}
      title="Conversation state — the agent's structured working memory. Click to view and edit."
    >
      {' · '}state: {summarize(counts)}
    </button>
  );
}

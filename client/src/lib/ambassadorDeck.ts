/**
 * Standalone Ambassador "Command Deck" — pure helpers shared by the deck surface.
 *
 * The deck is the ambassador opened *without* a conversation: a full-screen front door
 * for "what have my agents been doing / discovered?". It runs against a single persistent
 * per-user thread (a minted id passed where a conversation_id normally goes), so its
 * cross-conversation tools (survey/roster) have a home with no chat open. Keeping the id
 * derivation + starter prompts here makes them testable and keeps `AmbassadorPanel` lean.
 */

/**
 * The deck's persistent thread id. Per-user when auth is on (so histories don't bleed on a
 * shared server), `deck:default` otherwise. The `deck:` prefix can't collide with a real
 * conversation id (those are hex uuids — no colon), and since the ambassador only writes to
 * its Redis sidecar (never `conversation_logs`), this thread never shows up in its own survey.
 */
export function deckThreadId(userId?: number | string | null): string {
  const id = userId === 0 ? '0' : userId ? String(userId) : 'default';
  return `deck:${id}`;
}

export interface DeckInquiry {
  thread_id: string;
  title: string;
  updated_at?: string | null;
}

/**
 * Order the deck's Inquiry list for display: the home deck thread first (pinned), then
 * the rest newest-first by `updated_at`. Pure so it's unit-testable; the registry already
 * returns newest-first, but this guarantees the pin regardless of server order.
 */
export function orderInquiries(threads: DeckInquiry[], deckThreadId: string): DeckInquiry[] {
  const rest = threads
    .filter((t) => t.thread_id !== deckThreadId)
    .sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));
  const home = threads.find((t) => t.thread_id === deckThreadId);
  return home ? [home, ...rest] : rest;
}

/** The label to show for an Inquiry chip/row — its title, or a stable fallback. */
export function inquiryLabel(inq: DeckInquiry, deckThreadId: string): string {
  if (inq.title.trim()) return inq.title.trim();
  return inq.thread_id === deckThreadId ? 'Command Deck' : 'New Inquiry';
}

export interface DeckStarter {
  label: string;
  prompt: string;
}

/**
 * Starter chips for the deck — survey + roster oriented, so the first tap exercises the
 * conversation-agnostic tools (`survey_conversations`, `list_agents`) rather than anything
 * that needs a focused conversation.
 */
export const DECK_STARTERS: DeckStarter[] = [
  {
    label: 'What have my agents been working on?',
    prompt: 'Survey my recent conversations and summarize what my agents have been working on.',
  },
  {
    label: 'List my agents',
    prompt: 'List my agents and what each one can do.',
  },
  {
    label: 'What have my agents discovered?',
    prompt: 'Across my recent conversations, what have my agents discovered or concluded lately?',
  },
];

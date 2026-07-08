/**
 * conversationTitle — shared "auto-title a conversation" helper used by the
 * sidebar row menu, the Relay menu, and anywhere else that wants a smart title.
 *
 * Assembles the compact, high-signal inputs the titler wants — the conversation
 * state plus the first and latest message (both truncated) — and calls the
 * `/api/prompts/title` endpoint. Best-effort: returns null on any failure so
 * callers can just surface a toast and move on.
 */

import { api } from './api';

/** Loose message shape — tab messages carry `type`, server messages carry `role`. */
interface LooseMessage {
  content?: string;
  role?: string;
  type?: string;
}

const TRUNCATE = 400;
const STATE_TRUNCATE = 600;

function clip(text: string | undefined, limit: number): string {
  const t = (text ?? '').trim();
  return t.length <= limit ? t : `${t.slice(0, limit).trimEnd()}…`;
}

function roleOf(m: LooseMessage): string {
  return (m.role ?? m.type ?? '').toLowerCase();
}

/** A short, plain-text rendering of the conversation-state slots for the titler. */
async function renderState(conversationId: string): Promise<string> {
  try {
    const { state } = await api.getConversationState(conversationId);
    const parts: string[] = [];
    const push = (label: string, entries: { text: string }[] | undefined) => {
      const texts = (entries ?? []).map(e => e.text).filter(Boolean);
      if (texts.length) parts.push(`${label}: ${texts.join('; ')}`);
    };
    push('Goals', state.goals);
    push('Decisions', state.decisions);
    push('Open threads', state.open_threads);
    return clip(parts.join('\n'), STATE_TRUNCATE);
  } catch {
    return '';
  }
}

/**
 * Generate a title for a conversation. Pass the caller's in-memory messages (an
 * open tab's) to avoid a fetch; otherwise they're pulled from the server. Returns
 * the title, or null if there's nothing to title / the request failed.
 */
export async function generateTitleFor(
  conversationId: string | null | undefined,
  localMessages?: LooseMessage[],
): Promise<string | null> {
  let messages: LooseMessage[] = localMessages?.length ? localMessages : [];
  if (!messages.length && conversationId) {
    try {
      messages = (await api.getConversationMessages(conversationId)).messages;
    } catch {
      messages = [];
    }
  }

  const conversational = messages.filter(m => {
    const r = roleOf(m);
    return r === 'user' || r === 'assistant';
  });
  const first = clip((conversational.find(m => roleOf(m) === 'user') ?? conversational[0])?.content, TRUNCATE);
  const last = clip(conversational[conversational.length - 1]?.content, TRUNCATE);
  const state = conversationId ? await renderState(conversationId) : '';

  if (!first && !last && !state) return null;

  try {
    const { title } = await api.generateTitle({ state, first, last });
    return title.trim() || null;
  } catch {
    return null;
  }
}

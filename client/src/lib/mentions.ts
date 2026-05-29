/**
 * Client @-mention helpers for inline agent routing (16.5 client follow-up).
 *
 * The backend parses `@[\w-]+` from the message text and routes the turn; these
 * pure helpers power the composer autocomplete and the sent-bubble emphasis.
 * Mirrors `api/agentx_ai/agent/mentions.py`.
 */

import type { AgentProfile } from './api/types';

/** A token char is a word char or hyphen (agent_ids are adj-adj-noun). */
const TOKEN_CHAR = /[\w-]/;
/** Left boundary: '@' must not directly follow a word char, '/', or '@' (skips emails/paths). */
const BOUNDARY_BEFORE = /[\w/@]/;

export interface ActiveMention {
  /** Text between '@' and the caret (may be empty right after '@'). */
  query: string;
  /** Index of the '@'. */
  start: number;
  /** Caret index (end of the mention span being typed). */
  end: number;
}

/**
 * Return the `@token` the caret is currently inside, or null.
 *
 * Scans left from `caret` for an unbroken run of token chars terminated by an
 * '@' with a valid left boundary. Returns null if the caret sits past a space
 * or outside any mention.
 */
export function getActiveMention(text: string, caret: number): ActiveMention | null {
  let i = caret - 1;
  while (i >= 0 && TOKEN_CHAR.test(text[i])) i--;
  // text[i] must be the '@'.
  if (i < 0 || text[i] !== '@') return null;
  // Left-boundary check: char before '@' (if any) must not be word/'/'/'@'.
  if (i > 0 && BOUNDARY_BEFORE.test(text[i - 1])) return null;
  return { query: text.slice(i + 1, caret), start: i, end: caret };
}

/**
 * Replace the `[start, end)` span with `@<slug> ` (trailing space). Returns the
 * new text and the caret offset just after the inserted token.
 */
export function applyMention(
  text: string,
  span: { start: number; end: number },
  slug: string,
): { text: string; caret: number } {
  const insert = `@${slug} `;
  const next = text.slice(0, span.start) + insert + text.slice(span.end);
  return { text: next, caret: span.start + insert.length };
}

/** Resolve a single token (no '@') to an agent_id, or null. agent_id first, then single-word name. */
export function resolveMentionToken(token: string, profiles: AgentProfile[]): string | null {
  const byId = profiles.find(p => p.agentId === token);
  if (byId) return byId.agentId;
  const lowered = token.toLowerCase();
  const byName = profiles.find(p => p.name.toLowerCase() === lowered);
  return byName ? byName.agentId : null;
}

const MENTION_RE = /(?:^|[^\w/@])@([\w-]+)/g;

/** Ordered, deduped agent_ids for every resolvable `@mention` in the text. */
export function extractMentionedAgentIds(text: string, profiles: AgentProfile[]): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(MENTION_RE)) {
    const agentId = resolveMentionToken(m[1], profiles);
    if (agentId && !out.includes(agentId)) out.push(agentId);
  }
  return out;
}

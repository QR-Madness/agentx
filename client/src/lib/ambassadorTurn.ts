/**
 * Turn-context extraction for the Ambassador (Phase 16.6).
 *
 * The ambassador briefs one assistant turn, but the agent's prose is only half
 * the turn — it also *did* things (searched the web, pulled sources, built a
 * table). This gathers that substance from the in-memory transcript so the
 * briefing can interpret the turn instead of merely paraphrasing the reply.
 *
 * Done client-side (not from the server transcript) because a just-finished
 * live turn may not be persisted yet — the same reason briefings are keyed by
 * client message id. The payload is deliberately compact + capped: the
 * ambassador needs to know *what* happened, not ingest raw tool output.
 */

import {
  isUserMessage,
  isToolCallMessage,
  isToolResultMessage,
  isExhibitMessage,
  type AssistantMessage,
  type ConversationMessage,
} from './messages';
import type { AmbassadorTurnArtifacts, AmbassadorSource } from './api';

/**
 * Resolve the display name of the agent that produced a turn, for the briefing
 * to speak of it by name. Mirrors (and sharpens) the transcript's own fallback
 * chain: the turn's stamped `agentName` → the profile that produced it (by
 * `profileId`, correct in multi-agent conversations) → a conversation-level
 * fallback. Restored/older turns often lack a stamped `agentName`, so without
 * this the briefing silently degrades to a generic "your agent".
 */
export function resolveTurnAgentName(
  message: Pick<AssistantMessage, 'agentName' | 'profileId'>,
  opts: { nameByProfileId?: (id: string) => string | undefined; fallback?: string } = {},
): string {
  const direct = message.agentName?.trim();
  if (direct) return direct;
  if (message.profileId && opts.nameByProfileId) {
    const byProfile = opts.nameByProfileId(message.profileId)?.trim();
    if (byProfile) return byProfile;
  }
  return opts.fallback?.trim() ?? '';
}

// Bounds so a tool-heavy turn can't bloat the briefing prompt.
const MAX_TOOLS = 12;
const MAX_SOURCES = 12;
const MAX_EXHIBITS = 8;
const ARG_DETAIL_MAX = 120;
const RESULT_PREVIEW_MAX = 180;

function squish(text: string, max: number): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/** Pick the most briefing-relevant arg (a query/url/path/text) or compact JSON. */
function summarizeArgs(args: Record<string, unknown>): string | undefined {
  for (const key of ['query', 'q', 'url', 'path', 'text', 'question', 'name']) {
    const v = args[key];
    if (typeof v === 'string' && v.trim()) return squish(v, ARG_DETAIL_MAX);
  }
  const keys = Object.keys(args);
  if (keys.length === 0) return undefined;
  try {
    return squish(JSON.stringify(args), ARG_DETAIL_MAX);
  } catch {
    return undefined;
  }
}

/** Nearest preceding user message — gives the ambassador the turn's prompt. */
function precedingUserText(messages: ConversationMessage[], assistantIndex: number): string {
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    const m = messages[i];
    if (m && isUserMessage(m)) return m.content;
  }
  return '';
}

/**
 * Gather a turn's prompt text + the artifacts produced between the preceding
 * user message and the briefed assistant reply. `artifacts` is omitted when the
 * turn was pure prose (nothing to add beyond the reply itself).
 */
export function gatherTurnContext(
  messages: ConversationMessage[],
  assistantMessageId: string,
): { userText: string; artifacts?: AmbassadorTurnArtifacts } {
  const assistantIndex = messages.findIndex((m) => m.id === assistantMessageId);
  if (assistantIndex < 0) return { userText: '' };

  const userText = precedingUserText(messages, assistantIndex);

  // The turn's artifacts are everything since the preceding user message.
  let start = 0;
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (isUserMessage(messages[i]!)) {
      start = i + 1;
      break;
    }
  }
  const slice = messages.slice(start, assistantIndex);

  // tool_result content can arrive as a separate message — index it by call id
  // so a tool_call without an inline result still gets a preview.
  const resultById = new Map<string, string>();
  for (const m of slice) {
    if (isToolResultMessage(m) && m.content) resultById.set(m.toolCallId, m.content);
  }

  const tools: NonNullable<AmbassadorTurnArtifacts['tools']> = [];
  const sources: AmbassadorSource[] = [];
  const exhibits: NonNullable<AmbassadorTurnArtifacts['exhibits']> = [];
  const seenSource = new Set<string>();

  for (const m of slice) {
    if (isToolCallMessage(m)) {
      if (tools.length >= MAX_TOOLS) continue;
      const raw = m.result?.content ?? resultById.get(m.toolCallId);
      tools.push({
        name: m.toolName,
        detail: summarizeArgs(m.arguments ?? {}),
        ok: m.result?.success ?? (m.status === 'completed' ? true : undefined),
        result: raw ? squish(raw, RESULT_PREVIEW_MAX) : undefined,
      });
    } else if (isExhibitMessage(m)) {
      for (const el of m.exhibit.elements) {
        if (el.type === 'citation') {
          for (const s of el.sources) {
            const key = s.url || s.label;
            if (!key || seenSource.has(key) || sources.length >= MAX_SOURCES) continue;
            seenSource.add(key);
            sources.push({ label: s.label || s.url || '', url: s.url });
          }
        } else if (exhibits.length < MAX_EXHIBITS) {
          if (el.type === 'table') {
            exhibits.push({
              kind: 'table',
              title: el.title || el.caption,
              detail: `${el.columns.length} columns × ${el.rows.length} rows`,
            });
          } else if (el.type === 'mermaid') {
            exhibits.push({ kind: 'diagram', title: el.title });
          } else if (el.type === 'choice') {
            exhibits.push({
              kind: 'choice',
              title: el.title,
              detail: `${el.options.length} options`,
            });
          }
        }
      }
    }
  }

  if (tools.length === 0 && sources.length === 0 && exhibits.length === 0) {
    return { userText };
  }
  const artifacts: AmbassadorTurnArtifacts = {};
  if (tools.length) artifacts.tools = tools;
  if (sources.length) artifacts.sources = sources;
  if (exhibits.length) artifacts.exhibits = exhibits;
  return { userText, artifacts };
}

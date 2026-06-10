/**
 * Ambassador API (Phase 16.6) — the client surface for the parallel,
 * non-polluting conversation interpreter. Kept as its own domain module (rather
 * than folded into the chat stream surface) because an ambassador run is a
 * separate, namespaced (`ambassador_*`) event stream.
 */

import { request as apiRequest, getBaseUrl, registerStreamController } from './core';
import { getAuthToken, getActiveGatewayToken } from '../storage';

export type AmbassadorStatus = 'streaming' | 'done' | 'error' | 'empty_provider' | 'cancelled';

/** One read-only tool the *ambassador* called while answering (its own tool belt —
 *  summarize/explore/read/list a conversation). Surfaced live as a chip so you can
 *  see it reading/surveying. Distinct from `AmbassadorToolArtifact` (the watched
 *  agent's tools). Live-only for now — not persisted to the sidecar. */
export interface AmbassadorToolCall {
  tool: string;
  args?: Record<string, unknown>;
  /** True once the tool returned (its `ambassador_tool_result` arrived). */
  done?: boolean;
}

/** One persisted per-turn briefing (sidecar record / live state). */
export interface AmbassadorBriefing {
  message_id: string;
  status: AmbassadorStatus;
  summary: string;
  error?: string;
  run_id?: string;
  created_at?: string;
  updated_at?: string;
  /** Read-only tools the ambassador called while composing this briefing (live). */
  toolCalls?: AmbassadorToolCall[];
}

/** One tool the agent ran during the turn (compacted for grounding). */
export interface AmbassadorToolArtifact {
  name: string;
  /** Compact arg summary — e.g. the query/url/path. */
  detail?: string;
  ok?: boolean;
  /** Truncated result preview (a hint, never the full output). */
  result?: string;
}

/** One source the turn cited (from a citation exhibit). */
export interface AmbassadorSource {
  label: string;
  url?: string;
}

/** A non-citation artifact the turn presented (table, diagram, choice). */
export interface AmbassadorExhibitArtifact {
  kind: string;
  title?: string;
  detail?: string;
}

/**
 * The substance of a turn beyond the agent's prose — what it actually *did*
 * (searched, pulled sources, built a table). Lets the ambassador interpret the
 * turn instead of merely paraphrasing the reply. Compacted + capped client-side.
 */
export interface AmbassadorTurnArtifacts {
  tools?: AmbassadorToolArtifact[];
  sources?: AmbassadorSource[];
  exhibits?: AmbassadorExhibitArtifact[];
}

/** One persisted free-form Q&A entry (sidecar record / live state). */
export interface AmbassadorQA {
  qa_id: string;
  question: string;
  answer: string;
  status: AmbassadorStatus;
  error?: string;
  run_id?: string;
  created_at?: string;
  updated_at?: string;
  /** Read-only tools the ambassador called while answering this question (live). */
  toolCalls?: AmbassadorToolCall[];
}

/** The conversation the person is *currently in* (their active chat tab) — ambient
 *  context distinct from the ambassador's focus, so it knows where they are now. */
export interface AmbassadorActiveConversation {
  id: string;
  title?: string;
}

export interface AskAmbassadorRequest {
  conversation_id: string;
  /** Client-stable id the Q&A is keyed by. */
  qa_id: string;
  question: string;
  /** Resolved display name of the conversation's agent, so the answer names it. */
  agent_name?: string;
  /** Latest-turn substance, as extra grounding for the answer. */
  artifacts?: AmbassadorTurnArtifacts;
  /** Where the person currently is (ambient context, distinct from the focus). */
  active_conversation?: AmbassadorActiveConversation;
}

export interface DraftRelayRequest {
  conversation_id: string;
  /** The user's rough intent; the ambassador shapes it into a ready-to-send message. */
  intent: string;
  agent_name?: string;
  artifacts?: AmbassadorTurnArtifacts;
}

export interface BriefTurnRequest {
  conversation_id: string;
  message_id: string;
  assistant_text: string;
  user_text?: string;
  /** Display name of the briefed agent, so the briefing speaks of it by name. */
  agent_name?: string;
  /** What the agent actually did this turn (tools, sources, exhibits). */
  artifacts?: AmbassadorTurnArtifacts;
}

export interface SpeakRequest {
  /** The text to speak (a briefing summary or Q&A answer). */
  text: string;
  /** Ambassador profile whose speech model/voice to use (default ambassador if omitted). */
  agent_profile_id?: string;
  /** Optional per-call overrides. */
  voice?: string;
  model?: string;
}

export type VoiceCommandAction = 'answer' | 'relay';

export interface VoiceCommandRequest {
  conversation_id: string;
  /** The transcribed spoken command. */
  transcript: string;
  agent_name?: string;
  artifacts?: AmbassadorTurnArtifacts;
  /** Where the person currently is (ambient context, distinct from the focus). */
  active_conversation?: AmbassadorActiveConversation;
}

export interface VoiceCommandResult {
  /** `answer` = the ambassador answers you (spoken); `relay` = a draft to send to the agent. */
  action: VoiceCommandAction;
  text: string;
  /** Set when an `answer` was persisted to the Q&A sidecar. */
  qa_id?: string | null;
}

export interface TranscribeRequest {
  /** Base64-encoded audio (no data-URI prefix). */
  audio: string;
  /** OpenRouter format token (webm | m4a | ogg | wav | mp3). */
  format: string;
  /** Ambassador profile whose STT model to use (default ambassador if omitted). */
  agent_profile_id?: string;
  /** Optional per-call overrides. */
  model?: string;
  /** Optional ISO-639-1 language hint. */
  language?: string;
}

export interface AmbassadorStreamCallbacks {
  onChunk?: (text: string) => void;
  onDone?: (summary: string, status: AmbassadorStatus) => void;
  onError?: (error: string) => void;
  /** The ambassador started a read-only tool (summarize/explore/read/list). */
  onToolCall?: (tool: string, args?: Record<string, unknown>) => void;
  /** A tool returned. */
  onToolResult?: (tool: string) => void;
  /** The run's event buffer expired — fall back to the persisted briefing. */
  onMissing?: () => void;
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getAuthToken();
  if (token) headers['X-Auth-Token'] = token;
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) headers['AgentX-Gateway-Token'] = gatewayToken;
  return headers;
}

/** Minimal SSE pump for the namespaced ambassador event set. */
async function pumpAmbassadorSse(
  response: Response,
  callbacks: AmbassadorStreamCallbacks,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
  const state = { type: '', data: '' };

  const handle = (type: string, data: Record<string, unknown>) => {
    switch (type) {
      case 'ambassador_chunk':
        if (typeof data.text === 'string') callbacks.onChunk?.(data.text);
        break;
      case 'ambassador_done':
        callbacks.onDone?.(
          typeof data.summary === 'string' ? data.summary : '',
          (data.status as AmbassadorStatus) || 'done',
        );
        break;
      case 'ambassador_error':
        callbacks.onError?.(typeof data.error === 'string' ? data.error : 'Briefing failed');
        break;
      case 'ambassador_tool_call':
        if (typeof data.tool === 'string')
          callbacks.onToolCall?.(data.tool, (data.args as Record<string, unknown>) ?? undefined);
        break;
      case 'ambassador_tool_result':
        if (typeof data.tool === 'string') callbacks.onToolResult?.(data.tool);
        break;
      case 'run_missing':
        callbacks.onMissing?.();
        break;
      default:
        break; // ambassador_start, close — no-op
    }
  };

  const processLines = (lines: string[]) => {
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        state.type = line.slice(7);
      } else if (line.startsWith('data: ')) {
        state.data = line.slice(6);
        if (state.type && state.data) {
          try {
            handle(state.type, JSON.parse(state.data));
          } catch {
            /* ignore malformed frame */
          }
          state.type = '';
          state.data = '';
        }
      }
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    processLines(lines);
  }
  if (buffer.trim()) processLines(buffer.split('\n'));
}

export const ambassadorApi = {
  /** The shipped default persona text for the ambassador's functional voices,
   *  so the editor can show + diff overrides against them. */
  async ambassadorPersonaDefaults(): Promise<{ briefing: string; qa: string; draft: string }> {
    return apiRequest('/api/agent/ambassador/persona-defaults');
  },

  /** Kick off a parallel briefing of one turn; returns its detached run_id. */
  async briefTurn(req: BriefTurnRequest): Promise<{ run_id: string }> {
    return apiRequest('/api/agent/ambassador/brief-turn', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /** Replay a conversation's persisted briefings + Q&A (cold-open / reload). */
  async fetchAmbassadorBriefings(
    conversationId: string,
  ): Promise<{ briefings: AmbassadorBriefing[]; qa: AmbassadorQA[] }> {
    const res = await apiRequest<{
      conversation_id: string;
      briefings: AmbassadorBriefing[];
      qa?: AmbassadorQA[];
    }>(`/api/agent/ambassador/${encodeURIComponent(conversationId)}`);
    return { briefings: res.briefings ?? [], qa: res.qa ?? [] };
  },

  /** Ask the ambassador a free-form question about a conversation. */
  async askAmbassador(req: AskAmbassadorRequest): Promise<{ run_id: string; qa_id: string }> {
    return apiRequest('/api/agent/ambassador/ask', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * Draft a message FROM you TO the agent (the outbound relay) from a rough intent.
   * Returns the draft for you to review/edit before sending — it never sends.
   */
  async draftRelay(req: DraftRelayRequest): Promise<{ draft: string }> {
    return apiRequest('/api/agent/ambassador/draft', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * Synthesize spoken audio (MP3) for a briefing / Q&A answer via the ambassador's
   * speech model. Returns the audio Blob. Throws an `ApiError`-shaped object on
   * failure — notably a 422 with `code: 'voice_unconfigured'` when no OpenRouter
   * key is set (the response body is JSON in that case, not audio).
   */
  async speak(req: SpeakRequest, signal?: AbortSignal): Promise<Blob> {
    const baseUrl = getBaseUrl();
    const response = await fetch(`${baseUrl}/api/agent/ambassador/speak`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(req),
      signal,
    });
    if (!response.ok) {
      let message = `Speech synthesis failed (${response.status})`;
      let code: string | undefined;
      try {
        const body = await response.json();
        if (typeof body?.error === 'string') message = body.error;
        if (typeof body?.code === 'string') code = body.code;
      } catch {
        /* non-JSON error body — keep the generic message */
      }
      throw { message, status: response.status, kind: 'http', details: { code } };
    }
    return response.blob();
  },

  /**
   * Transcribe a push-to-talk recording to text via the ambassador's STT model.
   * Returns `{text}` for the user to review/edit — the caller never auto-sends it.
   * Throws an `ApiError`-shaped object on failure (e.g. 422 `transcription_unconfigured`).
   */
  async transcribe(req: TranscribeRequest): Promise<{ text: string }> {
    return apiRequest('/api/agent/ambassador/transcribe', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * Route a spoken voice command: the ambassador infers intent and either answers
   * it (spoken Q&A) or returns a `relay` draft for the user to review and send.
   * Never throws server-side errors into the call — degrades to a spoken notice.
   */
  async voiceCommand(req: VoiceCommandRequest): Promise<VoiceCommandResult> {
    return apiRequest('/api/agent/ambassador/voice-command', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /** Tail a briefing run's `ambassador_*` SSE stream. Returns an abort handle. */
  streamAmbassador(
    runId: string,
    callbacks: AmbassadorStreamCallbacks,
  ): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/ambassador/stream?run_id=${encodeURIComponent(runId)}`, {
      method: 'GET',
      headers: authHeaders(),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Ambassador stream failed: ${response.status}`);
        await pumpAmbassadorSse(response, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(error?.message ?? 'Ambassador stream failed');
        }
      });

    return { abort: () => controller.abort() };
  },
};

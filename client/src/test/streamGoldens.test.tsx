/**
 * Golden stream replay — the streaming engine's contract suite.
 *
 * Each fixture under fixtures/streams/ is a REAL recorded run (see
 * scripts/capture_stream_fixture.py): the verbatim SSE stream off the run's
 * Redis event bus plus the persisted conversation payload. Tests replay the
 * stream through the production wiring — `useChatStream.attach()` → the real
 * `dispatchSseEvent` → handlers → reducer → message store — and assert:
 *
 *  1. the normalized terminal transcript structure (snapshot per scenario),
 *  2. live-vs-restored parity for the delegation family: every delegation /
 *     work-order card the live stream produced must survive a reload with the
 *     same identity, status, mode, and report-delivered state (live ⊆
 *     restored — the persisted payload may span a longer conversation than
 *     the single captured run, and user bubbles are client-authored on send,
 *     so full-transcript equality is deliberately NOT asserted; the restored
 *     structure is pinned in the same snapshot instead),
 *  3. generic invariants (nothing left "streaming" after the stream settles,
 *     the pump stops at the first close, every delegation reaches a terminal
 *     state).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

import { useChatStream } from '../components/chat/useChatStream';
import { NotificationProvider } from '../contexts/NotificationContext';
import { api } from '../lib/api';
import { dispatchSseEvent, type StreamCallbacks } from '../lib/api/streaming';
import { mapServerMessages } from '../contexts/conversation/mapServerMessages';
import type { ServerMessage } from '../lib/api';
import type { ConversationMessage } from '../lib/messages';

// Fixture files as raw text via Vite's glob (the importBoundary.test.ts
// pattern) — no node fs, works in the jsdom environment.
const rawFixtures = import.meta.glob('/src/test/fixtures/streams/*/*', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const SCENARIOS = [...new Set(
  Object.keys(rawFixtures)
    .map(k => k.match(/\/streams\/([^/]+)\//)?.[1])
    .filter((s): s is string => !!s),
)].sort();

interface FixtureEvent { type: string; data: Record<string, unknown> }

function loadFixture(name: string): {
  events: FixtureEvent[];
  conversation: ServerMessage[];
} {
  const raw = (file: string) => rawFixtures[`/src/test/fixtures/streams/${name}/${file}`];
  const events: FixtureEvent[] = [];
  for (const line of (raw('events.jsonl') ?? '').split('\n')) {
    if (!line.trim()) continue;
    const parsed = JSON.parse(line);
    if (parsed._state) continue; // run-state header line
    const sse: string = parsed.sse;
    if (!sse.startsWith('event: ')) continue;
    const type = sse.split('\n', 1)[0].slice(7).trim();
    const dataLine = sse.split('\n').find(l => l.startsWith('data: '));
    if (!dataLine) continue;
    events.push({ type, data: JSON.parse(dataLine.slice(6)) });
  }
  return {
    events,
    conversation: JSON.parse(raw('conversation.json') ?? '[]'),
  };
}

/** Message-store double with useTabMessages semantics (append; merge-by-id). */
function makeStore() {
  const messages: ConversationMessage[] = [];
  return {
    messages,
    appendMessage: (m: ConversationMessage) => { messages.push(m); },
    updateMessage: (id: string, patch: Partial<ConversationMessage>) => {
      const i = messages.findIndex(m => m.id === id);
      if (i >= 0) messages[i] = { ...messages[i], ...patch } as ConversationMessage;
    },
  };
}

/** Replay a fixture through the production attach wiring. Returns the store +
 * how many events the pump consumed before a `close` stopped it. */
function replay(events: FixtureEvent[]) {
  const store = makeStore();
  let captured: StreamCallbacks | null = null;
  const attachSpy = vi.spyOn(api, 'attachChatRun').mockImplementation(
    (_runId: string, callbacks: StreamCallbacks) => {
      captured = callbacks;
      return { abort: () => {} };
    },
  );

  const wrapper = ({ children }: { children: ReactNode }) => (
    <NotificationProvider>{children}</NotificationProvider>
  );
  const hook = renderHook(
    () => useChatStream({
      appendMessage: store.appendMessage,
      updateMessage: store.updateMessage,
      resolveAgentName: (agentId: string) => `name(${agentId})`,
    }),
    { wrapper },
  );

  act(() => { hook.result.current.attach('golden-run'); });
  expect(captured).not.toBeNull();

  const controller = new AbortController();
  let consumed = 0;
  act(() => {
    for (const ev of events) {
      consumed += 1;
      if (dispatchSseEvent(ev.type, ev.data, captured!, controller)) break;
    }
  });

  attachSpy.mockRestore();
  hook.unmount();
  return { store, consumed, state: hook.result.current.state };
}

/** Normalize a transcript to its structural skeleton (ids/timestamps/content
 * scrubbed; delegation ids kept — they're fixture-deterministic). */
function normalize(messages: ConversationMessage[]) {
  return messages.map(m => {
    switch (m.type) {
      case 'user':
        return { type: m.type, steered: m.steered === true };
      case 'assistant':
        return { type: m.type, hasContent: !!(m.content && m.content.trim()) };
      case 'delegation':
        return {
          type: m.type,
          delegationId: m.delegationId,
          target: m.targetAgentId,
          status: m.status,
          mode: m.mode ?? 'await',
          reportDelivered: m.reportDelivered === true,
        };
      case 'work_order_report':
        return { type: m.type, delegationId: m.delegationId, status: m.status };
      case 'tool_call':
        return { type: m.type, tool: m.toolName, status: m.status };
      case 'plan_execution':
        return {
          type: m.type,
          status: m.status,
          subtasks: m.subtasks.map(s => s.status),
        };
      case 'exhibit':
        return { type: m.type, elements: m.exhibit.elements.map(e => e.type) };
      default:
        return { type: m.type };
    }
  });
}

/** Multiset key for order-insensitive parity comparison. */
function multiset(entries: ReturnType<typeof normalize>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const e of entries) {
    const k = JSON.stringify(e);
    out[k] = (out[k] ?? 0) + 1;
  }
  return out;
}

describe.each(SCENARIOS.map(s => [s] as const))('golden stream: %s', (name: string) => {
  const fixture = loadFixture(name);

  beforeEach(() => vi.useRealTimers());
  afterEach(() => vi.restoreAllMocks());

  it('replays through the production wiring to a stable structure', () => {
    const { store, consumed } = replay(fixture.events);
    // The pump stops at the FIRST close (production behavior). Whatever the
    // server appended after it is bus-only. Completed runs currently append a
    // duplicate close — asserted here so the backend fix shows up as a diff.
    const leftover = fixture.events.slice(consumed).map(e => e.type);
    expect({
      transcript: normalize(store.messages),
      restoredTranscript: normalize(mapServerMessages(fixture.conversation)),
      leftoverAfterClose: leftover,
    }).toMatchSnapshot();
  });

  it('leaves no message stuck streaming after the stream settles', () => {
    const { store } = replay(fixture.events);
    for (const m of store.messages) {
      if (m.type === 'delegation') {
        expect(['completed', 'failed', 'cancelled']).toContain(m.status);
      }
      if (m.type === 'tool_call') {
        expect(m.status).not.toBe('running');
      }
    }
  });

  it('delegation cards survive restore faithfully (live ⊆ restored)', () => {
    const { store } = replay(fixture.events);
    const restored = mapServerMessages(fixture.conversation);
    const family = new Set(['delegation', 'work_order_report']);
    const live = normalize(store.messages).filter(e => family.has(e.type as string));
    const budget = multiset(normalize(restored).filter(e => family.has(e.type as string)));
    for (const e of live) {
      const k = JSON.stringify(e);
      expect(budget[k], `live card missing/short in restored: ${k}`).toBeGreaterThan(0);
      budget[k] -= 1;
    }
  });
});

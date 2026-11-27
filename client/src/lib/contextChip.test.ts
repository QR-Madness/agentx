import { describe, it, expect } from 'vitest';
import { contextChipState, seedContextInfo, seedModelId } from './contextChip';
import type { AssistantMessage, ConversationMessage, UserMessage } from './messages';

describe('contextChipState', () => {
  it('hides only when the window is unknown/invalid', () => {
    expect(contextChipState(null)).toBeNull();
    expect(contextChipState(undefined)).toBeNull();
    expect(contextChipState({ window: 0, used: 0 })).toBeNull();
  });

  it('shows at all times when the window is known, including a fresh chat at 0%', () => {
    // Brand-new conversation: used defaults to 0, chip still appears.
    expect(contextChipState({ window: 128_000 })).toEqual({
      label: '0% context',
      warn: false,
      title: 'Context: 0 / 128,000 tokens',
    });
    // Low usage no longer hides it.
    const low = contextChipState({ window: 10_000, used: 900 });
    expect(low).toEqual({ label: '9% context', warn: false, title: 'Context: 900 / 10,000 tokens' });
  });

  it('shows a quiet percentage in the mid range', () => {
    const s = contextChipState({ window: 10_000, used: 6_200 });
    expect(s).toEqual({
      label: '62% context',
      warn: false,
      title: 'Context: 6,200 / 10,000 tokens',
    });
  });

  it('warns from 75% with the summarization hint', () => {
    const s = contextChipState({ window: 10_000, used: 8_000 });
    expect(s?.warn).toBe(true);
    expect(s?.label).toBe('80% context');
    expect(s?.title).toContain('older turns are summarized automatically');
  });

  it('appends compression telemetry when present', () => {
    const dropped = contextChipState({ window: 10_000, used: 9_000, droppedTurns: 3 });
    expect(dropped?.title).toContain('3 older turns compressed');
    const one = contextChipState({ window: 10_000, used: 9_000, droppedTurns: 1 });
    expect(one?.title).toContain('1 older turn compressed');
    const summarized = contextChipState({ window: 10_000, used: 9_000, summarized: true });
    expect(summarized?.title).toContain('summary active');
  });

  it('caps the label at 100%', () => {
    expect(contextChipState({ window: 1_000, used: 1_500 })?.label).toBe('100% context');
  });
});

// Seeding: the chip before a turn's `done` reports the server's window.

const GLOBAL_DEFAULT = 'vercel:google/gemini-3.1-flash-lite';
const CATALOG = [
  { id: GLOBAL_DEFAULT, context_window: 1_048_576 },
  { id: 'openrouter:acme/small-model', context_length: 32_768 },
];

const user = (content: string): UserMessage => ({
  id: `u-${content}`, timestamp: '2026-11-26T00:00:00Z', type: 'user', content,
});
const assistant = (content: string, extra: Partial<AssistantMessage> = {}): AssistantMessage => ({
  id: `a-${content}`, timestamp: '2026-11-26T00:00:01Z', type: 'assistant', content, ...extra,
});

describe('seedModelId', () => {
  it('resolves "Default model" to the Global Default Model', () => {
    // An unpinned profile and no conversation override: the composer passes ''.
    expect(seedModelId([], '', GLOBAL_DEFAULT)).toBe(GLOBAL_DEFAULT);
    expect(seedModelId([], undefined, GLOBAL_DEFAULT)).toBe(GLOBAL_DEFAULT);
  });

  it('prefers the effective model, and a recorded turn over both', () => {
    expect(seedModelId([], 'openrouter:acme/small-model', GLOBAL_DEFAULT)).toBe(
      'openrouter:acme/small-model',
    );
    const msgs: ConversationMessage[] = [user('hi'), assistant('hello', { model: 'turn-model' })];
    expect(seedModelId(msgs, 'openrouter:acme/small-model', GLOBAL_DEFAULT)).toBe('turn-model');
  });

  it('names no model when nothing is set anywhere', () => {
    expect(seedModelId([], '', null)).toBeUndefined();
    expect(seedModelId([user('hi')], undefined, undefined)).toBeUndefined();
  });
});

describe('seedContextInfo', () => {
  it('seeds a fresh "Default model" conversation at 0% of its real window', () => {
    const modelId = seedModelId([], '', GLOBAL_DEFAULT);
    const info = seedContextInfo({ messages: [], modelId, models: CATALOG });
    expect(info).toEqual({ window: 1_048_576, used: 0 });
    expect(contextChipState(info)).toEqual({
      label: '0% context',
      warn: false,
      title: 'Context: 0 / 1,048,576 tokens',
    });
  });

  it('seeds nothing for an unknown model: no chip, never a percentage of a guessed window', () => {
    const msgs: ConversationMessage[] = [user('x'.repeat(40_000))];
    const info = seedContextInfo({ messages: msgs, modelId: 'vercel:acme/not-listed', models: CATALOG });
    expect(info).toBeNull();
    expect(contextChipState(info)).toBeNull();
    // Nor when no model resolves at all (no Global Default configured).
    expect(seedContextInfo({ messages: msgs, modelId: undefined, models: CATALOG })).toBeNull();
  });

  it('lets a Model Limits override win over the catalog, and stand in for a missing entry', () => {
    const limits = { models: { [GLOBAL_DEFAULT]: { context_window: 200_000 } } };
    expect(seedContextInfo({ messages: [], modelId: GLOBAL_DEFAULT, models: CATALOG, limits })?.window).toBe(
      200_000,
    );
    const unlisted = { models: { 'vercel:acme/not-listed': { context_window: 64_000 } } };
    expect(
      seedContextInfo({ messages: [], modelId: 'vercel:acme/not-listed', models: CATALOG, limits: unlisted }),
    ).toEqual({ window: 64_000, used: 0 });
  });

  it('accepts a catalog that reports context_length instead of context_window', () => {
    expect(
      seedContextInfo({ messages: [], modelId: 'openrouter:acme/small-model', models: CATALOG })?.window,
    ).toBe(32_768);
  });

  it("measures used from the latest recorded turn, else estimates from the transcript", () => {
    const recorded: ConversationMessage[] = [
      user('hi'),
      assistant('first', { tokensInput: 900, tokensOutput: 100 }),
      user('again'),
      assistant('second', { tokensInput: 1_500, tokensOutput: 250 }),
    ];
    expect(seedContextInfo({ messages: recorded, modelId: GLOBAL_DEFAULT, models: CATALOG })?.used).toBe(1_750);

    const unrecorded: ConversationMessage[] = [user('a'.repeat(400)), assistant('b'.repeat(401))];
    expect(seedContextInfo({ messages: unrecorded, modelId: GLOBAL_DEFAULT, models: CATALOG })?.used).toBe(201);
  });
});

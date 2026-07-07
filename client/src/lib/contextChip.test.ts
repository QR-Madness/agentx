import { describe, it, expect } from 'vitest';
import { contextChipState } from './contextChip';

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

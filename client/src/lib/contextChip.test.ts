import { describe, it, expect } from 'vitest';
import { contextChipState } from './contextChip';

describe('contextChipState', () => {
  it('hides below 50% usage and on missing/invalid info', () => {
    expect(contextChipState(null)).toBeNull();
    expect(contextChipState(undefined)).toBeNull();
    expect(contextChipState({ window: 0, used: 0 })).toBeNull();
    expect(contextChipState({ window: 10_000, used: 4_900 })).toBeNull();
  });

  it('shows a quiet percentage from 50%', () => {
    const s = contextChipState({ window: 10_000, used: 6_200 });
    expect(s).toEqual({
      label: '62% ctx',
      warn: false,
      title: 'Context: 6,200 / 10,000 tokens',
    });
  });

  it('warns from 75% with the summarization hint', () => {
    const s = contextChipState({ window: 10_000, used: 8_000 });
    expect(s?.warn).toBe(true);
    expect(s?.label).toBe('80% ctx');
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
    expect(contextChipState({ window: 1_000, used: 1_500 })?.label).toBe('100% ctx');
  });
});

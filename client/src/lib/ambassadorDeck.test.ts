import { describe, it, expect } from 'vitest';
import { deckThreadId, DECK_STARTERS } from './ambassadorDeck';

describe('deckThreadId', () => {
  it('scopes by user id when one is present', () => {
    expect(deckThreadId(42)).toBe('deck:42');
    expect(deckThreadId('alice')).toBe('deck:alice');
  });

  it('falls back to a shared default with no user', () => {
    expect(deckThreadId()).toBe('deck:default');
    expect(deckThreadId(null)).toBe('deck:default');
    expect(deckThreadId(undefined)).toBe('deck:default');
  });

  it('keeps user id 0 distinct from "no user"', () => {
    expect(deckThreadId(0)).toBe('deck:0');
  });

  it('uses the deck: prefix so it can never collide with a real conversation id', () => {
    expect(deckThreadId(7).startsWith('deck:')).toBe(true);
    expect(deckThreadId(7)).toContain(':'); // hex-uuid conversation ids have no colon
  });
});

describe('DECK_STARTERS', () => {
  it('offers survey + roster oriented prompts', () => {
    expect(DECK_STARTERS.length).toBeGreaterThan(0);
    for (const s of DECK_STARTERS) {
      expect(s.label.trim()).not.toBe('');
      expect(s.prompt.trim()).not.toBe('');
    }
  });
});

import { describe, it, expect } from 'vitest';
import { deckThreadId, DECK_STARTERS, orderInquiries, inquiryLabel } from './ambassadorDeck';

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

describe('orderInquiries', () => {
  const DECK = 'deck:u';
  it('pins the home deck first, then newest-first by updated_at', () => {
    const threads = [
      { thread_id: 'inq:u:a', title: 'A', updated_at: '2026-06-20' },
      { thread_id: DECK, title: '', updated_at: '2026-06-10' },
      { thread_id: 'inq:u:b', title: 'B', updated_at: '2026-06-22' },
    ];
    expect(orderInquiries(threads, DECK).map((t) => t.thread_id)).toEqual([
      DECK, 'inq:u:b', 'inq:u:a',
    ]);
  });

  it('handles a missing home deck (deck not in list yet)', () => {
    const threads = [{ thread_id: 'inq:u:a', title: 'A', updated_at: '2026-06-20' }];
    expect(orderInquiries(threads, DECK).map((t) => t.thread_id)).toEqual(['inq:u:a']);
  });
});

describe('inquiryLabel', () => {
  const DECK = 'deck:u';
  it('uses the title when present', () => {
    expect(inquiryLabel({ thread_id: 'inq:u:a', title: 'Migration' }, DECK)).toBe('Migration');
  });
  it('falls back to Command Deck for the home thread, New Inquiry otherwise', () => {
    expect(inquiryLabel({ thread_id: DECK, title: '' }, DECK)).toBe('Command Deck');
    expect(inquiryLabel({ thread_id: 'inq:u:a', title: '  ' }, DECK)).toBe('New Inquiry');
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

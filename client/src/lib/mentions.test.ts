import { describe, it, expect } from 'vitest';
import {
  getActiveMention,
  applyMention,
  resolveMentionToken,
  extractMentionedAgentIds,
} from './mentions';
import type { AgentProfile } from './api/types';

const profiles = [
  { id: '1', name: 'Mobius', agentId: 'bright-grand-fern' },
  { id: '2', name: 'Code Reviewer', agentId: 'lucky-rising-thistle' },
] as AgentProfile[];

describe('getActiveMention', () => {
  it('detects the @token the caret is inside', () => {
    const text = 'hey @bri';
    expect(getActiveMention(text, text.length)).toEqual({ query: 'bri', start: 4, end: 8 });
  });

  it('matches an empty query right after @', () => {
    expect(getActiveMention('say @', 5)).toEqual({ query: '', start: 4, end: 5 });
  });

  it('returns null when the caret is past a space', () => {
    expect(getActiveMention('hey @bri now', 'hey @bri now'.length)).toBeNull();
  });

  it('ignores emails (no left boundary)', () => {
    const text = 'mail me@host';
    expect(getActiveMention(text, text.length)).toBeNull();
  });
});

describe('applyMention', () => {
  it('replaces the span with @slug and a trailing space', () => {
    const { text, caret } = applyMention('hey @bri', { start: 4, end: 8 }, 'bright-grand-fern');
    expect(text).toBe('hey @bright-grand-fern ');
    expect(caret).toBe(text.length);
  });

  it('keeps trailing text and reports the caret after the token', () => {
    const { text, caret } = applyMention('hi @x do it', { start: 3, end: 5 }, 'slug');
    expect(text).toBe('hi @slug  do it');
    expect(caret).toBe('hi @slug '.length);
  });
});

describe('resolveMentionToken', () => {
  it('resolves by agent_id then single-word name (case-insensitive)', () => {
    expect(resolveMentionToken('bright-grand-fern', profiles)).toBe('bright-grand-fern');
    expect(resolveMentionToken('mobius', profiles)).toBe('bright-grand-fern');
    expect(resolveMentionToken('nobody', profiles)).toBeNull();
  });
});

describe('extractMentionedAgentIds', () => {
  it('resolves, dedupes, and preserves order; ignores unmatched + emails', () => {
    const text = 'hi @Mobius and @bright-grand-fern again, not me@host or @ghost';
    expect(extractMentionedAgentIds(text, profiles)).toEqual(['bright-grand-fern']);
  });
});

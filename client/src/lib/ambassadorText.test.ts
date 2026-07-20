import { describe, it, expect } from 'vitest';
import { stripThinkBlocks } from './ambassadorText';

describe('stripThinkBlocks', () => {
  it('passes clean text through untouched', () => {
    expect(stripThinkBlocks('The gist: 42 units.')).toBe('The gist: 42 units.');
    expect(stripThinkBlocks('')).toBe('');
  });

  it('removes closed think blocks (both tag spellings)', () => {
    expect(stripThinkBlocks('<think>secret</think>answer')).toBe('answer');
    expect(stripThinkBlocks('<thinking>secret</thinking>answer')).toBe('answer');
    expect(stripThinkBlocks('a<think>x</think>b<think>y</think>c')).toBe('abc');
  });

  it('removes an unterminated trailing block', () => {
    expect(stripThinkBlocks('answer<think>still reasoning…')).toBe('answer');
  });

  it('degrades to tag removal when the whole reply was reasoning', () => {
    // A visible answer beats a blank one.
    expect(stripThinkBlocks('<think>only reasoning</think>')).toBe('only reasoning');
  });

  it('handles the live-caught leak shape (block before the real answer)', () => {
    const leaked = '<think>**Analyzing**\n\nplan {with braces}</think>Your agents have been busy.';
    expect(stripThinkBlocks(leaked)).toBe('Your agents have been busy.');
  });
});

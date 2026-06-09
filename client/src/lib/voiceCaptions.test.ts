import { describe, it, expect } from 'vitest';
import { appendCaption, type Caption } from './voiceCaptions';

describe('appendCaption', () => {
  it('appends a trimmed entry', () => {
    const out = appendCaption([], 'you', '  what did it find?  ');
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ role: 'you', text: 'what did it find?' });
    expect(out[0].id).toBeTruthy();
  });

  it('ignores blank / whitespace-only text', () => {
    expect(appendCaption([], 'ambassador', '')).toHaveLength(0);
    expect(appendCaption([], 'ambassador', '   \n ')).toHaveLength(0);
  });

  it('drops a consecutive identical same-role line (re-played briefing, double callback)', () => {
    let list: Caption[] = [];
    list = appendCaption(list, 'ambassador', 'It searched the county index.');
    list = appendCaption(list, 'ambassador', 'It searched the county index.');
    expect(list).toHaveLength(1);
  });

  it('keeps an identical line when the role differs', () => {
    let list: Caption[] = [];
    list = appendCaption(list, 'you', 'summarize this');
    list = appendCaption(list, 'ambassador', 'summarize this');
    expect(list).toHaveLength(2);
  });

  it('keeps a repeated line that is not consecutive', () => {
    let list: Caption[] = [];
    list = appendCaption(list, 'you', 'again');
    list = appendCaption(list, 'ambassador', 'okay');
    list = appendCaption(list, 'you', 'again');
    expect(list).toHaveLength(3);
  });

  it('does not mutate the input list', () => {
    const input: Caption[] = [];
    const out = appendCaption(input, 'you', 'hi');
    expect(input).toHaveLength(0);
    expect(out).toHaveLength(1);
  });
});

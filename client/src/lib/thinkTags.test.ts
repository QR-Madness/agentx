/**
 * Think-tag hygiene — the DOM-leak regressions (chat-ux backlog item).
 *
 * The live streaming buffer can be cut ANYWHERE: mid-thought (delegation-start
 * flushes, steers, max_tokens truncation) or mid-TAG (a `<think>` split across
 * SSE chunk boundaries). Every cut must strip clean — a surviving raw tag
 * reaches the markdown/DOM renderer as `The tag <think> is unrecognized`.
 *
 * The boundary sweep runs against the REAL recorded reasoning stream
 * (fixtures/streams/think-heavy, deepseek-r1): the full content is re-cut at
 * every position across its think-tag regions.
 */

import { describe, it, expect } from 'vitest';
import { extractThinking, stripThinkingTags } from './messages';

const rawFixtures = import.meta.glob('/src/test/fixtures/streams/*/events.jsonl', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

function fixtureContent(): string {
  const key = Object.keys(rawFixtures).find(k => k.includes('/think-heavy/'));
  const raw = (key && rawFixtures[key]) || '';
  let acc = '';
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    const parsed = JSON.parse(line);
    if (parsed._state) continue;
    const sse: string = parsed.sse;
    if (!sse.startsWith('event: chunk')) continue;
    const dataLine = sse.split('\n').find(l => l.startsWith('data: '));
    if (dataLine) acc += JSON.parse(dataLine.slice(6)).content ?? '';
  }
  return acc;
}

describe('stripThinkingTags (streaming)', () => {
  it('strips an UNCLOSED think block (mid-thought flush/truncation)', () => {
    const cut = 'Intro. <think>half a thought that never clo';
    expect(stripThinkingTags(cut, true)).toBe('Intro.');
  });

  it('strips a partial opener split across chunk boundaries', () => {
    for (const tail of ['<', '<t', '<thi', '<think', '<thinking', '[thin', '<internal_mono']) {
      expect(stripThinkingTags(`Answer so far ${tail}`, true)).toBe('Answer so far');
    }
  });

  it('keeps legitimate angle brackets and complete non-think tags', () => {
    expect(stripThinkingTags('proof: 5 < 6', true)).toBe('proof: 5 < 6');
    expect(stripThinkingTags('use `<b>` for bold', true)).toBe('use `<b>` for bold');
  });

  it('sweeps every cut position across the recorded R1 stream cleanly', () => {
    const content = fixtureContent();
    expect(content).toContain('<think');
    // Every region where a tag could be split: sweep a window around each
    // '<' that begins a think open/close tag.
    const tagStarts = [...content.matchAll(/<\/?think(?:ing)?>/gi)].map(m => m.index!);
    expect(tagStarts.length).toBeGreaterThan(0);
    for (const start of tagStarts) {
      for (let cut = start; cut <= Math.min(start + 12, content.length); cut++) {
        const visible = stripThinkingTags(content.slice(0, cut), true);
        expect(visible).not.toMatch(/<\/?think/i);
        expect(visible).not.toMatch(/<t?h?i?n?k?i?n?g?$/i);
      }
    }
    // And the settled full content strips to think-free text.
    expect(stripThinkingTags(content, true)).not.toMatch(/<\/?think/i);
  });
});

describe('extractThinking', () => {
  it('captures closed blocks', () => {
    expect(extractThinking('<think>alpha</think> answer')).toBe('alpha');
  });

  it('captures an unclosed trailing block (mid-thought flush keeps the thought)', () => {
    expect(extractThinking('so far <think>beta reasoning')).toBe('beta reasoning');
  });

  it('captures closed + unclosed together, in order', () => {
    expect(extractThinking('<think>one</think> mid <think>two'))
      .toBe('one\n\ntwo');
  });

  it('returns null when there is no thinking', () => {
    expect(extractThinking('plain answer, 5 < 6')).toBeNull();
  });

  it('round-trips the recorded R1 stream: thought preserved, content clean', () => {
    const content = fixtureContent();
    const thinking = extractThinking(content);
    expect(thinking).toBeTruthy();
    expect(stripThinkingTags(content, true)).not.toContain(thinking!.slice(0, 24));
  });
});

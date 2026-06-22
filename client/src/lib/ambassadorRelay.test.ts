import { describe, it, expect, vi } from 'vitest';
import { relayToActiveConversation } from './ambassadorRelay';

describe('relayToActiveConversation', () => {
  it('sends to the active tab and names where it landed', () => {
    const send = vi.fn().mockReturnValue(true);
    const res = relayToActiveConversation('use metric units', { id: 't1', title: 'Atlas' }, send);
    expect(send).toHaveBeenCalledWith('t1', 'use metric units');
    expect(res).toEqual({ ok: true, note: 'Sent to Atlas.' });
  });

  it('reports a fold when the target has a running turn', () => {
    const send = vi.fn().mockReturnValue(true);
    const res = relayToActiveConversation('hi', { id: 't1', title: 'Atlas', activeRun: { runId: 'r1' } }, send);
    expect(res).toEqual({ ok: true, note: 'Folded into the running turn.' });
  });

  it('falls back to a generic name when the tab has no title', () => {
    const send = vi.fn().mockReturnValue(true);
    const res = relayToActiveConversation('hi', { id: 't1' }, send);
    expect(res.note).toBe('Sent to the conversation.');
  });

  it('fails (no send) when there is no active conversation', () => {
    const send = vi.fn();
    const res = relayToActiveConversation('hi', null, send);
    expect(send).not.toHaveBeenCalled();
    expect(res).toEqual({ ok: false, note: 'Open a conversation to relay into.' });
  });

  it('surfaces a failure when the seam has no live handler', () => {
    const send = vi.fn().mockReturnValue(false);
    const res = relayToActiveConversation('hi', { id: 't1', title: 'Atlas' }, send);
    expect(res).toEqual({ ok: false, note: 'Could not send to the conversation.' });
  });

  it('ignores empty / whitespace-only text', () => {
    const send = vi.fn();
    const res = relayToActiveConversation('   ', { id: 't1' }, send);
    expect(send).not.toHaveBeenCalled();
    expect(res.ok).toBe(false);
  });

  it('trims the relayed text before sending', () => {
    const send = vi.fn().mockReturnValue(true);
    relayToActiveConversation('  hello  ', { id: 't1' }, send);
    expect(send).toHaveBeenCalledWith('t1', 'hello');
  });
});

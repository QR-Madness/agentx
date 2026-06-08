import { describe, it, expect, beforeEach, vi } from 'vitest';
import { speechPlayer } from './audio';

// jsdom implements neither media playback nor object URLs — stub both so the
// SpeechPlayer's caching / state logic can be exercised headlessly.
beforeEach(() => {
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  HTMLMediaElement.prototype.pause = vi.fn();
  let n = 0;
  globalThis.URL.createObjectURL = vi.fn(() => `blob:mock/${n++}`);
  globalThis.URL.revokeObjectURL = vi.fn();
  speechPlayer.stop();
});

const blob = () => new Blob(['x'], { type: 'audio/mpeg' });

describe('SpeechPlayer', () => {
  it('caches synthesized audio by id+text (replay never re-synthesizes)', async () => {
    const synth = vi.fn(async () => blob());
    await speechPlayer.speak('replay', 'hello there', synth);
    await speechPlayer.speak('replay', 'hello there', synth);
    expect(synth).toHaveBeenCalledTimes(1);
  });

  it('re-synthesizes when the text changes for the same id', async () => {
    const synth = vi.fn(async () => blob());
    await speechPlayer.speak('change', 'first text', synth);
    await speechPlayer.speak('change', 'second text', synth);
    expect(synth).toHaveBeenCalledTimes(2);
  });

  it('reflects playing + idle state in the snapshot', async () => {
    const synth = vi.fn(async () => blob());
    await speechPlayer.speak('state', 'speak me', synth);
    expect(speechPlayer.getSnapshot().playingId).toBe('state');
    speechPlayer.stop();
    expect(speechPlayer.getSnapshot().playingId).toBeNull();
    expect(speechPlayer.getSnapshot().loadingId).toBeNull();
  });

  it('passes an abort signal to the synthesizer', async () => {
    const synth = vi.fn(async (_t: string, signal: AbortSignal) => {
      expect(signal).toBeInstanceOf(AbortSignal);
      return blob();
    });
    await speechPlayer.speak('signal', 'with abort', synth);
    expect(synth).toHaveBeenCalledTimes(1);
  });
});

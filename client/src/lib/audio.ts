/**
 * SpeechPlayer — a framework-agnostic singleton that turns ambassador text into
 * spoken audio and plays it back (the "MP3 utility").
 *
 * Responsibilities:
 *  - Synthesize-on-demand via an injected `Synthesize` fn (kept API-agnostic so
 *    this file never imports the api layer), caching the resulting Blob URLs by
 *    text so replays + panel re-entry don't re-synthesize (TTS bills per char).
 *  - A playback **queue** over one shared `HTMLAudioElement` — a new utterance
 *    stops the prior one. The queue is single-item today but lets sentence-chunked
 *    synthesis drop in later (enqueue N urls, play them back-to-back).
 *  - **Autoplay unlock**: browsers block programmatic audio until a user gesture.
 *    `unlock()` (called on the voice-mode-enter click) blesses the shared element
 *    with a silent buffer so later auto-spoken briefings actually play.
 *  - Abort the in-flight synth on stop / switch, and a subscribe/snapshot surface
 *    for the React `useSpeech` hook (via useSyncExternalStore).
 */

/** Synthesize spoken audio for `text`. Injected by the hook so this stays api-agnostic. */
export type Synthesize = (text: string, signal: AbortSignal) => Promise<Blob>;

export interface SpeechSnapshot {
  /** The item id currently playing audio, if any. */
  playingId: string | null;
  /** The item id whose audio is being synthesized, if any. */
  loadingId: string | null;
}

const CACHE_LIMIT = 30;

/** A valid, empty (0-sample) WAV — instant silence used to unlock autoplay. */
function silentAudioDataUri(): string {
  const bytes = new Uint8Array([
    0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
  ]);
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return `data:audio/wav;base64,${btoa(bin)}`;
}

/** Tiny, stable string hash (djb2) for cache keys. */
function hashText(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

class SpeechPlayer {
  private audio: HTMLAudioElement | null = null;
  private cache = new Map<string, string>(); // cacheKey -> object URL
  private queue: string[] = []; // object URLs pending playback
  private playingId: string | null = null;
  private loadingId: string | null = null;
  private abort: AbortController | null = null;
  private unlocked = false;

  private listeners = new Set<() => void>();
  private snapshot: SpeechSnapshot = { playingId: null, loadingId: null };

  // ── External-store surface (for useSyncExternalStore) ─────────────────────
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): SpeechSnapshot => this.snapshot;

  private emit(): void {
    this.snapshot = { playingId: this.playingId, loadingId: this.loadingId };
    for (const l of this.listeners) l();
  }

  private ensureAudio(): HTMLAudioElement {
    if (!this.audio) {
      const a = new Audio();
      a.addEventListener('ended', () => this.advance());
      a.addEventListener('error', () => this.advance());
      this.audio = a;
    }
    return this.audio;
  }

  /** Bless the shared element for later programmatic playback. Call on a user gesture. */
  unlock(): void {
    if (this.unlocked) return;
    const a = this.ensureAudio();
    const prevSrc = a.src;
    a.src = silentAudioDataUri();
    a.muted = true;
    a.play()
      .then(() => {
        a.pause();
        a.muted = false;
        a.currentTime = 0;
        if (prevSrc) a.src = prevSrc;
        this.unlocked = true;
      })
      .catch(() => {
        a.muted = false; // best-effort; on-demand (gesture-driven) playback still works
      });
  }

  /** Synthesize (or reuse cached) audio for `text` and play it under `id`. */
  async speak(id: string, text: string, synthesize: Synthesize): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) return;
    const key = `${id}:${hashText(trimmed)}`;

    let url = this.cache.get(key);
    if (!url) {
      // New synthesis — cancel any prior in-flight one and show loading.
      this.abort?.abort();
      this.abort = new AbortController();
      this.loadingId = id;
      this.emit();
      try {
        const blob = await synthesize(trimmed, this.abort.signal);
        url = URL.createObjectURL(blob);
        this.cacheSet(key, url);
      } catch (err) {
        if (this.loadingId === id) {
          this.loadingId = null;
          this.emit();
        }
        throw err;
      }
      if (this.loadingId === id) this.loadingId = null;
    }

    this.play(id, [url]);
  }

  /** Replace the queue and start playing `urls` back-to-back under `id`. */
  private play(id: string, urls: string[]): void {
    const a = this.ensureAudio();
    this.queue = urls.slice(1);
    this.playingId = id;
    a.src = urls[0];
    void a.play().catch(() => this.advance());
    this.emit();
  }

  /** Advance to the next queued url, or settle to idle when the queue drains. */
  private advance(): void {
    const a = this.ensureAudio();
    const next = this.queue.shift();
    if (next) {
      a.src = next;
      void a.play().catch(() => this.advance());
      return;
    }
    if (this.playingId !== null) {
      this.playingId = null;
      this.emit();
    }
  }

  /** Stop playback + cancel any in-flight synthesis. Cached audio is kept for replay. */
  stop(): void {
    this.abort?.abort();
    this.abort = null;
    this.queue = [];
    if (this.audio) {
      this.audio.pause();
      this.audio.currentTime = 0;
    }
    const changed = this.playingId !== null || this.loadingId !== null;
    this.playingId = null;
    this.loadingId = null;
    if (changed) this.emit();
  }

  private cacheSet(key: string, url: string): void {
    this.cache.set(key, url);
    while (this.cache.size > CACHE_LIMIT) {
      const oldest = this.cache.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      const stale = this.cache.get(oldest);
      this.cache.delete(oldest);
      if (stale) URL.revokeObjectURL(stale);
    }
  }
}

/** Process-wide singleton — one shared audio element across the app. */
export const speechPlayer = new SpeechPlayer();

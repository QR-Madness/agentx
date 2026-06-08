/**
 * audioRecorder — mic capture for ambassador voice input (the STT/push-to-talk
 * half of voice mode). Captures **raw PCM** via the Web Audio API and encodes a
 * WAV client-side, rather than using `MediaRecorder`.
 *
 * Why not MediaRecorder: WebKitGTK's MediaRecorder backend (the Linux desktop
 * webview) frequently fails to finalize its GStreamer encode pipeline and yields
 * an empty clip (the `GstAppSrc … automatic-eos` warning). Web Audio + a hand-
 * rolled WAV encoder behaves identically across webkit2gtk, WKWebView, Chromium,
 * and browser dev mode, and OpenRouter's `/audio/transcriptions` accepts `wav`.
 *
 * Kept framework-agnostic (no React, no api layer). Transcription itself lives in
 * `api.transcribe`; this only captures.
 */

export type RecorderState = 'idle' | 'recording';

export interface RecordingResult {
  blob: Blob;
  /** OpenRouter `/audio/transcriptions` format token. */
  format: string;
}

/** A typed mic/recording failure the UI can message cleanly. */
export class RecordingError extends Error {
  constructor(
    message: string,
    /** Stable code: `unsupported` | `denied` | `no_device` | `failed`. */
    readonly code: 'unsupported' | 'denied' | 'no_device' | 'failed',
  ) {
    super(message);
    this.name = 'RecordingError';
  }
}

type AudioCtxCtor = typeof AudioContext;

function getAudioContextCtor(): AudioCtxCtor | undefined {
  if (typeof window === 'undefined') return undefined;
  return window.AudioContext ?? (window as unknown as { webkitAudioContext?: AudioCtxCtor }).webkitAudioContext;
}

/** Whether mic capture is possible here (secure context + Web Audio + getUserMedia). */
export function isRecordingSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    !!getAudioContextCtor()
  );
}

/** Encode mono Float32 PCM samples as a 16-bit PCM WAV Blob. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * blockAlign)
  view.setUint16(32, 2, true); // block align (1ch * 16-bit)
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: 'audio/wav' });
}

export class AudioRecorder {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private chunks: Float32Array[] = [];
  private sampleRate = 48000;
  private recording = false;

  get state(): RecorderState {
    return this.recording ? 'recording' : 'idle';
  }

  /** Request the mic and begin capturing PCM. Throws a `RecordingError` on failure. */
  async start(): Promise<void> {
    if (this.recording) return;
    const AudioCtx = getAudioContextCtor();
    if (!isRecordingSupported() || !AudioCtx) {
      throw new RecordingError('Voice input is not supported here.', 'unsupported');
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const name = (err as DOMException)?.name;
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        throw new RecordingError('Microphone access was denied.', 'denied');
      }
      if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        throw new RecordingError('No microphone was found.', 'no_device');
      }
      throw new RecordingError('Could not start recording.', 'failed');
    }
    this.stream = stream;
    const ctx = new AudioCtx();
    // AudioContexts can start suspended until a gesture; start() is gesture-driven.
    try {
      await ctx.resume();
    } catch {
      /* already running */
    }
    this.ctx = ctx;
    this.sampleRate = ctx.sampleRate;
    this.chunks = [];
    this.source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      // Copy — the event buffer is reused across callbacks.
      this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    // Route through a muted sink so onaudioprocess fires without echoing the mic.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    this.source.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);
    this.processor = processor;
    this.sink = sink;
    this.recording = true;
  }

  /** Stop capturing and resolve the recorded audio as a WAV (or `null` if empty). */
  async stop(): Promise<RecordingResult | null> {
    if (!this.recording) return null;
    this.recording = false;
    const chunks = this.chunks;
    const sampleRate = this.sampleRate;
    this.disconnect();
    await this.closeContext();
    this.teardown();

    const length = chunks.reduce((n, c) => n + c.length, 0);
    if (length === 0) return null;
    const samples = new Float32Array(length);
    let offset = 0;
    for (const c of chunks) {
      samples.set(c, offset);
      offset += c.length;
    }
    return { blob: encodeWav(samples, sampleRate), format: 'wav' };
  }

  /** Abandon an in-progress capture and release the mic (retake / bail). */
  cancel(): void {
    this.recording = false;
    this.disconnect();
    void this.closeContext();
    this.teardown();
  }

  private disconnect(): void {
    try {
      this.processor?.disconnect();
      this.source?.disconnect();
      this.sink?.disconnect();
    } catch {
      /* ignore */
    }
    if (this.processor) this.processor.onaudioprocess = null;
  }

  private async closeContext(): Promise<void> {
    try {
      if (this.ctx && this.ctx.state !== 'closed') await this.ctx.close();
    } catch {
      /* ignore */
    }
  }

  private teardown(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.ctx = null;
    this.source = null;
    this.processor = null;
    this.sink = null;
    this.chunks = [];
  }
}

/** Encode an audio Blob to base64 (no data-URI prefix) for the transcribe endpoint. */
export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const chunk = 0x8000; // avoid String.fromCharCode arg-count limits
  for (let i = 0; i < buf.length; i += chunk) {
    binary += String.fromCharCode(...buf.subarray(i, i + chunk));
  }
  return btoa(binary);
}

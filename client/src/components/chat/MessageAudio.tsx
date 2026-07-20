/**
 * MessageAudio — render the audio clips attached to a user turn (audio input).
 *
 * Each `ChatAudioRef` points at a served blob (`…/documents/{doc}/raw`) fetched through
 * the authed client and object-URL'd (`lib/mediaImage.ts::resolveMediaBlob` — the same
 * path images/avatars use; a raw `<audio src>` can't carry auth). Used on the user
 * bubble (live + restored) and the composer preview strip. A cached STT `transcript`
 * (from the server's capability fallback) folds out under the chip.
 */

import { useEffect, useRef, useState } from 'react';
import { AudioLines, Pause, Play, X } from 'lucide-react';
import { resolveMediaImage } from '../../lib/mediaImage';
import type { ChatAudioRef } from '../../lib/api/types';

/** The served raw-blob path for an audio ref. */
export function audioRefPath(ref: ChatAudioRef): string {
  return `/api/workspaces/${encodeURIComponent(ref.workspace_id)}/documents/${encodeURIComponent(ref.doc_id)}/raw`;
}

function fmtTime(secs: number): string {
  if (!Number.isFinite(secs) || secs <= 0) return '';
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function AudioChip({ audioRef, onRemove }: { audioRef: ChatAudioRef; onRemove?: () => void }) {
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);
  const elRef = useRef<HTMLAudioElement | null>(null);
  const path = audioRefPath(audioRef);

  useEffect(() => {
    let alive = true;
    const el = new Audio();
    elRef.current = el;
    el.onended = () => alive && setPlaying(false);
    el.onloadedmetadata = () => alive && setDuration(el.duration);
    resolveMediaImage(path)
      .then((u) => {
        if (!alive) return;
        el.src = u;
      })
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
      el.pause();
      el.src = '';
      elRef.current = null;
    };
  }, [path]);

  const toggle = () => {
    const el = elRef.current;
    if (!el || failed) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      void el.play().then(() => setPlaying(true)).catch(() => setFailed(true));
    }
  };

  return (
    <div className="inline-flex flex-col gap-1 max-w-full">
      <div className="inline-flex items-center gap-1.5 rounded-pill border border-line bg-surface-raised px-2.5 py-1.5 text-xs text-fg-secondary">
        <button
          type="button"
          onClick={toggle}
          disabled={failed}
          className="inline-flex items-center justify-center bg-transparent text-accent disabled:text-fg-muted"
          title={failed ? "Couldn't load the audio" : playing ? 'Pause' : 'Play'}
          aria-label={playing ? 'Pause audio attachment' : 'Play audio attachment'}
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <AudioLines size={13} className={playing ? 'text-accent animate-pulse' : 'text-fg-muted'} />
        <span>{failed ? 'Audio unavailable' : 'Audio'}</span>
        {duration !== null && <span className="font-mono text-2xs text-fg-muted">{fmtTime(duration)}</span>}
        {audioRef.transcript && (
          <button
            type="button"
            onClick={() => setShowTranscript((v) => !v)}
            className="bg-transparent text-2xs uppercase tracking-caps text-fg-muted hover:text-fg-secondary"
            title="Show the transcript sent to the model"
          >
            transcript
          </button>
        )}
        {onRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="inline-flex items-center justify-center bg-transparent text-fg-muted hover:text-error"
            title="Remove audio"
            aria-label="Remove audio attachment"
          >
            <X size={12} />
          </button>
        )}
      </div>
      {showTranscript && audioRef.transcript && (
        <div className="max-w-96 rounded-md border border-line-subtle bg-surface-sunken px-2 py-1.5 text-2xs text-fg-muted whitespace-pre-wrap">
          {audioRef.transcript}
        </div>
      )}
    </div>
  );
}

export function MessageAudio({
  audio,
  onRemove,
}: {
  audio: ChatAudioRef[];
  onRemove?: (index: number) => void;
}) {
  if (!audio.length) return null;
  return (
    <div className="flex flex-wrap items-start gap-1.5">
      {audio.map((ref, i) => (
        <AudioChip
          key={`${ref.doc_id}-${i}`}
          audioRef={ref}
          onRemove={onRemove ? () => onRemove(i) : undefined}
        />
      ))}
    </div>
  );
}

/**
 * VoiceSurface — the immersive, Discord-call-style voice surface (the "Voice" tab
 * of an opted-in ambassador). Minimal by design: a push-to-talk control, captions,
 * and a settings popover.
 *
 * Flow: hold-to-talk (or tap-to-toggle) captures your voice via `useDictation`;
 * the transcript goes to `api.voiceCommand`, where the ambassador **infers intent**
 * — a question it answers aloud (and persists as Q&A), or an instruction it drafts
 * as a **relay** you review and send into the conversation. Barge-in: starting to
 * talk cuts off the ambassador. The transcript echoes instantly + a "thinking…"
 * state masks routing latency. Nothing is ever auto-sent.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Radio, Mic, Loader2, Settings2, Send, RotateCcw, CornerUpRight, X } from 'lucide-react';
import { useSpeech } from '../../hooks/useSpeech';
import { useDictation } from '../../hooks/useDictation';
import { api } from '../../lib/api';
import type { AmbassadorActiveConversation, AmbassadorTurnArtifacts } from '../../lib/api';
import { appendCaption, type Caption } from '../../lib/voiceCaptions';

type PttMode = 'hold' | 'toggle';

const PTT_KEY = 'agentx:voice:pttMode';
const CAPTIONS_KEY = 'agentx:voice:captions';
const TOGGLE_MAX_MS = 60_000; // safety: never leave the mic open forever in toggle mode

function readPref<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T) || fallback;
  } catch {
    return fallback;
  }
}

interface VoiceSurfaceProps {
  conversationId: string;
  /** Ambassador profile id — supplies the speech/STT models. */
  agentProfileId?: string;
  /** The watched conversation's agent name — grounds voice commands. */
  agentName: string;
  /** The ambassador's own display name — for its status ("Echo is thinking…"). */
  ambassadorName?: string;
  /** Where the person currently is (ambient context, distinct from the focus). */
  activeConversation?: AmbassadorActiveConversation;
  /** Latest-turn substance, to ground a voice command. */
  artifacts?: AmbassadorTurnArtifacts;
  /** Text the panel is currently speaking (e.g. an auto-played briefing). */
  ambientSpokenText: string;
  /** Relay a confirmed draft into the conversation. Returns false if not sendable. */
  onRelay: (text: string) => boolean;
  /** Called after an answer is persisted, so the Text tab can refresh. */
  onAnswerPersisted: () => void;
}

/** The ambassador mark — a pulsing accent disc. */
function Orb({ active }: { active: boolean }) {
  return (
    <div className="relative flex h-28 w-28 items-center justify-center">
      {active && (
        <>
          <span className="absolute h-28 w-28 animate-ping rounded-full bg-accent/40 opacity-60 motion-reduce:hidden" />
          <span className="absolute h-24 w-24 animate-pulse rounded-full bg-accent/25 motion-reduce:animate-none" />
        </>
      )}
      <span className="relative flex h-20 w-20 items-center justify-center rounded-full bg-accent/20">
        <Radio size={34} className="text-accent" />
      </span>
    </div>
  );
}

export function VoiceSurface({
  conversationId,
  agentProfileId,
  agentName,
  ambassadorName,
  activeConversation,
  artifacts,
  ambientSpokenText,
  onRelay,
  onAnswerPersisted,
}: VoiceSurfaceProps) {
  const speech = useSpeech({ agentProfileId });
  const [pttMode, setPttMode] = useState<PttMode>(() => readPref<PttMode>(PTT_KEY, 'hold'));
  const [captionsOn, setCaptionsOn] = useState<boolean>(
    () => readPref<string>(CAPTIONS_KEY, 'on') !== 'off',
  );
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [routing, setRouting] = useState(false);
  // The running caption log (both sides), an accumulating, scrollable transcript.
  const [captionLog, setCaptionLog] = useState<Caption[]>([]);
  // Whether an answer just landed (gates the "relay that instead" override).
  const [hasAnswer, setHasAnswer] = useState(false);
  const [relayDraft, setRelayDraft] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastTranscript = useRef('');

  const addCaption = useCallback((role: Caption['role'], text: string) => {
    setCaptionLog((prev) => appendCaption(prev, role, text));
  }, []);

  // Route a captured transcript through the ambassador's intent inference.
  const handleTranscript = useCallback(
    async (transcript: string) => {
      lastTranscript.current = transcript;
      addCaption('you', transcript);
      setError(null);
      setRelayDraft(null);
      setRouting(true);
      try {
        const res = await api.voiceCommand({
          conversation_id: conversationId,
          transcript,
          agent_name: agentName,
          artifacts,
          active_conversation: activeConversation,
        });
        if (res.action === 'relay') {
          setRelayDraft(res.text);
        } else if (res.text) {
          addCaption('ambassador', res.text);
          setHasAnswer(true);
          const speakId = res.qa_id ? `qa:${res.qa_id}` : `vc:${Date.now()}`;
          void speech.speak(speakId, res.text);
          onAnswerPersisted();
        }
      } catch {
        setError('The ambassador could not process that.');
      } finally {
        setRouting(false);
      }
    },
    [conversationId, agentName, activeConversation, artifacts, speech, onAnswerPersisted, addCaption],
  );

  const dictation = useDictation({ agentProfileId, onTranscript: handleTranscript });
  const { start: startCapture, stopAndTranscribe, cancel: cancelCapture, recording } = dictation;

  // Begin talking: unlock audio (gesture) + barge-in (cut off the ambassador).
  const beginTalk = useCallback(() => {
    speech.unlock();
    speech.stop();
    setHasAnswer(false); // a fresh turn — retire the prior answer's override affordance
    void startCapture();
  }, [speech, startCapture]);

  const endTalk = useCallback(() => void stopAndTranscribe(), [stopAndTranscribe]);

  // Tap-to-toggle.
  const toggleTalk = useCallback(() => {
    if (recording) void stopAndTranscribe();
    else beginTalk();
  }, [recording, stopAndTranscribe, beginTalk]);

  // Toggle-mode safety: auto-stop a runaway open mic.
  useEffect(() => {
    if (pttMode !== 'toggle' || !recording) return;
    const t = window.setTimeout(() => void stopAndTranscribe(), TOGGLE_MAX_MS);
    return () => window.clearTimeout(t);
  }, [pttMode, recording, stopAndTranscribe]);

  // Stop capture if the surface unmounts (tab switch / conversation change).
  useEffect(() => () => cancelCapture(), [cancelCapture]);

  // --- Captions: an accumulating, auto-scrolling, scroll-back transcript ----
  const captionScrollRef = useRef<HTMLDivElement>(null);
  // Pinned to the bottom? Only auto-scroll when the reader is at the latest line,
  // so scrolling back to review history is never yanked away.
  const pinnedRef = useRef(true);
  // The last auto-played briefing text folded into the log (so re-renders of the
  // same spoken line don't re-append it).
  const lastAmbientRef = useRef('');

  // Fold the panel's auto-played briefing (spoken via the Text/Voice autoplay) into
  // the caption log, once per distinct line.
  useEffect(() => {
    const t = ambientSpokenText.trim();
    if (t && t !== lastAmbientRef.current) {
      lastAmbientRef.current = t;
      addCaption('ambassador', t);
    }
  }, [ambientSpokenText, addCaption]);

  // A new conversation is a clean slate — drop the prior call's captions/state.
  useEffect(() => {
    setCaptionLog([]);
    setHasAnswer(false);
    setRelayDraft(null);
    setError(null);
    lastTranscript.current = '';
    lastAmbientRef.current = '';
    pinnedRef.current = true;
  }, [conversationId]);

  // Auto-scroll to the newest caption, but only while pinned to the bottom.
  useEffect(() => {
    const el = captionScrollRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [captionLog]);

  const onCaptionScroll = useCallback(() => {
    const el = captionScrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);

  // Keyboard: Space is push-to-talk (hold or toggle), ignored while typing.
  useEffect(() => {
    const isTyping = () => {
      const el = document.activeElement;
      return (
        el instanceof HTMLElement &&
        (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT' || el.isContentEditable)
      );
    };
    const down = (e: globalThis.KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat || isTyping()) return;
      e.preventDefault();
      if (pttMode === 'toggle') toggleTalk();
      else beginTalk();
    };
    const up = (e: globalThis.KeyboardEvent) => {
      if (e.code !== 'Space' || isTyping()) return;
      e.preventDefault();
      if (pttMode === 'hold') endTalk();
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [pttMode, beginTalk, endTalk, toggleTalk]);

  const setMode = (mode: PttMode) => {
    setPttMode(mode);
    try {
      localStorage.setItem(PTT_KEY, mode);
    } catch {
      /* ignore */
    }
  };
  const setCaptions = (on: boolean) => {
    setCaptionsOn(on);
    try {
      localStorage.setItem(CAPTIONS_KEY, on ? 'on' : 'off');
    } catch {
      /* ignore */
    }
  };

  const transcribing = dictation.transcribing;
  const speaking = speech.playingId !== null;
  const status = recording
    ? 'Listening…'
    : transcribing
      ? 'Transcribing…'
      : routing
        ? `${ambassadorName || 'The ambassador'} is thinking…`
        : speaking
          ? 'Speaking…'
          : relayDraft !== null
            ? 'Review & send to the agent'
            : 'Ready';

  const orbActive = recording || speaking || routing;

  const sendRelay = () => {
    const text = (relayDraft ?? '').trim();
    if (!text) return;
    if (onRelay(text)) setRelayDraft(null);
  };
  const retake = () => {
    setRelayDraft(null);
    beginTalk();
  };

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden">
      {/* Settings */}
      <div className="flex items-center justify-end px-4 py-2">
        <button
          type="button"
          onClick={() => setSettingsOpen((v) => !v)}
          data-on={settingsOpen || undefined}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg data-[on=true]:text-accent"
          aria-label="Voice settings"
          title="Voice settings"
        >
          <Settings2 size={16} />
        </button>
        {settingsOpen && (
          <div className="absolute right-3 top-11 z-10 w-56 max-w-[calc(100vw-1.5rem)] rounded-lg border border-line bg-surface-overlay p-3 shadow-lg">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-fg-muted">
              Push-to-talk
            </p>
            <div className="mb-3 grid grid-cols-2 gap-1 rounded-md bg-surface-sunken p-1 text-xs">
              {(['hold', 'toggle'] as PttMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  data-on={pttMode === m || undefined}
                  className="rounded px-2 py-1 font-medium capitalize text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-fg data-[on=true]:shadow-sm"
                >
                  {m === 'hold' ? 'Hold' : 'Toggle'}
                </button>
              ))}
            </div>
            <label className="flex items-center justify-between text-xs text-fg-secondary">
              Captions
              <input
                type="checkbox"
                checked={captionsOn}
                onChange={(e) => setCaptions(e.target.checked)}
              />
            </label>
          </div>
        )}
      </div>

      {/* Hero: orb + status + captions */}
      <div className="flex flex-1 flex-col items-center justify-center gap-5 px-6 text-center">
        <Orb active={orbActive} />
        <span className="text-sm font-medium text-fg">{status}</span>

        {captionsOn && captionLog.length > 0 && (
          <div
            ref={captionScrollRef}
            onScroll={onCaptionScroll}
            className="flex max-h-44 w-full max-w-sm flex-col gap-2 overflow-y-auto px-0.5"
          >
            {captionLog.map((c) =>
              c.role === 'you' ? (
                <p
                  key={c.id}
                  className="max-w-[85%] self-end whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent px-3 py-1.5 text-left text-sm text-fg-inverse"
                >
                  {c.text}
                </p>
              ) : (
                <p
                  key={c.id}
                  className="max-w-[90%] self-start whitespace-pre-wrap break-words text-left text-sm leading-relaxed text-fg-secondary"
                >
                  {c.text}
                </p>
              ),
            )}
          </div>
        )}
        {error && <p className="text-xs text-error">{error}</p>}
      </div>

      {/* Footer: relay confirm strip, or the PTT control */}
      <div className="border-t border-line p-4">
        {relayDraft !== null ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-accent">
              <CornerUpRight size={13} /> Draft to {agentName || 'the agent'}
            </div>
            <textarea
              value={relayDraft}
              onChange={(e) => setRelayDraft(e.target.value)}
              rows={2}
              className="w-full resize-none rounded-lg border border-line bg-surface-raised px-3 py-2 text-sm text-fg outline-none focus:border-line-strong max-[600px]:text-base"
            />
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setRelayDraft(null)}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2 py-1.5 text-xs text-fg-muted transition-colors hover:text-fg"
                title="Discard"
              >
                <X size={13} /> discard
              </button>
              <button
                type="button"
                onClick={retake}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2 py-1.5 text-xs text-fg-muted transition-colors hover:text-fg"
                title="Re-record"
              >
                <RotateCcw size={13} /> retake
              </button>
              <button
                type="button"
                onClick={sendRelay}
                disabled={!relayDraft.trim()}
                className="ml-auto inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-fg-inverse transition-colors hover:bg-accent-secondary disabled:opacity-40"
              >
                <Send size={13} /> Send to agent
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <button
              type="button"
              disabled={!dictation.supported || transcribing}
              onPointerDown={(e) => {
                if (pttMode === 'hold') {
                  e.preventDefault();
                  beginTalk();
                }
              }}
              onPointerUp={() => pttMode === 'hold' && endTalk()}
              onPointerLeave={() => pttMode === 'hold' && recording && endTalk()}
              onClick={() => pttMode === 'toggle' && toggleTalk()}
              data-active={recording || undefined}
              className="inline-flex h-16 w-16 items-center justify-center rounded-full border border-line bg-surface-raised text-fg-muted transition-colors hover:border-accent hover:text-accent data-[active=true]:border-accent data-[active=true]:bg-accent data-[active=true]:text-fg-inverse disabled:cursor-not-allowed disabled:opacity-40"
              aria-label={pttMode === 'hold' ? 'Hold to talk' : 'Tap to talk'}
              title={dictation.supported ? (pttMode === 'hold' ? 'Hold to talk' : 'Tap to talk') : 'Voice input is not supported here'}
            >
              {transcribing ? <Loader2 size={22} className="animate-spin" /> : <Mic size={24} />}
            </button>
            <span className="text-center text-[11px] leading-snug text-fg-muted">
              {dictation.error ? (
                <span className="text-error">{dictation.error}</span>
              ) : !dictation.supported ? (
                'Voice input isn’t available here.'
              ) : pttMode === 'hold' ? (
                <>
                  Hold the mic or <kbd className="rounded bg-surface-sunken px-1 text-fg-secondary">Space</kbd> to talk.
                </>
              ) : (
                <>Tap the mic (or <kbd className="rounded bg-surface-sunken px-1 text-fg-secondary">Space</kbd>) to start and stop.</>
              )}
            </span>
            {hasAnswer && (
              <button
                type="button"
                onClick={() => setRelayDraft(lastTranscript.current)}
                className="text-[11px] text-fg-muted underline-offset-2 transition-colors hover:text-accent hover:underline"
                title="Send what you said to the agent instead"
              >
                ↦ relay that to the agent instead
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

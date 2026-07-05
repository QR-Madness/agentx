/**
 * VoiceBar — the push-to-talk footer for an ambassador in voice mode.
 *
 * The Inquiry stream above **is** the transcript (a spoken question persists as a real
 * Q&A entry in the thread), so this is just the *input*: hold/tap to talk, and the
 * ambassador infers intent — a question it answers aloud (and into the Inquiry), or an
 * instruction it drafts as a **relay** you review and send. Barge-in cuts it off; nothing
 * is ever auto-sent. Voice and text share the same body — only this bar differs.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, Loader2, Settings2, Send, RotateCcw, CornerUpRight, X } from 'lucide-react';
import { useSpeech } from '../../hooks/useSpeech';
import { useDictation } from '../../hooks/useDictation';
import { api } from '../../lib/api';
import type { AmbassadorActiveConversation } from '../../lib/api';

type PttMode = 'hold' | 'toggle';

const PTT_KEY = 'agentx:voice:pttMode';
const TOGGLE_MAX_MS = 60_000; // safety: never leave the mic open forever in toggle mode

function readPref<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T) || fallback;
  } catch {
    return fallback;
  }
}

interface VoiceBarProps {
  conversationId: string;
  /** Ambassador profile id — supplies the speech/STT models. */
  agentProfileId?: string;
  /** The watched conversation's agent name — grounds voice commands. */
  agentName: string;
  /** The ambassador's own display name — for its status ("Echo is thinking…"). */
  ambassadorName?: string;
  /** Where the person currently is (ambient context, distinct from the focus). */
  activeConversation?: AmbassadorActiveConversation;
  /** Relay a confirmed draft into the conversation (may go to the server when off-tab).
   *  Returns where it landed (or why not). */
  onRelay: (text: string) => { ok: boolean; note: string } | Promise<{ ok: boolean; note: string }>;
  /** Called after an answer is persisted, so the Inquiry stream refreshes it in. */
  onAnswerPersisted: () => void;
}

export function VoiceBar({
  conversationId,
  agentProfileId,
  agentName,
  ambassadorName,
  activeConversation,
  onRelay,
  onAnswerPersisted,
}: VoiceBarProps) {
  const speech = useSpeech({ agentProfileId });
  const [pttMode, setPttMode] = useState<PttMode>(() => readPref<PttMode>(PTT_KEY, 'hold'));
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [routing, setRouting] = useState(false);
  // Whether an answer just landed (gates the "relay that instead" override).
  const [hasAnswer, setHasAnswer] = useState(false);
  const [relayDraft, setRelayDraft] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Transient confirmation after a successful relay ("Sent to …") — so voice gets the
  // same closure the text composer's flash gives.
  const [sent, setSent] = useState<string | null>(null);
  const lastTranscript = useRef('');

  // Route a captured transcript through the ambassador's intent inference. The answer
  // (or the question) lands in the shared Inquiry stream — no separate caption log.
  const handleTranscript = useCallback(
    async (transcript: string) => {
      lastTranscript.current = transcript;
      setError(null);
      setRelayDraft(null);
      setRouting(true);
      try {
        const res = await api.voiceCommand({
          conversation_id: conversationId,
          transcript,
          agent_name: agentName,
          active_conversation: activeConversation,
        });
        if (res.action === 'relay') {
          setRelayDraft(res.text);
        } else if (res.text) {
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
    [conversationId, agentName, activeConversation, speech, onAnswerPersisted],
  );

  const dictation = useDictation({ agentProfileId, onTranscript: handleTranscript });
  const { start: startCapture, stopAndTranscribe, cancel: cancelCapture, recording } = dictation;

  // Begin talking: unlock audio (gesture) + barge-in (cut off the ambassador).
  const beginTalk = useCallback(() => {
    speech.unlock();
    speech.stop();
    setHasAnswer(false); // a fresh turn — retire the prior answer's override affordance
    setSent(null); // and the prior relay confirmation
    void startCapture();
  }, [speech, startCapture]);

  const endTalk = useCallback(() => void stopAndTranscribe(), [stopAndTranscribe]);

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

  // Stop capture if the bar unmounts (tab switch / conversation change).
  useEffect(() => () => cancelCapture(), [cancelCapture]);

  // A new conversation is a clean slate for the transient bar state.
  useEffect(() => {
    setHasAnswer(false);
    setRelayDraft(null);
    setError(null);
    setSent(null);
    lastTranscript.current = '';
  }, [conversationId]);

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

  const transcribing = dictation.transcribing;
  const speaking = speech.playingId !== null;
  const busy = recording || transcribing || routing || speaking;
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
            : sent
              ? sent
              : 'Hold to talk';

  const sendRelay = async () => {
    const text = (relayDraft ?? '').trim();
    if (!text) return;
    const res = await onRelay(text);
    if (res.ok) {
      setRelayDraft(null);
      setError(null);
      setSent(res.note);
      window.setTimeout(() => setSent((cur) => (cur === res.note ? null : cur)), 2500);
    } else {
      setError(res.note); // keep the draft so the user can edit & retry
    }
  };
  const retake = () => {
    setRelayDraft(null);
    beginTalk();
  };

  return (
    <div className="relative flex flex-col gap-2 border-t border-line p-3">
      {/* Status + settings */}
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-fg-secondary">
          {busy && <Loader2 size={12} className="animate-spin text-accent" />}
          {status}
        </span>
        <button
          type="button"
          onClick={() => setSettingsOpen((v) => !v)}
          data-on={settingsOpen || undefined}
          className="inline-flex h-7 w-7 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg data-[on=true]:text-accent"
          aria-label="Voice settings"
          title="Voice settings"
        >
          <Settings2 size={15} />
        </button>
        {settingsOpen && (
          <div className="absolute bottom-full right-3 z-10 mb-2 w-52 max-w-[calc(100vw-1.5rem)] rounded-lg border border-line bg-surface-overlay p-3 shadow-lg">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-fg-muted">
              Push-to-talk
            </p>
            <div className="grid grid-cols-2 gap-1 rounded-md bg-surface-sunken p-1 text-xs">
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
          </div>
        )}
      </div>

      {error && <p className="text-xs text-error">{error}</p>}

      {relayDraft !== null ? (
        <div className="flex flex-col gap-2 animate-in fade-in-0 slide-in-from-bottom-1 duration-200">
          <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-accent">
            <CornerUpRight size={13} /> Draft to {agentName || 'the agent'}
          </div>
          <textarea
            value={relayDraft}
            onChange={(e) => setRelayDraft(e.target.value)}
            rows={2}
            className="ax-field ax-field--sm w-full resize-none rounded-lg max-[600px]:text-base"
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
              onClick={() => void sendRelay()}
              disabled={!relayDraft.trim()}
              className="ml-auto inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-fg-inverse shadow-sm transition hover:brightness-110 active:brightness-95 disabled:opacity-40"
            >
              <Send size={13} /> Send to agent
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3">
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
            className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-full border border-line bg-surface-raised text-fg-muted transition-colors hover:border-accent hover:text-accent data-[active=true]:border-accent data-[active=true]:bg-accent data-[active=true]:text-fg-inverse disabled:cursor-not-allowed disabled:opacity-40"
            aria-label={pttMode === 'hold' ? 'Hold to talk' : 'Tap to talk'}
            title={dictation.supported ? (pttMode === 'hold' ? 'Hold to talk' : 'Tap to talk') : 'Voice input is not supported here'}
          >
            {transcribing ? <Loader2 size={20} className="animate-spin" /> : <Mic size={20} />}
          </button>
          <span className="flex-1 text-[11px] leading-snug text-fg-muted">
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
            {hasAnswer && (
              <button
                type="button"
                onClick={() => setRelayDraft(lastTranscript.current)}
                className="ml-1 text-fg-muted underline-offset-2 transition-colors hover:text-accent hover:underline"
                title="Send what you said to the agent instead"
              >
                ↦ relay instead
              </button>
            )}
          </span>
        </div>
      )}
    </div>
  );
}

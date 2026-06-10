/**
 * AmbassadorPanel — the parallel conversation interpreter's surface (Phase 16.6).
 *
 * Subscribed to the *active* conversation tab. It briefs the conversation's turns,
 * answers free-form questions about it, and relays your messages into it — all
 * without the ambassador ever entering the transcript itself.
 *
 * Layout note: this typed/visual surface is the **fallback**. The hero on this
 * panel will be voice (TTS briefings + spoken Q&A); the header is intentionally
 * kept as a clean identity zone so that voice hero can slot in above the body
 * without a rewrite. Everything here stays for no-mic / type-it-out use.
 *
 * Briefings/Q&A live only here + the server's `ambassador:` Redis sidecar.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import {
  Radio,
  Loader2,
  AlertTriangle,
  X,
  Sparkles,
  RotateCcw,
  Ban,
  Send,
  Wand2,
  CornerUpRight,
  Volume2,
  Square,
  AudioLines,
  MessageSquare,
} from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import { isAssistantMessage, type AssistantMessage } from '../../lib/messages';
import { gatherTurnContext, resolveTurnAgentName } from '../../lib/ambassadorTurn';
import { getAvatarIcon } from '../../lib/avatars';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useSpeech } from '../../hooks/useSpeech';
import { VoiceSurface } from './VoiceSurface';
import { Button } from '../ui';
import { api } from '../../lib/api';
import type { AmbassadorBriefing, AmbassadorQA } from '../../lib/api';

type PanelMode = 'ask' | 'relay';

/** Per-item playback controls passed down from the panel's single `useSpeech`. */
interface SpeechControls {
  playingId: string | null;
  loadingId: string | null;
  speak: (id: string, text: string) => void;
  stop: () => void;
}

/** A speaker toggle for one speakable item (briefing summary or Q&A answer). */
function SpeakButton({
  id,
  text,
  speech,
  className = '',
}: {
  id: string;
  text: string;
  speech: SpeechControls;
  className?: string;
}) {
  const playing = speech.playingId === id;
  const loading = speech.loadingId === id;
  const label = playing ? 'Stop' : loading ? 'Preparing voice…' : 'Listen';
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-accent/15 hover:text-accent data-[on=true]:text-accent ${className}`}
      data-on={playing || undefined}
      onClick={() => (playing ? speech.stop() : speech.speak(id, text))}
    >
      {loading ? (
        <Loader2 size={13} className="animate-spin" />
      ) : playing ? (
        <Square size={13} />
      ) : (
        <Volume2 size={14} />
      )}
    </button>
  );
}

function snippet(text: string, max = 130): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/** A streaming text cursor in the ambassador's accent. */
function Cursor() {
  return <span className="ml-0.5 inline-block h-3.5 w-px animate-pulse bg-accent align-middle" />;
}

/** The ambassador's avatar motif — a small accent disc with the Radio mark. */
/** The ambassador's identity mark — its customizable avatar (from the ambassador
 *  profile) on an accent disc, falling back to the generic radio mark. */
function AmbassadorMark({ size = 20, avatar }: { size?: number; avatar?: string }) {
  const Icon = avatar ? getAvatarIcon(avatar) : Radio;
  return (
    <span
      className="flex shrink-0 items-center justify-center rounded-full bg-accent/15"
      style={{ width: size, height: size }}
    >
      <Icon size={Math.round(size * 0.55)} className="text-accent" />
    </span>
  );
}

/** A section divider: uppercase label + count + hairline rule. */
function SectionLabel({ children, count }: { children: ReactNode; count?: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">
        {children}
      </span>
      {typeof count === 'number' && <span className="text-[11px] text-fg-muted">{count}</span>}
      <span className="h-px flex-1 bg-line" />
    </div>
  );
}

/** A small status chip mirroring the briefing's lifecycle. */
function StatusChip({ status }: { status: AmbassadorBriefing['status'] | undefined }) {
  if (!status) return null;
  const base =
    'inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide';
  switch (status) {
    case 'streaming':
      return (
        <span className={`${base} bg-accent/15 text-accent`}>
          <Loader2 size={10} className="animate-spin" /> briefing
        </span>
      );
    case 'done':
      return <span className={`${base} bg-surface-sunken text-success`}>briefed</span>;
    case 'cancelled':
      return (
        <span className={`${base} bg-surface-sunken text-fg-muted`}>
          <Ban size={10} /> cancelled
        </span>
      );
    case 'error':
      return <span className={`${base} bg-surface-sunken text-error`}>error</span>;
    case 'empty_provider':
      return <span className={`${base} bg-surface-sunken text-warning`}>no model</span>;
    default:
      return null;
  }
}

function BriefingBody({ briefing }: { briefing: AmbassadorBriefing | undefined }) {
  if (!briefing) return null;
  if (briefing.status === 'error') {
    return (
      <p className="flex items-start gap-1.5 whitespace-pre-wrap text-sm text-error">
        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
        <span>{briefing.error || 'The ambassador could not brief this turn.'}</span>
      </p>
    );
  }
  if (briefing.status === 'empty_provider') {
    return (
      <p className="whitespace-pre-wrap text-sm text-warning">
        {briefing.summary || 'No model provider is configured for the ambassador.'}
      </p>
    );
  }
  const streaming = briefing.status === 'streaming';
  if (!briefing.summary && !streaming) return null;
  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-fg">
      {briefing.summary}
      {streaming && <Cursor />}
    </p>
  );
}

/** One free-form Q&A exchange — your question, the ambassador's answer. */
function QaItem({
  entry,
  onCancel,
  speech,
}: {
  entry: AmbassadorQA;
  onCancel: () => void;
  speech: SpeechControls;
}) {
  const streaming = entry.status === 'streaming';
  const speakable = entry.status === 'done' && !!entry.answer.trim();
  return (
    <li className="flex flex-col gap-2">
      <div className="max-w-[88%] self-end rounded-2xl rounded-br-md bg-accent px-3 py-1.5 text-sm leading-snug text-fg-inverse shadow-sm">
        {entry.question}
      </div>
      <div className="flex max-w-[94%] items-start gap-2 self-start">
        <AmbassadorMark size={22} />
        <div className="min-w-0 pt-0.5 text-sm leading-relaxed text-fg">
          {entry.status === 'error' ? (
            <span className="flex items-start gap-1.5 text-error">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {entry.error || 'The ambassador could not answer that.'}
            </span>
          ) : entry.status === 'empty_provider' ? (
            <span className="text-warning">
              {entry.answer || 'No model provider is configured for the ambassador.'}
            </span>
          ) : (
            <span className="whitespace-pre-wrap">
              {entry.answer}
              {streaming && <Cursor />}
              {entry.status === 'cancelled' && !entry.answer && (
                <span className="text-fg-muted">cancelled</span>
              )}
            </span>
          )}
          {streaming && (
            <button
              type="button"
              className="ml-2 inline-flex items-center gap-1 align-middle text-xs text-fg-muted transition-colors hover:text-error"
              onClick={onCancel}
              title="Cancel"
            >
              <X size={12} /> cancel
            </button>
          )}
          {speakable && (
            <SpeakButton
              id={`qa:${entry.qa_id}`}
              text={entry.answer}
              speech={speech}
              className="ml-1 align-middle"
            />
          )}
        </div>
      </div>
    </li>
  );
}

export function AmbassadorPanel() {
  const { activeTab, relayToConversation } = useConversation();
  const { briefingForMessage, briefingsFor, refresh, ccTurn, cancel, qaFor, ask, cancelQa } =
    useAmbassador();
  const { profiles, getAgentName } = useAgentProfile();
  const conversationId = activeTab?.sessionId;
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<PanelMode>('ask');
  const [refining, setRefining] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  // The briefing ambassador — supplies the speech model/voice and the voice-mode
  // opt-in. profiles includes ambassador-kind entries (only chat routing filters).
  const ambassadorProfile = useMemo(
    () =>
      profiles.find((p) => p.kind === 'ambassador' && p.isDefaultAmbassador) ??
      profiles.find((p) => p.kind === 'ambassador'),
    [profiles],
  );
  const voiceEnabled = ambassadorProfile?.ambassador?.voiceMode === true;

  const { playingId, loadingId, speak, stop: stopSpeech, unlock } = useSpeech({
    agentProfileId: ambassadorProfile?.id,
  });
  const speech: SpeechControls = useMemo(
    () => ({ playingId, loadingId, speak, stop: stopSpeech }),
    [playingId, loadingId, speak, stopSpeech],
  );

  // Voice-enabled ambassadors lead with a Voice tab (immersive surface), with the
  // text panel as a secondary [Voice | Text] tab. Non-opted-in ⇒ text only.
  const [tab, setTab] = useState<'voice' | 'text'>('voice');
  const voiceActive = voiceEnabled && tab === 'voice';
  // Items already auto-spoken in voice mode (so a re-render / history never replays).
  const spokenRef = useRef<Set<string>>(new Set());

  // Subscribe: repopulate from the sidecar whenever the active conversation changes.
  useEffect(() => {
    if (conversationId) void refresh(conversationId);
  }, [conversationId, refresh]);

  // On conversation switch: stop any audio + seed existing done items as
  // already-spoken, so switching never auto-replays a conversation's history.
  useEffect(() => {
    stopSpeech();
    if (!conversationId) return;
    const seed = spokenRef.current;
    for (const b of briefingsFor(conversationId)) if (b.status === 'done') seed.add(`brief:${b.message_id}`);
    for (const q of qaFor(conversationId)) if (q.status === 'done') seed.add(`qa:${q.qa_id}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  const messages = activeTab?.messages ?? [];
  // Assistant turns (non-empty content), newest first, carrying their index.
  const turns = useMemo(
    () =>
      (messages
        .map((m, i) => ({ m, i }))
        .filter((x) => isAssistantMessage(x.m) && x.m.content.trim().length > 0)
        .reverse() as { m: AssistantMessage; i: number }[]),
    [messages],
  );

  const brief = (m: AssistantMessage) => {
    if (!conversationId) return;
    const { userText, artifacts } = gatherTurnContext(messages, m.id);
    const tabProfileName = activeTab?.profileId
      ? profiles.find((p) => p.id === activeTab.profileId)?.name
      : undefined;
    const agentName = resolveTurnAgentName(m, {
      nameByProfileId: (id) => profiles.find((p) => p.id === id)?.name,
      fallback: tabProfileName ?? getAgentName(),
    });
    ccTurn(conversationId, m, { userText, artifacts, agentName });
  };

  const latest = turns[0];
  const latestStreaming =
    !!latest && briefingForMessage(conversationId, latest.m.id)?.status === 'streaming';

  // Q&A thread, oldest first (newest sits next to the input). Brand-new local
  // entries (no created_at yet) sort last.
  const qaList = useMemo(() => {
    const items = qaFor(conversationId);
    return [...items].sort((a, b) => {
      const ta = a.created_at ? Date.parse(a.created_at) : Infinity;
      const tb = b.created_at ? Date.parse(b.created_at) : Infinity;
      return ta - tb;
    });
  }, [qaFor, conversationId]);

  const anyStreaming = useMemo(
    () =>
      briefingsFor(conversationId).some((b) => b.status === 'streaming') ||
      qaFor(conversationId).some((q) => q.status === 'streaming'),
    [briefingsFor, qaFor, conversationId],
  );

  // Conversation-level agent name + latest-turn substance, to ground a question/draft.
  const convAgentName = useMemo(() => {
    const tabProfileName = activeTab?.profileId
      ? profiles.find((p) => p.id === activeTab.profileId)?.name
      : undefined;
    return tabProfileName ?? getAgentName();
  }, [activeTab?.profileId, profiles, getAgentName]);

  const latestArtifacts = () =>
    latest ? gatherTurnContext(messages, latest.m.id).artifacts : undefined;

  const runActive = !!activeTab?.activeRun?.runId;

  // The text currently being spoken (for the immersive surface caption).
  const spokenText = useMemo(() => {
    if (!playingId) return '';
    if (playingId.startsWith('brief:')) {
      const mid = playingId.slice('brief:'.length);
      return briefingForMessage(conversationId, mid)?.summary ?? '';
    }
    if (playingId.startsWith('qa:')) {
      const qid = playingId.slice('qa:'.length);
      return qaFor(conversationId).find((q) => q.qa_id === qid)?.answer ?? '';
    }
    return '';
  }, [playingId, conversationId, briefingForMessage, qaFor]);

  const seedSpoken = () => {
    const seed = spokenRef.current;
    for (const b of briefingsFor(conversationId)) if (b.status === 'done') seed.add(`brief:${b.message_id}`);
    for (const q of qaFor(conversationId)) if (q.status === 'done') seed.add(`qa:${q.qa_id}`);
  };

  const openVoiceTab = () => {
    unlock(); // bless the audio element on this gesture so autoplay works
    seedSpoken(); // don't replay briefings already on screen
    setTab('voice');
  };
  const openTextTab = () => {
    stopSpeech();
    setTab('text');
  };

  // Latest-turn substance (memoized) to ground voice commands.
  const voiceArtifacts = useMemo(
    () => (latest ? gatherTurnContext(messages, latest.m.id).artifacts : undefined),
    [messages, latest],
  );
  // Relay a confirmed voice draft into the conversation (a real user turn).
  const relayVoiceCommand = useCallback(
    (text: string) => (activeTab ? relayToConversation(activeTab.id, text) : false),
    [activeTab, relayToConversation],
  );
  const onAnswerPersisted = useCallback(() => {
    if (conversationId) void refresh(conversationId);
  }, [conversationId, refresh]);

  // Voice-tab autoplay: speak the freshest newly-settled briefing / answer once.
  useEffect(() => {
    if (!voiceActive || !conversationId) return;
    type Item = { id: string; text: string; ts: number };
    const items: Item[] = [];
    for (const b of briefingsFor(conversationId))
      if (b.status === 'done' && b.summary.trim())
        items.push({ id: `brief:${b.message_id}`, text: b.summary, ts: Date.parse(b.updated_at ?? '') || 0 });
    for (const q of qaFor(conversationId))
      if (q.status === 'done' && q.answer.trim())
        items.push({ id: `qa:${q.qa_id}`, text: q.answer, ts: Date.parse(q.updated_at ?? '') || 0 });
    const fresh = items.filter((it) => !spokenRef.current.has(it.id));
    if (fresh.length === 0) return;
    fresh.sort((a, b) => a.ts - b.ts);
    for (const it of fresh) spokenRef.current.add(it.id); // mark all seen…
    const last = fresh[fresh.length - 1];
    speak(last.id, last.text); // …but only voice the most recent
  }, [voiceActive, conversationId, briefingsFor, qaFor, speak]);

  const showFlash = (msg: string) => {
    setFlash(msg);
    window.setTimeout(() => setFlash((cur) => (cur === msg ? null : cur)), 2500);
  };

  const submitAsk = () => {
    if (!conversationId) return;
    const q = input.trim();
    if (!q) return;
    ask(conversationId, q, { agentName: convAgentName, artifacts: latestArtifacts() });
    setInput('');
  };

  // Relay the message into the actual conversation (a real user turn, or a steer
  // into the running turn). The ambassador stays a non-participant — you're the author.
  const submitRelay = () => {
    if (!activeTab) return;
    const t = input.trim();
    if (!t) return;
    const ok = relayToConversation(activeTab.id, t);
    if (!ok) {
      showFlash('Open the conversation to relay a message.');
      return;
    }
    setInput('');
    showFlash(runActive ? 'Folded into the running turn.' : 'Sent to the conversation.');
  };

  // Optional: let the ambassador shape a rough intent into a ready-to-send message.
  const refine = async () => {
    if (!conversationId) return;
    const intent = input.trim();
    if (!intent) return;
    setRefining(true);
    try {
      const { draft } = await api.draftRelay({
        conversation_id: conversationId,
        intent,
        agent_name: convAgentName,
        artifacts: latestArtifacts(),
      });
      if (draft) setInput(draft);
    } catch {
      /* keep the raw intent — the relay still works */
    } finally {
      setRefining(false);
    }
  };

  const submit = () => (mode === 'ask' ? submitAsk() : submitRelay());

  const onInputKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const footerHelp =
    flash ??
    (mode === 'relay'
      ? 'Sent as your own message — the ambassador never speaks into the conversation itself.'
      : 'Answered from the conversation only — never added to the transcript.');

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header — identity zone (future home of the voice/TTS hero). */}
      <div className="flex flex-col gap-1.5 border-b border-line px-4 pb-3.5 pt-4">
        {/* pr clears the drawer shell's absolute close button (top-right, ~56px). */}
        <div className="flex flex-wrap items-center gap-2.5 pr-12">
          <span className="relative inline-flex">
            <AmbassadorMark size={26} avatar={ambassadorProfile?.avatar} />
            {anyStreaming && (
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-ping rounded-full bg-accent" />
            )}
          </span>
          <h2 className="text-base font-semibold text-fg">{ambassadorProfile?.name || 'Ambassador'}</h2>
          {anyStreaming && (
            <span className="inline-flex items-center gap-1 rounded-full bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-accent">
              live
            </span>
          )}
          {voiceEnabled && conversationId && (
            <div className="ml-auto inline-flex items-center gap-0.5 rounded-full bg-surface-sunken p-0.5 text-xs">
              <button
                type="button"
                onClick={openVoiceTab}
                data-on={tab === 'voice' || undefined}
                className="inline-flex items-center gap-1 rounded-full px-2.5 py-1 font-medium text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-accent data-[on=true]:text-fg-inverse data-[on=true]:shadow-sm"
                title="Immersive voice"
              >
                <AudioLines size={13} /> Voice
              </button>
              <button
                type="button"
                onClick={openTextTab}
                data-on={tab === 'text' || undefined}
                className="inline-flex items-center gap-1 rounded-full px-2.5 py-1 font-medium text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-fg data-[on=true]:shadow-sm"
                title="Text"
              >
                <MessageSquare size={13} /> Text
              </button>
            </div>
          )}
        </div>
        <p className="text-xs leading-relaxed text-fg-muted">
          {conversationId ? (
            <>
              Interpreting <span className="font-medium text-fg-secondary">{convAgentName}</span> in
              parallel — it briefs and answers without ever entering the conversation.
            </>
          ) : (
            'Your parallel interpreter — it reads a conversation and briefs or answers, without ever entering it.'
          )}
        </p>
      </div>

      {voiceActive && conversationId && (
        <VoiceSurface
          conversationId={conversationId}
          agentProfileId={ambassadorProfile?.id}
          agentName={convAgentName}
          artifacts={voiceArtifacts}
          ambientSpokenText={spokenText}
          onRelay={relayVoiceCommand}
          onAnswerPersisted={onAnswerPersisted}
        />
      )}

      {/* Voice mode but no conversation open yet — never render a blank surface. */}
      {voiceActive && !conversationId && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
          <AmbassadorMark size={40} avatar={ambassadorProfile?.avatar} />
          <p className="text-sm text-fg-muted">
            Open a conversation and you can talk to the ambassador about it here.
          </p>
        </div>
      )}

      {!voiceActive && (
      <>
      {/* Body */}
      <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-4">
        {!conversationId ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
            <AmbassadorMark size={40} avatar={ambassadorProfile?.avatar} />
            <p className="text-sm text-fg-muted">
              Open a conversation and the ambassador can brief its turns or answer your questions
              about it here.
            </p>
          </div>
        ) : (
          <>
            {turns.length === 0 ? (
              <p className="rounded-lg border border-dashed border-line bg-surface-sunken p-4 text-center text-sm text-fg-muted">
                No replies yet — you can still ask the ambassador about this conversation below.
              </p>
            ) : (
              <section className="flex flex-col gap-3">
                <Button
                  variant="primary"
                  className="w-full"
                  loading={latestStreaming}
                  onClick={() => latest && brief(latest.m)}
                >
                  {!latestStreaming && <Sparkles size={15} />}
                  {latestStreaming ? 'Briefing the latest turn…' : 'Brief the latest turn'}
                </Button>

                <SectionLabel count={turns.length}>Turns</SectionLabel>

                <ul className="flex flex-col gap-2">
                  {turns.map(({ m }, idx) => {
                    const briefing = briefingForMessage(conversationId, m.id);
                    const turnNo = turns.length - idx; // oldest = 1
                    const streaming = briefing?.status === 'streaming';
                    const briefed = !!briefing && briefing.status !== 'streaming';
                    return (
                      <li
                        key={m.id}
                        className="flex flex-col gap-2 rounded-lg border border-line bg-surface-raised p-3 transition-colors data-[briefed=true]:border-line-strong"
                        data-briefed={briefed || undefined}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-fg-secondary">
                            Turn {turnNo}
                          </span>
                          <StatusChip status={briefing?.status} />
                          {briefed && briefing?.summary?.trim() && (
                            <SpeakButton
                              id={`brief:${m.id}`}
                              text={briefing.summary}
                              speech={speech}
                              className="ml-auto"
                            />
                          )}
                          {streaming ? (
                            <button
                              type="button"
                              className="ml-auto inline-flex items-center gap-1 rounded-md border border-line px-2 py-1 text-xs font-medium text-fg-muted transition-colors hover:border-error hover:text-error"
                              onClick={() => cancel(conversationId, m.id)}
                              title="Cancel briefing"
                            >
                              <X size={12} /> cancel
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="inline-flex items-center gap-1 rounded-md border border-line px-2 py-1 text-xs font-medium text-accent transition-colors hover:border-accent hover:bg-accent/15 data-[lead=true]:ml-auto"
                              data-lead={!(briefed && briefing?.summary?.trim()) || undefined}
                              onClick={() => brief(m)}
                              title={briefing ? 'Brief this turn again' : 'Brief this turn'}
                            >
                              {briefing ? <RotateCcw size={12} /> : <Radio size={12} />}
                              {briefing ? 're-brief' : 'brief'}
                            </button>
                          )}
                        </div>
                        <p className="border-l-2 border-line pl-2 text-xs italic text-fg-muted">
                          {snippet(m.content)}
                        </p>
                        <BriefingBody briefing={briefing} />
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}

            {qaList.length > 0 && (
              <section className="flex flex-col gap-3">
                <SectionLabel count={qaList.length}>Questions</SectionLabel>
                <ul className="flex flex-col gap-3.5">
                  {qaList.map((entry) => (
                    <QaItem
                      key={entry.qa_id}
                      entry={entry}
                      onCancel={() => cancelQa(conversationId, entry.qa_id)}
                      speech={speech}
                    />
                  ))}
                </ul>
              </section>
            )}
          </>
        )}
      </div>

      {/* Pinned input — Ask the ambassador, or Relay a message to the agent. */}
      {conversationId && (
        <div className="flex flex-col gap-2 border-t border-line p-3">
          {/* Mode toggle — segmented control. */}
          <div className="grid grid-cols-2 gap-1 rounded-lg bg-surface-sunken p-1 text-xs">
            <button
              type="button"
              onClick={() => setMode('ask')}
              className="inline-flex items-center justify-center gap-1.5 rounded-md px-2 py-1.5 font-medium text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-fg data-[on=true]:shadow-sm"
              data-on={mode === 'ask' || undefined}
            >
              <Radio size={12} /> Ask
            </button>
            <button
              type="button"
              onClick={() => setMode('relay')}
              className="inline-flex items-center justify-center gap-1.5 rounded-md px-2 py-1.5 font-medium text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-fg data-[on=true]:shadow-sm"
              data-on={mode === 'relay' || undefined}
            >
              <CornerUpRight size={12} /> Relay to agent
            </button>
          </div>

          <div className="flex items-end gap-2 rounded-lg border border-line bg-surface-raised px-2 py-1.5 transition-colors focus-within:border-line-strong">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onInputKeyDown}
              rows={mode === 'relay' ? 2 : 1}
              placeholder={
                mode === 'ask'
                  ? `Ask ${convAgentName ? `about ${convAgentName}` : 'about this conversation'}…`
                  : runActive
                    ? `Tell ${convAgentName} something — folds into the running turn…`
                    : `Tell ${convAgentName} something — sent as your message…`
              }
              className="max-h-32 flex-1 resize-none bg-transparent p-0 text-sm text-fg outline-none placeholder:text-fg-muted max-[600px]:text-base"
            />
            {mode === 'relay' && (
              <button
                type="button"
                onClick={refine}
                disabled={!input.trim() || refining}
                className="inline-flex h-8 shrink-0 items-center gap-1 rounded-md border border-line px-2 text-xs text-fg-muted transition-colors hover:border-line-strong hover:text-accent disabled:opacity-40"
                title="Let the ambassador shape this into a ready-to-send message"
              >
                {refining ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
                Refine
              </button>
            )}
            <button
              type="button"
              onClick={submit}
              disabled={!input.trim()}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent text-fg-inverse transition-colors hover:bg-accent-secondary disabled:opacity-40"
              title={mode === 'ask' ? 'Ask the ambassador' : 'Send to the conversation'}
            >
              {mode === 'ask' ? <Send size={14} /> : <CornerUpRight size={14} />}
            </button>
          </div>

          {/* One stable helper line — flash takes over briefly, no layout shift. */}
          <p className="px-0.5 text-[11px] leading-snug text-fg-muted">{footerHelp}</p>
        </div>
      )}
      </>
      )}
    </div>
  );
}

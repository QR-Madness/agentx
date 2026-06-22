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
  Send,
  Wand2,
  CornerUpRight,
  Volume2,
  Square,
  AudioLines,
  MessageSquare,
  Check,
  MoreHorizontal,
  Pencil,
  Trash2,
  Copy,
  ArrowDown,
} from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import { getAvatarIcon } from '../../lib/avatars';
import { toolChipLabel } from '../../lib/ambassadorTools';
import { relayToActiveConversation } from '../../lib/ambassadorRelay';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useSpeech } from '../../hooks/useSpeech';
import { useStickyScroll } from '../../hooks/useStickyScroll';
import { VoiceBar } from './VoiceBar';
import { AmbassadorConversationSwitcher, type SwitcherItem } from './AmbassadorConversationSwitcher';
import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui';
import { useConfirm } from '../ui/ConfirmDialog';
import { api } from '../../lib/api';
import type { AmbassadorBriefing, AmbassadorQA, AmbassadorToolCall } from '../../lib/api';

type PanelMode = 'ask' | 'relay';

/** Creative, conversation-scoped openers for a fresh Inquiry (fire as an ask). */
const STARTERS = ['Catch me up', "What's been decided?", "What's unresolved?"] as const;

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

/** Copy an answer/briefing to the clipboard, with a brief check confirmation. */
function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked — no-op */
    }
  };
  return (
    <button
      type="button"
      aria-label={copied ? 'Copied' : 'Copy'}
      title={copied ? 'Copied' : 'Copy'}
      onClick={copy}
      className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-accent/15 hover:text-accent ${className}`}
    >
      {copied ? <Check size={13} className="text-success" /> : <Copy size={13} />}
    </button>
  );
}

/** Compact relative timestamp ("now", "5m", "2h", "3d") from an ISO string. */
function relativeTime(iso?: string): string {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 45) return 'now';
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.round(hrs / 24);
  if (days < 7) return `${days}d`;
  return new Date(then).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
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

/** Live chips for the read-only tools the ambassador calls while answering —
 *  spinner while running, check when done — so you can see it reading/surveying. */
function ToolChips({ calls }: { calls?: AmbassadorToolCall[] }) {
  if (!calls?.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {calls.map((c, i) => (
        <span
          key={`${c.tool}-${i}`}
          className="inline-flex items-center gap-1 rounded-full bg-surface-sunken px-2 py-0.5 text-[11px] text-fg-secondary"
        >
          {c.done ? (
            <Check size={11} className="text-success" />
          ) : (
            <Loader2 size={11} className="animate-spin text-accent" />
          )}
          {toolChipLabel(c.tool, c.args)}
          {!c.done && '…'}
        </span>
      ))}
    </div>
  );
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
  const copyable = !streaming && !!entry.answer.trim();
  return (
    <li className="flex flex-col gap-2">
      <div className="max-w-[88%] self-end rounded-2xl rounded-br-md bg-accent px-3 py-1.5 text-sm leading-snug text-fg-inverse shadow-sm">
        {entry.question}
      </div>
      <div className="flex max-w-[94%] items-start gap-2 self-start">
        <AmbassadorMark size={22} />
        <div className="flex min-w-0 flex-col gap-1 pt-0.5 text-sm leading-relaxed text-fg">
          <ToolChips calls={entry.toolCalls} />
          <div>
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
          </div>
          {!streaming && (
            <div className="flex items-center gap-0.5 text-fg-muted">
              {entry.created_at && (
                <span className="mr-0.5 text-[10px] tabular-nums">{relativeTime(entry.created_at)}</span>
              )}
              {speakable && <SpeakButton id={`qa:${entry.qa_id}`} text={entry.answer} speech={speech} />}
              {copyable && <CopyButton text={entry.answer} />}
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

/** One briefing in the unified Inquiry stream — the ambassador's take on a turn,
 *  rendered as its own turn (mark + tool chips + prose + speak), so briefings and
 *  Q&A read as one conversation. The per-turn trigger lives in the Turns strip. */
function BriefingItem({
  briefing,
  onCancel,
  speech,
}: {
  briefing: AmbassadorBriefing;
  onCancel: () => void;
  speech: SpeechControls;
}) {
  const streaming = briefing.status === 'streaming';
  const speakable = briefing.status === 'done' && !!briefing.summary.trim();
  const copyable = !streaming && !!briefing.summary.trim();
  return (
    <li className="flex max-w-[94%] items-start gap-2 self-start">
      <AmbassadorMark size={22} />
      <div className="flex min-w-0 flex-col gap-1 pt-0.5">
        <ToolChips calls={briefing.toolCalls} />
        <BriefingBody briefing={briefing} />
        {streaming ? (
          <button
            type="button"
            className="inline-flex w-fit items-center gap-1 text-xs text-fg-muted transition-colors hover:text-error"
            onClick={onCancel}
            title="Cancel briefing"
          >
            <X size={12} /> cancel
          </button>
        ) : (
          (briefing.created_at || speakable || copyable) && (
            <div className="flex items-center gap-0.5 text-fg-muted">
              {briefing.created_at && (
                <span className="mr-0.5 text-[10px] tabular-nums">{relativeTime(briefing.created_at)}</span>
              )}
              {speakable && (
                <SpeakButton id={`brief:${briefing.message_id}`} text={briefing.summary} speech={speech} />
              )}
              {copyable && <CopyButton text={briefing.summary} />}
            </div>
          )
        )}
      </div>
    </li>
  );
}

export function AmbassadorPanel() {
  const { activeTab, tabs, relayToConversation } = useConversation();
  const {
    briefingsFor, refresh, cancel, qaFor,
    threadFor, titleFor, renameThread, clearThread, ask, cancelQa,
  } = useAmbassador();
  const confirm = useConfirm();
  const { profiles, getAgentName } = useAgentProfile();

  // Conversations the ambassador can be pointed at — the open tabs that have a saved
  // session (id + title). (Server-history conversations are a follow-up for the full
  // command deck.)
  const conversationItems = useMemo<SwitcherItem[]>(
    () =>
      tabs
        .filter((t) => !!t.sessionId)
        .map((t) => ({ id: t.sessionId as string, title: t.title || 'Conversation' })),
    [tabs],
  );

  // The conversation the person is *currently in* (their active chat tab).
  const activeConversationId = activeTab?.sessionId ?? undefined;
  // The ambassador's own focus — independent of the chat tab. Snapshots the active
  // conversation when the panel first opens, then **stays put** (switching chat tabs
  // never moves it); the switcher / "← current" change it explicitly.
  const [focusedConversationId, setFocusedConversationId] = useState<string | undefined>(undefined);
  useEffect(() => {
    if (focusedConversationId === undefined && activeConversationId) {
      setFocusedConversationId(activeConversationId);
    }
  }, [focusedConversationId, activeConversationId]);
  const conversationId = focusedConversationId ?? activeConversationId;
  const isFocusActive = !!conversationId && conversationId === activeConversationId;

  const [input, setInput] = useState('');
  const [mode, setMode] = useState<PanelMode>('ask');
  const [refining, setRefining] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  // Inline Inquiry rename (null = not renaming).
  const [renamingTitle, setRenamingTitle] = useState<string | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (renamingTitle !== null) renameInputRef.current?.select();
  }, [renamingTitle]);

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
  // Items already auto-spoken in voice mode (so a re-render never replays).
  const spokenRef = useRef<Set<string>>(new Set());
  // Items that were created HERE this session (seen in `streaming` state) — the only
  // ones auto-speak may voice. Persisted items loaded by `refresh()` arrive already
  // `done` and are never in here, so switching/reopening a conversation never speaks
  // or re-synthesizes its history (no TTS request, no spend). The ambassador waits.
  const locallyStreamedRef = useRef<Set<string>>(new Set());

  // Subscribe: repopulate from the sidecar whenever the focused conversation changes.
  useEffect(() => {
    if (conversationId) void refresh(conversationId);
  }, [conversationId, refresh]);

  // On focus switch: stop audio, drop any half-typed rename, and re-pin to the latest.
  useEffect(() => {
    stopSpeech();
    setRenamingTitle(null);
    scrollToBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  // Record items that stream locally (the user asked/briefed here this session), so
  // only those are eligible for auto-speak.
  useEffect(() => {
    if (!conversationId) return;
    const seen = locallyStreamedRef.current;
    for (const b of briefingsFor(conversationId)) if (b.status === 'streaming') seen.add(`brief:${b.message_id}`);
    for (const q of qaFor(conversationId)) if (q.status === 'streaming') seen.add(`qa:${q.qa_id}`);
  }, [conversationId, briefingsFor, qaFor]);

  // The ambassador runs PARALLEL to your conversations — it isn't bound to one. The
  // panel is its own space (the Inquiry). Turn-coupling lives in the chat's CC button;
  // here you ask it, brief a whole conversation, or relay — it scopes via its own tools,
  // so we never feed it per-turn artifacts.
  const thread = useMemo(() => threadFor(conversationId), [threadFor, conversationId]);
  // The Inquiry's own title, or the focused conversation's title as a fallback.
  const inquiryTitle = titleFor(conversationId);

  // The Inquiry body follows new content while pinned to the bottom (jump pill otherwise).
  const { ref: bodyRef, atBottom, scrollToBottom, onScroll } = useStickyScroll([thread]);

  const anyStreaming = useMemo(
    () =>
      briefingsFor(conversationId).some((b) => b.status === 'streaming') ||
      qaFor(conversationId).some((q) => q.status === 'streaming'),
    [briefingsFor, qaFor, conversationId],
  );

  // Agent-name hint for the FOCUSED conversation: only when the focus is the active
  // tab do we know it client-side; for any other conversation the backend names agents
  // per-conversation (so we pass nothing rather than mislabel).
  const convAgentName = useMemo(() => {
    if (!isFocusActive) return '';
    const tabProfileName = activeTab?.profileId
      ? profiles.find((p) => p.id === activeTab.profileId)?.name
      : undefined;
    return tabProfileName ?? getAgentName();
  }, [isFocusActive, activeTab?.profileId, profiles, getAgentName]);

  // Display label for the focused conversation (agent name when active, else its title).
  const focusItem = useMemo(
    () => conversationItems.find((it) => it.id === conversationId),
    [conversationItems, conversationId],
  );
  const focusTitle = isFocusActive ? (convAgentName || 'your agent') : (focusItem?.title || 'that conversation');

  // Where the person is *now* (ambient context) — only worth sending when the
  // ambassador is focused elsewhere (otherwise the focus already is where they are).
  const activeConversation = useMemo(
    () =>
      !isFocusActive && activeConversationId
        ? { id: activeConversationId, title: activeTab?.title }
        : undefined,
    [isFocusActive, activeConversationId, activeTab?.title],
  );

  // Relay only ever reaches the conversation the person is *in* (the active tab) — it is
  // the one conversation with a live send handler (ChatPanel registers it per active tab).
  const runActive = !!activeTab?.activeRun?.runId;

  const openVoiceTab = () => {
    unlock(); // bless the audio element on this gesture so autoplay works
    setTab('voice');
  };
  const openTextTab = () => {
    stopSpeech();
    setTab('text');
  };

  // Relay a message into the active conversation — a real user turn (or a steer into its
  // running turn). The ambassador stays a non-participant; you are the author. Returns
  // where it landed (or why it couldn't) so voice + text both give the same closure.
  const relay = useCallback(
    (text: string) => relayToActiveConversation(text, activeTab, relayToConversation),
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
    // Only items created HERE this session (streamed locally) are eligible — never
    // persisted history loaded from the sidecar. This is what keeps reopening silent.
    const fresh = items.filter(
      (it) => locallyStreamedRef.current.has(it.id) && !spokenRef.current.has(it.id),
    );
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

  const askAmbassador = (q: string) => {
    if (!conversationId) return;
    const text = q.trim();
    if (!text) return;
    ask(conversationId, text, { agentName: convAgentName, activeConversation });
  };

  const submitAsk = () => {
    if (!input.trim()) return;
    askAmbassador(input);
    setInput('');
  };

  // Brief the WHOLE conversation on demand — a one-tap question the ambassador answers
  // with its conversation tools (no turn coupling; this is "brief me on this conversation").
  const briefConversation = () => askAmbassador('Brief me on this conversation — what it is about and where it stands.');

  // Relay the message into the active conversation (a real user turn, or a steer into a
  // running turn). The ambassador stays a non-participant — you're the author.
  const submitRelay = () => {
    if (!input.trim()) return;
    const res = relay(input);
    if (res.ok) setInput('');
    showFlash(res.note);
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

  // --- Inquiry actions (⋯ menu): rename + clear --------------------------------
  const startRenameInquiry = () => setRenamingTitle(inquiryTitle || '');
  const commitRenameInquiry = () => {
    if (renamingTitle !== null && conversationId) renameThread(conversationId, renamingTitle);
    setRenamingTitle(null);
  };
  const clearInquiry = async () => {
    if (!conversationId) return;
    const ok = await confirm({
      title: 'Clear this Inquiry?',
      body: 'Everything the ambassador has said here — briefings and answers — will be removed. This can’t be undone.',
      confirmLabel: 'Clear',
      danger: true,
    });
    if (ok) clearThread(conversationId);
  };

  const footerHelp =
    flash ??
    (mode === 'relay'
      ? 'Sent as your own message — the ambassador never speaks into the conversation itself.'
      : 'Answered from the conversation only — never added to the transcript.');

  // --- Header sub-elements (shared by the compact bar + the voice hero) --------

  // The Inquiry title doubles as the conversation switcher; renaming swaps it for an input.
  const titleControl = conversationId && (
    renamingTitle !== null ? (
      <input
        ref={renameInputRef}
        value={renamingTitle}
        onChange={(e) => setRenamingTitle(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commitRenameInquiry();
          else if (e.key === 'Escape') setRenamingTitle(null);
        }}
        onBlur={commitRenameInquiry}
        placeholder={focusItem?.title || 'Name this Inquiry'}
        aria-label="Rename this Inquiry"
        className="min-w-0 max-w-[170px] rounded-md border border-line-strong bg-surface-raised px-1.5 py-0.5 text-sm text-fg outline-none"
      />
    ) : (
      <AmbassadorConversationSwitcher
        variant="inline"
        items={conversationItems}
        focusedId={conversationId}
        activeId={activeConversationId}
        onSelect={setFocusedConversationId}
        title={inquiryTitle}
      />
    )
  );

  const modeToggle = voiceEnabled && conversationId && (
    <div className="inline-flex shrink-0 items-center gap-0.5 rounded-full bg-surface-sunken p-0.5">
      <button
        type="button"
        onClick={openVoiceTab}
        data-on={tab === 'voice' || undefined}
        className="inline-flex h-6 w-6 items-center justify-center rounded-full text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-accent data-[on=true]:shadow-sm"
        title="Immersive voice"
        aria-label="Voice"
      >
        <AudioLines size={13} />
      </button>
      <button
        type="button"
        onClick={openTextTab}
        data-on={tab === 'text' || undefined}
        className="inline-flex h-6 w-6 items-center justify-center rounded-full text-fg-secondary transition-colors hover:text-fg data-[on=true]:bg-surface-raised data-[on=true]:text-accent data-[on=true]:shadow-sm"
        title="Text"
        aria-label="Text"
      >
        <MessageSquare size={13} />
      </button>
    </div>
  );

  const overflowMenu = conversationId && (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          title="Inquiry actions"
          aria-label="Inquiry actions"
          className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg"
        >
          <MoreHorizontal size={16} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onSelect={briefConversation}>
          <Sparkles size={14} /> Brief this conversation
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => window.setTimeout(startRenameInquiry, 0)}>
          <Pencil size={14} /> Rename Inquiry
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => { void clearInquiry(); }}
          className="text-error focus:text-error"
        >
          <Trash2 size={14} /> Clear Inquiry
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  const contextLine: ReactNode = !conversationId
    ? 'Your parallel operator — it watches your conversations and answers, without ever entering them.'
    : isFocusActive
      ? <>watching <span className="font-medium text-fg-secondary">{focusTitle}</span> · in parallel</>
      : <>on <span className="font-medium text-fg-secondary">{focusTitle}</span> · stays put</>;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header — ONE stable command bar in both modes (the context is unified; only the
          footer input differs). Voice adds a subtle accent tint, never a layout morph, so
          the Voice/Text toggle never moves and you can't get stranded. */}
      <div
        className={`flex flex-col gap-1 border-b border-line px-4 pb-2.5 pt-3.5 transition-colors duration-300 ${voiceActive ? 'bg-gradient-to-b from-accent/8 to-transparent' : ''}`}
      >
        {/* pr-10 clears the shell's absolute close button (top-right). */}
        <div className="flex items-center gap-2 pr-10">
          <span className="relative inline-flex shrink-0">
            <AmbassadorMark size={26} avatar={ambassadorProfile?.avatar} />
            {anyStreaming && (
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-ping rounded-full bg-accent" />
            )}
          </span>
          <div className="flex min-w-0 flex-1 items-center gap-1">
            <span className="shrink-0 text-sm font-semibold text-fg">
              {ambassadorProfile?.name || 'Ambassador'}
            </span>
            {conversationId && (
              <>
                <span className="shrink-0 text-fg-muted">·</span>
                {titleControl}
              </>
            )}
          </div>
          {modeToggle}
          {overflowMenu}
        </div>
        <p className="pl-[34px] text-[11px] leading-snug text-fg-muted">{contextLine}</p>
      </div>

      {/* Body — the Inquiry (shared by text + voice; only the footer differs). */}
      <div className="relative flex min-h-0 flex-1 flex-col">
        <div
          ref={bodyRef}
          onScroll={onScroll}
          className="flex flex-1 flex-col gap-4 overflow-y-auto p-4"
        >
          {!conversationId ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
              <AmbassadorMark size={40} avatar={ambassadorProfile?.avatar} />
              <p className="max-w-[18rem] text-sm text-fg-muted">
                Open a conversation and the ambassador can answer, brief, and relay — in parallel,
                without ever entering it.
              </p>
            </div>
          ) : thread.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 px-4 text-center">
              <AmbassadorMark size={38} avatar={ambassadorProfile?.avatar} />
              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium text-fg">Start an Inquiry</p>
                <p className="max-w-[18rem] text-xs text-fg-muted">
                  Ask anything about your conversations, or have the ambassador brief this one.
                </p>
              </div>
              <Button variant="primary" onClick={briefConversation}>
                <Sparkles size={15} /> Brief this conversation
              </Button>
              <div className="flex flex-wrap items-center justify-center gap-1.5">
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => askAmbassador(s)}
                    className="rounded-full border border-line bg-surface-raised px-2.5 py-1 text-xs text-fg-secondary transition-colors hover:border-line-strong hover:text-fg"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <ul className="flex flex-col gap-3.5">
              {thread.map((item) =>
                item.kind === 'qa' ? (
                  <QaItem
                    key={`qa:${item.id}`}
                    entry={item.qa}
                    onCancel={() => cancelQa(conversationId, item.id)}
                    speech={speech}
                  />
                ) : (
                  <BriefingItem
                    key={`brief:${item.id}`}
                    briefing={item.briefing}
                    onCancel={() => cancel(conversationId, item.id)}
                    speech={speech}
                  />
                ),
              )}
            </ul>
          )}
        </div>
        {!atBottom && thread.length > 0 && (
          <button
            type="button"
            onClick={scrollToBottom}
            className="absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-1 rounded-full border border-line bg-surface-overlay px-3 py-1 text-xs text-fg-secondary shadow-md backdrop-blur-sm transition-colors hover:border-line-strong hover:text-fg animate-in fade-in-0 slide-in-from-bottom-2 duration-200"
          >
            <ArrowDown size={13} /> Latest
          </button>
        )}
      </div>

      {/* Footer — push-to-talk in voice mode, the text composer otherwise. */}
      {conversationId && (voiceActive ? (
        <VoiceBar
          conversationId={conversationId}
          agentProfileId={ambassadorProfile?.id}
          agentName={convAgentName}
          ambassadorName={ambassadorProfile?.name}
          activeConversation={activeConversation}
          onRelay={relay}
          onAnswerPersisted={onAnswerPersisted}
        />
      ) : (
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
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent text-fg-inverse shadow-sm transition hover:brightness-110 active:brightness-95 disabled:opacity-40"
              title={mode === 'ask' ? 'Ask the ambassador' : 'Send to the conversation'}
            >
              {mode === 'ask' ? <Send size={14} /> : <CornerUpRight size={14} />}
            </button>
          </div>

          {/* One stable helper line — flash takes over briefly, no layout shift. */}
          <p className="px-0.5 text-[11px] leading-snug text-fg-muted">{footerHelp}</p>
        </div>
      ))}
    </div>
  );
}

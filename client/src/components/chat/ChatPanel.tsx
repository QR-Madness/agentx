/**
 * ChatPanel — Core chat UI for rendering conversations
 * Consumes active tab from ConversationContext and handles messaging
 */

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Send,
  Bot,
  Square,
  ChevronUp,
  ChevronDown,
  Layers,
  Sparkles,
  Workflow as WorkflowIcon,
  Crown,
  Box,
  Database,
  DatabaseZap,
  Cpu,
  X,
  ArrowRightLeft,
  PanelLeftOpen,
} from 'lucide-react';
import { api } from '../../lib/api';
import { RelayMenu } from './relay/RelayMenu';
import { MessageContent } from './MessageContent';
import { ThinkingBubble } from './ThinkingBubble';
import { MessageBubble } from './MessageBubble';
import { StepGroup } from './StepGroup';
import { groupMessagesBySteps } from './groupMessagesBySteps';
import { AgentSelectorDropdown } from './AgentSelectorDropdown';
import { CheckpointsBadge } from './CheckpointsBadge';
import { useConversation } from '../../contexts/ConversationContext';
import { usePlans } from '../../contexts/PlansContext';
import { useNotify } from '../../contexts/NotificationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import { getActiveMention, applyMention, extractMentionedAgentIds } from '../../lib/mentions';
import { MentionAutocomplete } from './MentionAutocomplete';
import {
  type UserMessage,
  type PlanExecutionMessage,
  type ExhibitMessage,
  type AssistantMessage,
  createMessageId,
  stripThinkingTags,
} from '../../lib/messages';
import type { Exhibit } from '../../lib/exhibits';
import {
  RESUME_CONFIRM,
  RESUME_DISMISS,
  resumeExhibitId,
  isResumeExhibitId,
  planIdFromResumeExhibit,
  buildResumeNudge,
  expiryLabel,
  ttlPhrase,
} from '../../lib/planResume';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useModal } from '../../contexts/ModalContext';
import { useIsMobile } from '../../lib/hooks';
import { SURFACES } from '../../lib/surfaces';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import { useOpenAmbassador } from '../../hooks/useOpenAmbassador';
import { latestRun } from '../../lib/alloyTrace';
import { useChatStream } from './useChatStream';
import { getMeta } from '../../lib/conversationMeta';
import { fetchModelsOnce } from '../common/modelCatalog';
import { ModelPickerModal } from '../common/ModelPickerModal';
import './ChatPanel.css';

export function ChatPanel() {
  const {
    activeTab,
    appendMessage,
    updateMessage,
    setStreaming,
    setSessionId,
    setTabContextInfo,
    setActiveTabModel,
    updateTab,
    restoreConversation,
    registerRelay,
  } = useConversation();
  const { activeProfile, profiles, getAgentName, getProfileById } = useAgentProfile();
  const { getWorkflowById } = useAlloyWorkflow();
  const { openModal } = useModal();
  const isMobile = useIsMobile();
  const { briefingForMessage, ask: askAmbassador } = useAmbassador();
  const openAmbassador = useOpenAmbassador();
  const { upsertPlan, patchPlan } = usePlans();
  const { notifyError } = useNotify();

  // When a workflow is selected, the supervisor profile takes over.
  // Otherwise, the tab's per-tab profile (or the global active profile) is used.
  // Most recent Alloy run in this tab — drives the "Run trace" affordance.
  const traceRun = useMemo(
    () => latestRun(activeTab?.messages ?? []),
    [activeTab?.messages],
  );
  const openRunTrace = useCallback(() => {
    if (!traceRun) return;
    openModal({
      id: 'alloy-run-trace',
      type: 'modal',
      component: 'alloyRunTrace',
      size: 'full',
      props: { runId: traceRun.id },
    });
  }, [traceRun, openModal]);

  // CC the Ambassador on a turn — like forwarding it as an email *into* the Inquiry.
  // The turn becomes a message in the ambassador's own thread (it never enters the
  // chat) and it briefs that turn there. The ambassador runs in *parallel*; this is
  // the per-turn path (the panel itself is conversation-level). Docks beside the chat
  // on a wide screen (both panels live); falls back to a sheet when too narrow.
  const handleAmbassador = useCallback(
    (message: AssistantMessage) => {
      const sessionId = activeTab?.sessionId;
      const turn = message.content.trim();
      if (sessionId && turn) {
        askAmbassador(sessionId, `Here's a turn from this conversation — brief me on it:\n\n"${turn}"`);
      }
      openAmbassador();
    },
    [activeTab?.sessionId, askAmbassador, openAmbassador],
  );

  const ambassadorStatusFor = useCallback(
    (messageId: string): 'idle' | 'streaming' | 'done' | 'error' => {
      const b = briefingForMessage(activeTab?.sessionId, messageId);
      if (!b) return 'idle';
      if (b.status === 'streaming') return 'streaming';
      if (b.status === 'done') return 'done';
      if (b.status === 'error' || b.status === 'empty_provider') return 'error';
      return 'idle';
    },
    [briefingForMessage, activeTab?.sessionId],
  );

  const activeWorkflow = activeTab?.workflowId
    ? getWorkflowById(activeTab.workflowId)
    : null;
  const supervisorProfile = activeWorkflow
    ? profiles.find(p => p.agentId === activeWorkflow.supervisorAgentId) ?? null
    : null;
  const tabProfile = supervisorProfile
    ? supervisorProfile
    : activeTab?.profileId
      ? getProfileById(activeTab.profileId)
      : activeProfile;

  const [input, setInput] = useState('');
  // @-mention autocomplete state (16.5 client). span = the @token being typed.
  const [mention, setMention] = useState<{
    open: boolean;
    query: string;
    span: { start: number; end: number } | null;
    highlight: number;
  }>({ open: false, query: '', span: null, highlight: 0 });
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  // `isPinned` tracks whether the user is scrolled to (or near) the bottom
  // of the message list. We auto-scroll only when pinned — so streaming
  // doesn't yank the viewport away from a user reading older messages.
  const [isPinned, setIsPinned] = useState(true);
  const contextInfo = activeTab?.contextInfo ?? null;
  const [showRelay, setShowRelay] = useState(false);
  const [hasUnreadBgJobs, setHasUnreadBgJobs] = useState(false);
  // Bumped each time the agent saves a checkpoint mid-stream; the badge
  // watches this to refetch + flash.
  const [checkpointSignal, setCheckpointSignal] = useState(0);
  // When armed (via Relay), the next send routes to the background queue.
  const [bgArmed, setBgArmed] = useState(false);
  // Inline composer chips: effective model + whether the memory toggle is
  // still changeable (locks once the conversation has started).
  const effectiveModel = activeTab?.modelOverride || tabProfile?.defaultModel || '';
  const modelLabel = effectiveModel
    ? effectiveModel.split(/[:/]/).pop() || effectiveModel
    : 'Default model';
  const canToggleMemory =
    !!activeTab && activeTab.messages.length === 0 && !activeTab.sessionId;
  const useMemory = !(activeTab?.noMemorization ?? false);
  const setNoMemorization = useCallback(
    (next: boolean) => {
      if (activeTab) updateTab(activeTab.id, { noMemorization: next });
    },
    [activeTab, updateTab],
  );
  const agentName = supervisorProfile?.name ?? getAgentName();

  const resolveAgentName = useCallback(
    (agentId: string) => profiles.find(p => p.agentId === agentId)?.name,
    [profiles],
  );

  const stream = useChatStream({
    appendMessage,
    updateMessage,
    agentName,
    resolveAgentName,
    onSessionId: setSessionId,
    onContextInfo: (info) => {
      if (activeTab) setTabContextInfo(activeTab.id, info);
    },
    tabId: activeTab?.id,
    tabTitle: activeTab?.title,
    plans: { upsertPlan, patchPlan },
    onRunChanged: (runId) => {
      if (activeTab) updateTab(activeTab.id, { activeRun: runId ? { runId } : undefined });
    },
    onRunMissing: () => {
      // The run's event buffer expired but its turns are persisted — pull the
      // finished conversation from server history.
      if (activeTab?.sessionId) restoreConversation(activeTab.sessionId).catch(() => {});
    },
    onCheckpointSaved: () => setCheckpointSignal(n => n + 1),
  });

  const isTyping = stream.state.phase === 'streaming';
  const streamingContent = stream.state.liveContent;
  const activeDelegationCount = stream.state.activeDelegations.size;

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const profileButtonRef = useRef<HTMLButtonElement>(null);
  const relayButtonRef = useRef<HTMLButtonElement>(null);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
    setIsPinned(true);
  }, []);

  // Track pin state from the scroll container. We treat "within 24px of the
  // bottom" as pinned — small enough to feel right, big enough to survive
  // rounding from content height changes during streaming.
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      setIsPinned(distance < 24);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Auto-scroll only while pinned. Streaming chunks update both messages
  // and liveContent, both of which we want to follow when the user is at
  // the bottom; when they've scrolled up we leave the viewport alone.
  useEffect(() => {
    if (isPinned) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeTab?.messages, streamingContent, isPinned]);

  // Detach (not cancel) the stream when switching tabs / unmounting. The run
  // keeps executing server-side and is re-attached below when the tab is shown
  // again — closing/switching a tab must never kill an in-flight conversation.
  useEffect(() => {
    return () => {
      stream.detach();
    };
  }, [activeTab?.id, stream.detach]);

  // Resume an in-flight detached run when this tab is shown. Truncate the
  // transcript back to the triggering user turn first so the replay-from-0
  // rebuilds the assistant side without duplicating already-rendered messages.
  const attachedRunRef = useRef<string | null>(null);
  const activeRunId = activeTab?.activeRun?.runId;
  useEffect(() => {
    if (!activeTab || !activeRunId) return;
    if (isTyping) return;                         // already streaming (e.g. just sent)
    if (attachedRunRef.current === activeRunId) return;
    attachedRunRef.current = activeRunId;

    const msgs = activeTab.messages;
    let lastUser = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].type === 'user') { lastUser = i; break; }
    }
    if (lastUser >= 0 && lastUser < msgs.length - 1) {
      updateTab(activeTab.id, { messages: msgs.slice(0, lastUser + 1) });
    }
    stream.attach(activeRunId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab?.id, activeRunId]);

  // Mirror streaming phase into ConversationContext for cross-component awareness
  useEffect(() => {
    setStreaming(stream.state.phase === 'streaming');
  }, [stream.state.phase, setStreaming]);

  // Backfill the context-window indicator for conversations that were saved
  // before context_window/used was persisted on assistant turns, or that
  // were rehydrated from localStorage (the runtime contextInfo is stripped
  // on save). Uses the latest assistant message's model to look up the
  // window, and estimates `used` from message char count if no tokens were
  // recorded.
  useEffect(() => {
    if (!activeTab || activeTab.contextInfo) return;
    const msgs = activeTab.messages;
    if (!msgs.length) return;

    let modelId: string | undefined;
    let usedTokens: number | undefined;
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i];
      if (m.type !== 'assistant') continue;
      modelId = m.model;
      if (m.tokensInput !== undefined) {
        usedTokens = m.tokensInput + (m.tokensOutput ?? 0);
      }
      if (modelId) break;
    }
    if (!modelId) return;

    let cancelled = false;
    fetchModelsOnce().then((models) => {
      if (cancelled || !activeTab) return;
      const info = models.find((mm) => mm.id === modelId);
      const window = info?.context_window ?? info?.context_length;
      if (!window) return;
      const used =
        usedTokens ??
        Math.ceil(
          msgs.reduce(
            (n, m) =>
              n + (m.type === 'user' || m.type === 'assistant' ? m.content.length : 0),
            0,
          ) / 4,
        );
      setTabContextInfo(activeTab.id, { window, used });
    });
    return () => {
      cancelled = true;
    };
  }, [activeTab, setTabContextInfo]);

  // Profiles matching the active @query (by name or agent_id), like the selector.
  const mentionItems = useMemo(() => {
    if (!mention.open) return [];
    const q = mention.query.toLowerCase();
    return profiles.filter(
      p =>
        p.kind !== 'ambassador' &&  // ambassadors aren't routable chat agents
        (p.name.toLowerCase().includes(q) ||
          p.agentId.toLowerCase().includes(q) ||
          (p.tags ?? []).some(t => t.toLowerCase().includes(q))),
    );
  }, [mention.open, mention.query, profiles]);

  // Recompute the active @-mention from the caret; opens/filters or closes the popover.
  const refreshMention = (text: string, caret: number) => {
    const active = getActiveMention(text, caret);
    setMention(m =>
      active
        ? { open: true, query: active.query, span: { start: active.start, end: active.end }, highlight: 0 }
        : (m.open ? { open: false, query: '', span: null, highlight: 0 } : m),
    );
  };

  const closeMention = () =>
    setMention({ open: false, query: '', span: null, highlight: 0 });

  // Insert the picked agent's display name (friendly, shown verbatim in the
  // textarea). The backend (16.5) resolves a single-token @name to its agent.
  // Fall back to the agent_id slug only when the name isn't a single routable
  // token (e.g. contains a space) so routing always works.
  const pickMention = (profile: { name: string; agentId: string }) => {
    if (!mention.span) return;
    const token = /^[\w-]+$/.test(profile.name) ? profile.name : profile.agentId;
    const { text, caret } = applyMention(input, mention.span, token);
    setInput(text);
    closeMention();
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (el) { el.focus(); el.setSelectionRange(caret, caret); }
    });
  };

  const handleSend = async () => {
    if (!input.trim() || !activeTab) return;

    const userMessage: UserMessage = {
      id: createMessageId(),
      type: 'user',
      content: input,
      timestamp: new Date().toISOString(),
      targetAgentIds: extractMentionedAgentIds(input, profiles),
    };

    appendMessage(userMessage);
    const messageText = input;
    setInput('');
    closeMention();

    stream.send({
      message: messageText,
      session_id: activeTab.sessionId || undefined,
      agent_profile_id: tabProfile?.id,
      model: activeTab.modelOverride || undefined,
      use_memory: useMemory,
      workflow_id: activeTab.workflowId || undefined,
      workspace_id: getMeta(activeTab.sessionId ?? activeTab.id).workspaceId || undefined,
    });
  };

  // Submit a choice-element selection as the next user turn. Stable across
  // composer keystrokes (no `input` in deps) so the memoized transcript — and
  // mermaid renders — don't churn while the user types. Mirrors handleSend's
  // full request shape (incl. workflow_id) and guards against an in-flight run.
  const submitChoice = useCallback(
    (value: string, messageId: string) => {
      if (!activeTab || isTyping) return;

      // Resume-plan exhibit: "Dismiss" just resolves the card; "Resume plan"
      // re-checks Redis, then nudges the model with the done/remaining steps
      // (no PlanExecutor re-run — the agent continues the work itself).
      const src = activeTab.messages.find(m => m.id === messageId);
      if (src?.type === 'exhibit' && isResumeExhibitId(src.exhibit.id)) {
        updateMessage(messageId, { answeredValue: value });
        if (value === RESUME_DISMISS || !activeTab.sessionId) return;
        const planId = planIdFromResumeExhibit(src.exhibit.id);
        const sessionId = activeTab.sessionId;
        api.getPlanStatus(planId, sessionId)
          .then(res => {
            if (!res.found || !res.resumable) {
              notifyError('That plan can no longer be resumed (it expired or finished).');
              return;
            }
            const nudge = buildResumeNudge(res);
            appendMessage({
              id: createMessageId(), type: 'user', content: nudge,
              timestamp: new Date().toISOString(),
            });
            stream.send({
              message: nudge,
              session_id: sessionId,
              agent_profile_id: tabProfile?.id,
              model: activeTab.modelOverride || undefined,
              use_memory: useMemory,
              workflow_id: activeTab.workflowId || undefined,
              workspace_id: getMeta(activeTab.sessionId ?? activeTab.id).workspaceId || undefined,
            });
          })
          .catch(() => notifyError('Could not check the plan status.'));
        return;
      }

      const userMessage: UserMessage = {
        id: createMessageId(),
        type: 'user',
        content: value,
        timestamp: new Date().toISOString(),
      };
      appendMessage(userMessage);
      updateMessage(messageId, { answeredValue: value });
      stream.send({
        message: value,
        session_id: activeTab.sessionId || undefined,
        agent_profile_id: tabProfile?.id,
        model: activeTab.modelOverride || undefined,
        use_memory: useMemory,
        workflow_id: activeTab.workflowId || undefined,
        workspace_id: getMeta(activeTab.sessionId ?? activeTab.id).workspaceId || undefined,
      });
    },
    [activeTab, isTyping, tabProfile?.id, useMemory, appendMessage, updateMessage, stream.send, notifyError],
  );

  // Outbound relay from another surface (the Ambassador panel): fold the message
  // into the running turn if one is streaming, else start a fresh user turn.
  // Mirrors handleSend/handleSteer; the relayed text is a real *user* message.
  const relayMessage = useCallback(
    (text: string) => {
      const t = text.trim();
      if (!t || !activeTab) return;
      if (isTyping) {
        stream.steer(t);
        return;
      }
      appendMessage({
        id: createMessageId(),
        type: 'user',
        content: t,
        timestamp: new Date().toISOString(),
        targetAgentIds: extractMentionedAgentIds(t, profiles),
      });
      stream.send({
        message: t,
        session_id: activeTab.sessionId || undefined,
        agent_profile_id: tabProfile?.id,
        model: activeTab.modelOverride || undefined,
        use_memory: useMemory,
        workflow_id: activeTab.workflowId || undefined,
        workspace_id: getMeta(activeTab.sessionId ?? activeTab.id).workspaceId || undefined,
      });
    },
    [activeTab, isTyping, stream.steer, stream.send, appendMessage, profiles, tabProfile?.id, useMemory],
  );

  // Register this tab's relay handler so the Ambassador panel can reach it.
  useEffect(() => {
    if (!activeTab) return;
    return registerRelay(activeTab.id, relayMessage);
  }, [activeTab?.id, relayMessage, registerRelay]);

  const handleSendBackground = async () => {
    if (!input.trim() || !activeTab) return;
    const messageText = input;
    setInput('');
    try {
      await api.enqueueBackgroundChat({
        message: messageText,
        session_id: activeTab.sessionId || undefined,
        agent_profile_id: tabProfile?.id,
        model: activeTab.modelOverride || undefined,
        use_memory: useMemory,
        workflow_id: activeTab.workflowId || undefined,
      });
      setHasUnreadBgJobs(true);
    } catch (err) {
      console.error('Failed to enqueue background chat:', err);
      notifyError(err, 'Failed to queue background message');
    }
  };

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, []);

  useEffect(() => {
    autoResize();
  }, [input, autoResize]);

  // Handle prompt enhancement
  const handleEnhancePrompt = async () => {
    if (!input.trim() || isEnhancing) return;

    setIsEnhancing(true);
    try {
      // Build context from recent messages
      const context = messages.slice(-5).map(msg => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.type === 'user' || msg.type === 'assistant' ? msg.content : '',
      })).filter(msg => msg.content);

      const result = await api.enhancePrompt(input, context);
      setInput(result.enhanced_prompt);
    } catch (error) {
      console.error('Failed to enhance prompt:', error);
    } finally {
      setIsEnhancing(false);
    }
  };

  // Steer the running turn: fold the typed message into the in-flight run
  // instead of starting a new one. The steer bubble is appended when the server
  // echoes the `steer` event back, so we just clear the composer here.
  const handleSteer = () => {
    if (!input.trim()) return;
    stream.steer(input);
    setInput('');
    closeMention();
  };

  // Unified submit: steer while a turn streams, else background-queue when
  // armed, else start a new streaming turn.
  const submit = () => {
    if (!input.trim()) return;
    if (isTyping) {
      handleSteer();
    } else if (bgArmed) {
      handleSendBackground();
      setBgArmed(false);
    } else {
      handleSend();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // When the @-mention popover is open, it owns navigation keys.
    if (mention.open && mentionItems.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setMention(m => ({ ...m, highlight: Math.min(m.highlight + 1, mentionItems.length - 1) }));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setMention(m => ({ ...m, highlight: Math.max(m.highlight - 1, 0) }));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        const picked = mentionItems[mention.highlight];
        if (picked) pickMention(picked);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        closeMention();
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // Group messages by plan subtask so each step's output renders under a
  // collapsible "Step k/n" header. Memoized on the message array.
  const tabMessages = activeTab?.messages;
  const groupedItems = useMemo(
    () => groupMessagesBySteps(tabMessages ?? []),
    [tabMessages],
  );
  // planId → whether the plan is still running (drives default-collapse).
  const planRunningById = useMemo(() => {
    const map = new Map<string, boolean>();
    for (const m of tabMessages ?? []) {
      if (m.type === 'plan_execution') map.set(m.planId, m.status === 'running');
    }
    return map;
  }, [tabMessages]);

  // Listen for "jump to step" from the Plans drawer: scroll to the step group
  // (or plan card), expand it, and flash it.
  const activeTabId = activeTab?.id;
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ tabId: string; planId: string; subtaskId?: number }>).detail;
      if (!detail || detail.tabId !== activeTabId) return;
      const selector =
        detail.subtaskId != null
          ? `[data-step-anchor="${CSS.escape(`${detail.planId}:${detail.subtaskId}`)}"]`
          : `[data-plan-anchor="${CSS.escape(detail.planId)}"]`;
      // Defer one frame so a freshly-switched tab has rendered.
      requestAnimationFrame(() => {
        const el = container.querySelector<HTMLElement>(selector);
        if (!el) return;
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('flash-target');
        setIsPinned(false);
        setTimeout(() => el.classList.remove('flash-target'), 1500);
      });
    };
    window.addEventListener('agentx:jump-to-step', handler);
    return () => window.removeEventListener('agentx:jump-to-step', handler);
  }, [activeTabId]);

  // Auto-offer to resume an interrupted plan: when this conversation has a plan
  // card still 'running' whose Redis snapshot is still resumable, append a
  // one-time `choice` exhibit (id exh_resume_{planId}) nudging the user to
  // continue it. The exhibit itself (persisted, possibly answered) is the dedupe
  // — once offered it's never re-added.
  const resumeCheckedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const tab = activeTab;
    if (!tab?.sessionId || isTyping) return;
    const candidatePlanIds = Array.from(new Set(
      tab.messages
        .filter((m): m is PlanExecutionMessage =>
          m.type === 'plan_execution' && (m.status === 'running' || m.status === 'interrupted'))
        .map(m => m.planId),
    ));
    for (const planId of candidatePlanIds) {
      const key = `${tab.id}:${planId}`;
      if (resumeCheckedRef.current.has(key)) continue;
      const alreadyOffered = tab.messages.some(
        m => m.type === 'exhibit' && m.exhibit.id === resumeExhibitId(planId),
      );
      resumeCheckedRef.current.add(key);
      if (alreadyOffered) continue;
      api.getPlanStatus(planId, tab.sessionId)
        .then(res => {
          if (!res.found || !res.resumable) return;
          const total = res.subtask_count ?? res.subtasks?.length ?? 0;
          const done = res.completed_count ?? 0;
          const until = expiryLabel(res.ttl_seconds);
          const phrase = ttlPhrase(res.ttl_seconds);
          const lifetime = until ? ` It stays resumable for ${phrase} (until ${until}).` : '';
          const exhibit: Exhibit = {
            schemaVersion: 1,
            id: resumeExhibitId(planId),
            title: 'Interrupted plan',
            layout: 'stack',
            elements: [{
              type: 'choice',
              title: 'Interrupted plan',
              prompt: `An earlier plan was interrupted (${done}/${total} steps done).${lifetime} Resume it?`,
              options: [RESUME_CONFIRM, RESUME_DISMISS],
            }],
          };
          const msg: ExhibitMessage = {
            id: createMessageId(), type: 'exhibit',
            timestamp: new Date().toISOString(), exhibit,
          };
          appendMessage(msg);
        })
        .catch(() => { /* best-effort nudge */ });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab?.id, activeTab?.sessionId, activeTab?.messages, isTyping]);

  if (!activeTab) {
    return (
      <div className="chat-panel-empty">
        <Bot size={48} />
        <p>Select or create a conversation to start chatting</p>
        {isMobile && (
          <button
            type="button"
            className="chat-panel-empty-conv"
            onClick={() => openModal(SURFACES.conversations)}
          >
            <PanelLeftOpen size={16} />
            <span>Show conversations</span>
          </button>
        )}
      </div>
    );
  }

  const messages = activeTab.messages;

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-panel-header">
        <div className="chat-panel-info">
          {/* Mobile: the desktop Conversations rail is hidden <600px, so open it
              as a drawer from here (the command palette is the other entry). */}
          {isMobile && (
            <button
              type="button"
              className="chat-panel-conv-toggle"
              onClick={() => openModal(SURFACES.conversations)}
              title="Show conversations"
              aria-label="Show conversations"
            >
              <PanelLeftOpen size={18} />
            </button>
          )}
          <span className="chat-panel-title">{activeTab.title}</span>
          {activeTab.sessionId && (
            <span className="chat-panel-session">
              Session: {activeTab.sessionId.slice(0, 8)}...
            </span>
          )}
          {contextInfo && (
            <div className="context-usage" title={`${contextInfo.used.toLocaleString()} / ${contextInfo.window.toLocaleString()} tokens`}>
              <Layers size={12} />
              <div className="context-bar">
                <div
                  className="context-bar-fill"
                  style={{ width: `${Math.min((contextInfo.used / contextInfo.window) * 100, 100)}%` }}
                />
              </div>
              <span className="context-label">
                {(contextInfo.used / 1000).toFixed(1)}k / {(contextInfo.window / 1000).toFixed(0)}k
              </span>
            </div>
          )}
          <CheckpointsBadge
            conversationId={activeTab.sessionId}
            flashSignal={checkpointSignal}
          />
          {traceRun && (
            <button
              type="button"
              className="run-trace-badge"
              onClick={openRunTrace}
              title="View Alloy run trace"
            >
              <ArrowRightLeft size={12} />
              <span>Trace</span>
              <span className="run-trace-count">{traceRun.totals.count}</span>
            </button>
          )}
        </div>
      </div>

      {activeTab.noMemorization && (
        <div className="no-memo-banner" role="status">
          <Database size={14} />
          <span>
            <strong>No Memorization is on.</strong> This conversation will not be
            stored or recalled — treat its contents as ephemeral and avoid
            relying on continuity later.
          </span>
        </div>
      )}

      {/* Messages */}
      <div className="chat-panel-messages" ref={messagesContainerRef}>
        {messages.length === 0 && (
          <div className="chat-panel-welcome">
            <Bot size={32} />
            <p>Start a conversation by typing a message below</p>
          </div>
        )}

        <div className="chat-thread">
        {groupedItems.map((item) => {
          if (item.kind === 'stepGroup') {
            return (
              <StepGroup
                key={item.key}
                step={item.step}
                messages={item.messages}
                agentName={agentName}
                avatarId={tabProfile?.avatar}
                defaultCollapsed={!planRunningById.get(item.step.planId)}
                onSubmitChoice={submitChoice}
                busy={isTyping}
              />
            );
          }
          const { message } = item;
          // The plan_execution card is the jump target for plan-level focus.
          if (message.type === 'plan_execution') {
            return (
              <div key={message.id} data-plan-anchor={message.planId}>
                <MessageBubble message={message} agentName={agentName} avatarId={tabProfile?.avatar} onSubmitChoice={submitChoice} busy={isTyping} />
              </div>
            );
          }
          return (
            <MessageBubble
              key={message.id}
              message={message}
              agentName={agentName}
              avatarId={tabProfile?.avatar}
              onSubmitChoice={submitChoice}
              onAmbassador={handleAmbassador}
              ambassadorStatus={message.type === 'assistant' ? ambassadorStatusFor(message.id) : undefined}
              busy={isTyping}
            />
          );
        })}

        {/* Streaming message or typing indicator. Suppress the empty-state
            spinner while a delegation card is actively streaming — the card
            is the source of activity, the main bubble would just look stalled. */}
        {isTyping && (streamingContent || activeDelegationCount === 0) && (() => {
          const AvatarIcon = getAvatarIcon(tabProfile?.avatar);
          return (
          <div className="message-bubble assistant">
            <div className="message-avatar assistant-avatar">
              <AvatarIcon size={16} />
            </div>
            <div className="message-body">
              {streamingContent ? (
                <div className="streaming-message">
                  {(() => {
                    // Find all thinking blocks and show the last one (active during streaming)
                    const thinkMatches = [
                      ...streamingContent.matchAll(/<think(?:ing)?>([\s\S]*?)(?:<\/think(?:ing)?>|$)/gi)
                    ];
                    const lastMatch = thinkMatches[thinkMatches.length - 1];
                    return lastMatch ? (
                      <ThinkingBubble thinking={lastMatch[1]} isStreaming />
                    ) : null;
                  })()}
                  <MessageContent content={stripThinkingTags(streamingContent, true)} />
                </div>
              ) : (
                <div className="stream-spinner">
                  <div className="stream-spinner-ring" />
                  <span className="stream-spinner-text">
                    {stream.state.activity?.label || 'Thinking...'}
                  </span>
                </div>
              )}
            </div>
          </div>
          );
        })()}

        <div ref={messagesEndRef} />
        </div>

        {/* Jump-to-latest affordance — visible only when the user has
            scrolled away from the bottom. Clicking re-pins and follows
            new messages again. */}
        {!isPinned && (
          <button
            className="auto-scroll-toggle"
            onClick={() => scrollToBottom('smooth')}
            title="Jump to latest"
            aria-label="Jump to latest message"
          >
            <ChevronDown size={16} />
          </button>
        )}
      </div>

      {/* Input */}
      <div className="chat-panel-input">
        {bgArmed && (
          <div className="bg-armed-chip">
            <Box size={12} />
            <span>Next message runs in the background</span>
            <button
              onClick={() => setBgArmed(false)}
              aria-label="Cancel background mode"
              title="Cancel background mode"
            >
              <X size={12} />
            </button>
          </div>
        )}
        <div className="input-controls">
          <button
            ref={profileButtonRef}
            className={`profile-indicator ${showAgentSelector ? 'active' : ''}`}
            onClick={() => setShowAgentSelector(!showAgentSelector)}
            title="Select agent profile"
          >
            {activeWorkflow ? <Crown size={12} /> : <Sparkles size={12} />}
            <span>{tabProfile?.name || 'Select Agent'}</span>
            {activeWorkflow && (
              <span
                className="profile-indicator-workflow"
                title={`Alloy workflow: ${activeWorkflow.name}`}
              >
                <WorkflowIcon size={10} />
                <span>{activeWorkflow.name}</span>
              </span>
            )}
            <ChevronUp size={10} className={showAgentSelector ? 'rotated' : ''} />
          </button>
          <AgentSelectorDropdown
            isOpen={showAgentSelector}
            onClose={() => setShowAgentSelector(false)}
            anchorRef={profileButtonRef}
          />

          {/* Inline model chip — per-conversation model override */}
          <button
            className={`composer-chip ${activeTab?.modelOverride ? 'active' : ''}`}
            onClick={() => setShowModelPicker(true)}
            title={activeTab?.modelOverride ? 'Model (overridden for this chat)' : 'Model (from profile) — click to override'}
          >
            <Cpu size={12} />
            <span>{modelLabel}</span>
          </button>

          {/* Memory toggle chip — locks once the conversation has started */}
          <button
            className={`composer-chip ${activeTab?.noMemorization ? 'warn' : ''}`}
            onClick={() => canToggleMemory && setNoMemorization(!(activeTab?.noMemorization ?? false))}
            disabled={!canToggleMemory}
            title={
              activeTab?.noMemorization
                ? 'No Memorization is on — this chat is ephemeral'
                : canToggleMemory
                  ? 'Memorization on — click to make this chat ephemeral'
                  : 'Memorization locked once the conversation has started'
            }
          >
            {activeTab?.noMemorization ? <DatabaseZap size={12} /> : <Database size={12} />}
            <span>{activeTab?.noMemorization ? 'No memory' : 'Memory'}</span>
          </button>
        </div>
        <div className="input-container">
          <button
            ref={relayButtonRef}
            className={`relay-trigger ${showRelay ? 'active' : ''}`}
            onClick={() => {
              setShowRelay(v => !v);
              if (!showRelay) setHasUnreadBgJobs(false);
            }}
            title="Relay Module"
            aria-label="Open Relay menu"
          >
            <Box size={18} />
            {hasUnreadBgJobs && !showRelay && <span className="relay-trigger-badge" />}
          </button>
          <RelayMenu
            isOpen={showRelay}
            onClose={() => setShowRelay(false)}
            anchorRef={relayButtonRef}
            canEnhance={!!input.trim() && !isTyping}
            onEnhance={handleEnhancePrompt}
            isEnhancing={isEnhancing}
            canArmBackground={!!input.trim() || bgArmed}
            backgroundArmed={bgArmed}
            onToggleBackground={() => setBgArmed(v => !v)}
            onJobsChanged={() => setHasUnreadBgJobs(false)}
          />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              refreshMention(e.target.value, e.target.selectionStart ?? e.target.value.length);
            }}
            onClick={(e) => refreshMention(input, e.currentTarget.selectionStart ?? input.length)}
            onKeyDown={handleKeyDown}
            placeholder={
              isTyping
                ? 'Steer the running agent... (Shift+Enter for new line)'
                : 'Type your message... (Shift+Enter for new line)'
            }
            rows={1}
          />
          <MentionAutocomplete
            isOpen={mention.open}
            items={mentionItems}
            highlight={mention.highlight}
            anchorRef={textareaRef}
            onHover={(i) => setMention(m => ({ ...m, highlight: i }))}
            onPick={(p) => pickMention(p)}
            onClose={closeMention}
          />
          {isTyping ? (
            <>
              <button
                className="stop-button"
                onClick={() => {
                  stream.stop();
                  setStreaming(false);
                }}
                title="Stop generating"
              >
                <Square size={16} />
              </button>
              <button
                className="send-button steer"
                onClick={submit}
                disabled={!input.trim()}
                title="Steer the running agent"
              >
                <Send size={18} />
              </button>
            </>
          ) : (
            <button
              className={`send-button ${bgArmed ? 'armed' : ''}`}
              onClick={submit}
              disabled={!input.trim()}
              title={bgArmed ? 'Send to background' : 'Send'}
            >
              {bgArmed ? <Box size={18} /> : <Send size={18} />}
            </button>
          )}
        </div>
        <div className="input-stats">
          <span className={input.length > 4000 ? 'warning' : ''}>
            {input.length} chars · ~{Math.ceil(input.length / 4)} tokens
          </span>
        </div>
      </div>

      <ModelPickerModal
        isOpen={showModelPicker}
        onClose={() => setShowModelPicker(false)}
        value={activeTab?.modelOverride || ''}
        onChange={(modelId) => setActiveTabModel(modelId || null)}
        showDefault
      />
    </div>
  );
}


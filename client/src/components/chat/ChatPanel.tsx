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
} from 'lucide-react';
import { api } from '../../lib/api';
import { RelayMenu } from './relay/RelayMenu';
import { MessageContent } from './MessageContent';
import { ThinkingBubble } from './ThinkingBubble';
import { MessageBubble } from './MessageBubble';
import { StepGroup } from './StepGroup';
import { groupMessagesBySteps } from './groupMessagesBySteps';
import { AgentSelectorDropdown } from './AgentSelectorDropdown';
import { useConversation } from '../../contexts/ConversationContext';
import { usePlans } from '../../contexts/PlansContext';
import { useNotify } from '../../contexts/NotificationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import {
  type UserMessage,
  createMessageId,
  stripThinkingTags,
} from '../../lib/messages';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useChatStream } from './useChatStream';
import { fetchModelsOnce } from '../common/ModelSelector';
import './ChatPanel.css';

export function ChatPanel() {
  const {
    activeTab,
    appendMessage,
    updateMessage,
    setStreaming,
    setSessionId,
    setTabContextInfo,
    updateTab,
    restoreConversation,
  } = useConversation();
  const { activeProfile, profiles, getAgentName, getProfileById } = useAgentProfile();
  const { getWorkflowById } = useAlloyWorkflow();
  const { upsertPlan, patchPlan } = usePlans();
  const { notifyError } = useNotify();

  // When a workflow is selected, the supervisor profile takes over.
  // Otherwise, the tab's per-tab profile (or the global active profile) is used.
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
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  // `isPinned` tracks whether the user is scrolled to (or near) the bottom
  // of the message list. We auto-scroll only when pinned — so streaming
  // doesn't yank the viewport away from a user reading older messages.
  const [isPinned, setIsPinned] = useState(true);
  const contextInfo = activeTab?.contextInfo ?? null;
  const [showRelay, setShowRelay] = useState(false);
  const [hasUnreadBgJobs, setHasUnreadBgJobs] = useState(false);
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

  const handleSend = async () => {
    if (!input.trim() || !activeTab) return;

    const userMessage: UserMessage = {
      id: createMessageId(),
      type: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    appendMessage(userMessage);
    const messageText = input;
    setInput('');

    stream.send({
      message: messageText,
      session_id: activeTab.sessionId || undefined,
      agent_profile_id: tabProfile?.id,
      use_memory: useMemory,
      workflow_id: activeTab.workflowId || undefined,
    });
  };

  const handleSendBackground = async () => {
    if (!input.trim() || !activeTab) return;
    const messageText = input;
    setInput('');
    try {
      await api.enqueueBackgroundChat({
        message: messageText,
        session_id: activeTab.sessionId || undefined,
        agent_profile_id: tabProfile?.id,
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
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

  if (!activeTab) {
    return (
      <div className="chat-panel-empty">
        <Bot size={48} />
        <p>Select or create a conversation to start chatting</p>
      </div>
    );
  }

  const messages = activeTab.messages;

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-panel-header">
        <div className="chat-panel-info">
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
              />
            );
          }
          const { message } = item;
          // The plan_execution card is the jump target for plan-level focus.
          if (message.type === 'plan_execution') {
            return (
              <div key={message.id} data-plan-anchor={message.planId}>
                <MessageBubble message={message} agentName={agentName} avatarId={tabProfile?.avatar} />
              </div>
            );
          }
          return (
            <MessageBubble key={message.id} message={message} agentName={agentName} avatarId={tabProfile?.avatar} />
          );
        })}

        {/* Streaming message or typing indicator. Suppress the empty-state
            spinner while a delegation card is actively streaming — the card
            is the source of activity, the main bubble would just look stalled. */}
        {isTyping && (streamingContent || activeDelegationCount === 0) && (() => {
          const AvatarIcon = getAvatarIcon(tabProfile?.avatar);
          return (
          <div className="message-bubble assistant">
            <div className="message-avatar">
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
                  <span className="stream-spinner-text">Thinking...</span>
                </div>
              )}
            </div>
          </div>
          );
        })()}

        <div ref={messagesEndRef} />

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
            noMemorization={activeTab?.noMemorization ?? false}
            onToggleNoMemorization={() =>
              setNoMemorization(!(activeTab?.noMemorization ?? false))
            }
            canToggleNoMemorization={
              !!activeTab &&
              activeTab.messages.length === 0 &&
              !activeTab.sessionId
            }
            canEnhance={!!input.trim() && !isTyping}
            onEnhance={handleEnhancePrompt}
            isEnhancing={isEnhancing}
            canSendBackground={!!input.trim()}
            onSendBackground={handleSendBackground}
            onJobsChanged={() => setHasUnreadBgJobs(false)}
          />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Shift+Enter for new line)"
            rows={1}
            disabled={isTyping}
          />
          {isTyping ? (
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
          ) : (
            <button
              className="send-button"
              onClick={handleSend}
              disabled={!input.trim()}
            >
              <Send size={18} />
            </button>
          )}
        </div>
        <div className="input-stats">
          <span className={input.length > 4000 ? 'warning' : ''}>
            {input.length} chars · ~{Math.ceil(input.length / 4)} tokens
          </span>
        </div>
      </div>
    </div>
  );
}


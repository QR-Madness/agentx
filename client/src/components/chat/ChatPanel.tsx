/**
 * ChatPanel — Core chat UI for rendering conversations
 * Consumes active tab from ConversationContext and handles messaging
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send,
  Bot,
  Mic,
  Radio,
  Database,
  Square,
  ChevronUp,
  ChevronDown,
  Layers,
  Sparkles,
  Loader2,
  Workflow as WorkflowIcon,
  Crown,
} from 'lucide-react';
import { api, type ChatResponse } from '../../lib/api';
import { MessageContent } from './MessageContent';
import { ThinkingBubble } from './ThinkingBubble';
import { MessageBubble } from './MessageBubble';
import { AgentSelectorDropdown } from './AgentSelectorDropdown';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import {
  type UserMessage,
  type AssistantMessage,
  createMessageId,
} from '../../lib/messages';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useChatStream } from './useChatStream';
import './ChatPanel.css';

// Strip thinking tags from content
function stripThinkingTags(content: string, isStreaming = false): string {
  let result = content
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/\[thinking\][\s\S]*?\[\/thinking\]/gi, '')
    .replace(/\[think\][\s\S]*?\[\/think\]/gi, '')
    .replace(/<internal_monologue>[\s\S]*?<\/internal_monologue>/gi, '');

  if (isStreaming) {
    result = result
      .replace(/<thinking>[\s\S]*$/gi, '')
      .replace(/<think>[\s\S]*$/gi, '')
      .replace(/\[thinking\][\s\S]*$/gi, '')
      .replace(/\[think\][\s\S]*$/gi, '')
      .replace(/<internal_monologue>[\s\S]*$/gi, '');
  }

  return result.replace(/\n{3,}/g, '\n\n').trim();
}

export function ChatPanel() {
  const {
    activeTab,
    appendMessage,
    updateMessage,
    setStreaming,
    setSessionId,
  } = useConversation();
  const { activeProfile, profiles, getAgentName, getProfileById } = useAgentProfile();
  const { getWorkflowById } = useAlloyWorkflow();

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
  const [useStreamingMode, setUseStreamingMode] = useState(true);
  const [useMemory, setUseMemory] = useState(true);
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [contextInfo, setContextInfo] = useState<{
    window: number;
    used: number;
  } | null>(null);
  const [isNonStreamingTyping, setIsNonStreamingTyping] = useState(false);
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
    onContextInfo: setContextInfo,
  });

  const isTyping = stream.state.phase === 'streaming' || isNonStreamingTyping;
  const streamingContent = stream.state.liveContent;
  const activeDelegationCount = stream.state.activeDelegations.size;

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const profileButtonRef = useRef<HTMLButtonElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    if (autoScroll) {
      scrollToBottom();
    }
  }, [activeTab?.messages, streamingContent, scrollToBottom, autoScroll]);

  // Abort stream when switching tabs
  useEffect(() => {
    return () => {
      stream.stop();
    };
  }, [activeTab?.id, stream]);

  // Mirror streaming phase into ConversationContext for cross-component awareness
  useEffect(() => {
    setStreaming(stream.state.phase === 'streaming');
  }, [stream.state.phase, setStreaming]);

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

    if (useStreamingMode) {
      stream.send({
        message: messageText,
        session_id: activeTab.sessionId || undefined,
        agent_profile_id: tabProfile?.id,
        use_memory: useMemory,
        workflow_id: activeTab.workflowId || undefined,
      });
      return;
    }

    // Non-streaming mode
    setIsNonStreamingTyping(true);
    setStreaming(true);
    try {
      const response: ChatResponse = await api.chat({
        message: messageText,
        session_id: activeTab.sessionId || undefined,
        use_memory: useMemory,
      });

      if (response.session_id) setSessionId(response.session_id);

      const assistantMessage: AssistantMessage = {
        id: createMessageId(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        thinking: response.thinking,
        tokensUsed: response.tokens_used,
      };
      appendMessage(assistantMessage);
    } catch {
      const errorMessage: AssistantMessage = {
        id: createMessageId(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please check if the server is running.',
        timestamp: new Date().toISOString(),
      };
      appendMessage(errorMessage);
    } finally {
      setIsNonStreamingTyping(false);
      setStreaming(false);
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
        <div className="chat-panel-actions">
          <button
            className={`icon-button ${useMemory ? 'active' : ''}`}
            onClick={() => setUseMemory(!useMemory)}
            title={useMemory ? 'Memory enabled' : 'Memory disabled'}
          >
            <Database size={16} />
          </button>
          <button
            className={`icon-button ${useStreamingMode ? 'active' : ''}`}
            onClick={() => setUseStreamingMode(!useStreamingMode)}
            title={useStreamingMode ? 'Streaming enabled' : 'Streaming disabled'}
          >
            <Radio size={16} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-panel-messages">
        {messages.length === 0 && (
          <div className="chat-panel-welcome">
            <Bot size={32} />
            <p>Start a conversation by typing a message below</p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} agentName={agentName} avatarId={tabProfile?.avatar} />
        ))}

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

        {/* Auto-scroll toggle */}
        <button
          className={`auto-scroll-toggle ${autoScroll ? 'active' : ''}`}
          onClick={() => setAutoScroll(!autoScroll)}
          title={autoScroll ? 'Auto-scroll enabled (click to disable)' : 'Click to enable auto-scroll'}
        >
          <ChevronDown size={16} />
          {autoScroll && <span className="auto-indicator">AUTO</span>}
        </button>
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
          <button className="voice-button" disabled title="Voice input coming soon">
            <Mic size={18} />
          </button>
          <button
            className={`enhance-button ${isEnhancing ? 'enhancing' : ''}`}
            onClick={handleEnhancePrompt}
            disabled={!input.trim() || isEnhancing || isTyping}
            title="Enhance prompt"
          >
            {isEnhancing ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />}
          </button>
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
                setIsNonStreamingTyping(false);
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


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
  type ToolCallMessage,
  type MemoryInjectionMessage,
  type PlanExecutionMessage,
  type PlanSubtask,
  createMessageId,
} from '../../lib/messages';
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

// Extract all thinking content from text, concatenating multiple blocks
function extractThinking(content: string): string | null {
  const patterns = [
    /<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi,
    /\[think(?:ing)?\]([\s\S]*?)\[\/think(?:ing)?\]/gi,
    /<internal_monologue>([\s\S]*?)<\/internal_monologue>/gi,
  ];

  const thoughts: string[] = [];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(content)) !== null) {
      if (match[1]?.trim()) {
        thoughts.push(match[1].trim());
      }
    }
  }

  return thoughts.length > 0 ? thoughts.join('\n\n') : null;
}

export function ChatPanel() {
  const {
    activeTab,
    appendMessage,
    updateMessage,
    setStreaming,
    setSessionId,
  } = useConversation();
  const { activeProfile, getAgentName } = useAgentProfile();

  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [useStreamingMode, setUseStreamingMode] = useState(true);
  const [useMemory, setUseMemory] = useState(true);
  const [streamingContent, setStreamingContent] = useState('');
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [contextInfo, setContextInfo] = useState<{
    window: number;
    used: number;
  } | null>(null);
  const agentName = getAgentName();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<{ abort: () => void } | null>(null);
  const streamingContentRef = useRef('');
  const profileButtonRef = useRef<HTMLButtonElement>(null);
  // Track tool call message IDs for in-place updates
  const toolCallMessageIds = useRef<Map<string, string>>(new Map());
  // Track active plan execution for in-place updates
  const planMessageIdRef = useRef<string | null>(null);
  const planSubtasksRef = useRef<PlanSubtask[]>([]);

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
      streamAbortRef.current?.abort();
    };
  }, [activeTab?.id]);

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
    setIsTyping(true);
    setStreaming(true);

    if (useStreamingMode) {
      setStreamingContent('');
      streamingContentRef.current = '';

      streamAbortRef.current = api.streamChat(
        {
          message: messageText,
          session_id: activeTab.sessionId || undefined,
          agent_profile_id: activeProfile?.id,
          use_memory: useMemory,
        },
        {
          onStart: () => {
            // Stream started
          },
          onChunk: (content) => {
            streamingContentRef.current += content;
            setStreamingContent(streamingContentRef.current);
          },
          onMemoryContext: (data) => {
            // Create memory injection message when memories are retrieved
            if (data.facts.length > 0 || data.entities.length > 0 || data.relevant_turns?.length > 0) {
              const memoryMessage: MemoryInjectionMessage = {
                id: createMessageId(),
                type: 'memory_injection',
                timestamp: new Date().toISOString(),
                facts: data.facts,
                entities: data.entities,
                relevantTurns: data.relevant_turns || [],
                queryUsed: data.query,
              };
              appendMessage(memoryMessage);
            }
          },
          onToolCall: (data) => {
            // Flush current streaming content as an intermediate assistant message
            // This preserves the thinking/reasoning that led to the tool call
            const currentContent = streamingContentRef.current;
            if (currentContent.trim()) {
              const thinking = extractThinking(currentContent);
              const cleanContent = stripThinkingTags(currentContent, false);

              // Create intermediate message if there's thinking or content
              if (thinking || cleanContent) {
                const intermediateMsg: AssistantMessage = {
                  id: createMessageId(),
                  type: 'assistant',
                  timestamp: new Date().toISOString(),
                  content: cleanContent,
                  thinking: thinking || undefined,
                  agentName: agentName,
                };
                appendMessage(intermediateMsg);
              }

              // Clear streaming content for next round
              streamingContentRef.current = '';
              setStreamingContent('');
            }

            // Create tool call message when a tool is invoked
            const messageId = createMessageId();
            const toolCallMessage: ToolCallMessage = {
              id: messageId,
              type: 'tool_call',
              timestamp: new Date().toISOString(),
              toolName: data.tool,
              toolCallId: data.tool_call_id,
              arguments: data.arguments,
              status: 'running',
            };
            // Track message ID for in-place updates when result arrives
            toolCallMessageIds.current.set(data.tool_call_id, messageId);
            appendMessage(toolCallMessage);
          },
          onToolResult: (data) => {
            // Update existing tool call message with result (unified card)
            const messageId = toolCallMessageIds.current.get(data.tool_call_id);
            if (messageId) {
              updateMessage(messageId, {
                status: data.success ? 'completed' : 'failed',
                result: {
                  content: data.content,
                  success: data.success,
                  durationMs: data.duration_ms,
                },
              });
              toolCallMessageIds.current.delete(data.tool_call_id);
            }
          },
          onPlanStart: (data) => {
            const messageId = createMessageId();
            planMessageIdRef.current = messageId;
            // Initialize subtasks from the count with pending status
            const subtasks: PlanSubtask[] = [];
            planSubtasksRef.current = subtasks;

            const planMessage: PlanExecutionMessage = {
              id: messageId,
              type: 'plan_execution',
              timestamp: new Date().toISOString(),
              planId: data.plan_id,
              task: data.task,
              complexity: data.complexity,
              subtaskCount: data.subtask_count,
              status: 'running',
              subtasks,
            };
            appendMessage(planMessage);
          },
          onSubtaskStart: (data) => {
            if (!planMessageIdRef.current) return;
            // Add or update subtask in the tracked list
            const existing = planSubtasksRef.current.find(s => s.subtaskId === data.subtask_id);
            if (existing) {
              existing.status = 'running';
            } else {
              planSubtasksRef.current.push({
                subtaskId: data.subtask_id,
                description: data.description,
                subtaskType: data.type,
                status: 'running',
              });
            }
            updateMessage(planMessageIdRef.current, {
              subtasks: [...planSubtasksRef.current],
            });
          },
          onSubtaskComplete: (data) => {
            if (!planMessageIdRef.current) return;
            const subtask = planSubtasksRef.current.find(s => s.subtaskId === data.subtask_id);
            if (subtask) {
              subtask.status = 'completed';
              subtask.resultPreview = data.result_preview;
            }
            updateMessage(planMessageIdRef.current, {
              subtasks: [...planSubtasksRef.current],
              completedCount: planSubtasksRef.current.filter(s => s.status === 'completed').length,
            });
          },
          onSubtaskFailed: (data) => {
            if (!planMessageIdRef.current) return;
            const subtask = planSubtasksRef.current.find(s => s.subtaskId === data.subtask_id);
            if (subtask) {
              subtask.status = 'failed';
              subtask.error = data.error;
            }
            updateMessage(planMessageIdRef.current, {
              subtasks: [...planSubtasksRef.current],
            });
          },
          onPlanComplete: (data) => {
            if (!planMessageIdRef.current) return;
            updateMessage(planMessageIdRef.current, {
              status: data.completed_count === data.subtask_count ? 'completed' : 'failed',
              completedCount: data.completed_count,
              totalTimeMs: data.total_time_ms,
            });
            planMessageIdRef.current = null;
            planSubtasksRef.current = [];
          },
          onDone: (data) => {
            const finalContent = streamingContentRef.current;
            streamingContentRef.current = '';
            setStreamingContent('');
            setIsTyping(false);
            setStreaming(false);

            const cleanContent = stripThinkingTags(finalContent);
            if (cleanContent) {
              const assistantMessage: AssistantMessage = {
                id: createMessageId(),
                type: 'assistant',
                content: cleanContent,
                timestamp: new Date().toISOString(),
                thinking: data.thinking,
                latencyMs: data.total_time_ms,
                agentName: data.agent_name,
                tokensInput: data.tokens_input ?? undefined,
                tokensOutput: data.tokens_output ?? undefined,
              };
              appendMessage(assistantMessage);
            }

            if (data.session_id) {
              setSessionId(data.session_id);
            }

            // Update context usage info
            if (data.context_window && data.context_used) {
              setContextInfo({
                window: data.context_window,
                used: data.context_used,
              });
            }
          },
          onError: (error) => {
            console.error('Stream error:', error);
            streamingContentRef.current = '';
            setStreamingContent('');
            setIsTyping(false);
            setStreaming(false);

            const errorMessage: AssistantMessage = {
              id: createMessageId(),
              type: 'assistant',
              content: `Sorry, I encountered an error: ${error}`,
              timestamp: new Date().toISOString(),
            };
            appendMessage(errorMessage);
          },
        }
      );
    } else {
      // Non-streaming mode
      try {
        const response: ChatResponse = await api.chat({
          message: messageText,
          session_id: activeTab.sessionId || undefined,
          use_memory: useMemory,
        });

        if (response.session_id) {
          setSessionId(response.session_id);
        }

        const assistantMessage: AssistantMessage = {
          id: createMessageId(),
          type: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
          thinking: response.thinking,
          tokensUsed: response.tokens_used,
        };
        appendMessage(assistantMessage);
      } catch (error) {
        const errorMessage: AssistantMessage = {
          id: createMessageId(),
          type: 'assistant',
          content: 'Sorry, I encountered an error. Please check if the server is running.',
          timestamp: new Date().toISOString(),
        };
        appendMessage(errorMessage);
      } finally {
        setIsTyping(false);
        setStreaming(false);
      }
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
          <MessageBubble key={message.id} message={message} agentName={agentName} avatarId={activeProfile?.avatar} />
        ))}

        {/* Streaming message or typing indicator */}
        {isTyping && (() => {
          const AvatarIcon = getAvatarIcon(activeProfile?.avatar);
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
            <Sparkles size={12} />
            <span>{activeProfile?.name || 'Select Agent'}</span>
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
                streamAbortRef.current?.abort();
                streamingContentRef.current = '';
                setStreamingContent('');
                setIsTyping(false);
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


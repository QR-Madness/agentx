/**
 * ChatPanel — Core chat UI for rendering conversations
 * Consumes active tab from ConversationContext and handles messaging
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send,
  Bot,
  User,
  Loader2,
  Zap,
  Mic,
  Settings2,
  Radio,
  Database,
} from 'lucide-react';
import { api, type ChatResponse } from '../../lib/api';
import { MessageContent, ThinkingBubble, MessageActions } from '../chat';
import { useConversation } from '../../contexts/ConversationContext';
import {
  type ConversationMessage,
  type UserMessage,
  type AssistantMessage,
  createMessageId,
  isUserMessage,
  isAssistantMessage,
  isToolCallMessage,
  isToolResultMessage,
  isMemoryInjectionMessage,
  isSystemMessage,
  isErrorMessage,
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

export function ChatPanel() {
  const {
    activeTab,
    appendMessage,
    setStreaming,
    setSessionId,
  } = useConversation();

  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [useStreamingMode, setUseStreamingMode] = useState(true);
  const [useMemory, setUseMemory] = useState(true);
  const [streamingContent, setStreamingContent] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<{ abort: () => void } | null>(null);
  const streamingContentRef = useRef('');

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [activeTab?.messages, streamingContent, scrollToBottom]);

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
              };
              appendMessage(assistantMessage);
            }

            if (data.session_id) {
              setSessionId(data.session_id);
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
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
          <Bot size={20} />
          <span className="chat-panel-title">{activeTab.title}</span>
          {activeTab.sessionId && (
            <span className="chat-panel-session">
              Session: {activeTab.sessionId.slice(0, 8)}...
            </span>
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
          <button
            className="icon-button"
            title="Settings"
          >
            <Settings2 size={16} />
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
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Streaming message or typing indicator */}
        {isTyping && (
          <div className="message-bubble assistant">
            <div className="message-avatar">
              <Bot size={16} />
            </div>
            <div className="message-body">
              {streamingContent ? (
                <div className="streaming-message">
                  {(() => {
                    const thinkMatch = streamingContent.match(
                      /<think(?:ing)?>([\s\S]*?)(?:<\/think(?:ing)?>|$)/i
                    );
                    return thinkMatch ? (
                      <ThinkingBubble thinking={thinkMatch[1]} isStreaming />
                    ) : null;
                  })()}
                  <MessageContent content={stripThinkingTags(streamingContent, true)} />
                  <span className="streaming-cursor">▊</span>
                </div>
              ) : (
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-panel-input">
        <div className="input-container">
          <button className="voice-button" disabled title="Voice input coming soon">
            <Mic size={18} />
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Shift+Enter for new line)"
            rows={1}
            disabled={isTyping}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
          >
            {isTyping ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          </button>
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

function MessageBubble({ message }: { message: ConversationMessage }) {
  // Determine message content based on type
  const getContent = (): string => {
    if (isUserMessage(message)) return message.content;
    if (isAssistantMessage(message)) return message.content;
    if (isToolResultMessage(message)) return message.content;
    if (isSystemMessage(message)) return message.content;
    if (isErrorMessage(message)) return message.content;
    if (isToolCallMessage(message)) return `Tool call: ${message.toolName}`;
    if (isMemoryInjectionMessage(message)) return `Memory: ${message.facts.length} facts`;
    return '';
  };

  const content = getContent();
  const isUser = isUserMessage(message);
  const isAssistant = isAssistantMessage(message);

  return (
    <div className={`message-bubble ${message.type}`}>
      <div className="message-avatar">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-body">
        {/* Thinking bubble for assistant */}
        {isAssistant && message.thinking && (
          <ThinkingBubble thinking={message.thinking} />
        )}

        {/* Content */}
        {isAssistant ? (
          <MessageContent content={content} />
        ) : (
          <div className="message-text">{content}</div>
        )}

        {/* Meta info */}
        <div className="message-meta">
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {isAssistant && message.model && (
            <span className="message-model">{message.model.split('/').pop()}</span>
          )}
          {isAssistant && message.tokensUsed && (
            <span className="message-tokens">
              <Zap size={10} />
              {message.tokensUsed}
            </span>
          )}
        </div>

        {/* Actions */}
        {isAssistant && (
          <MessageActions
            content={content}
            isAssistant={true}
            timestamp={new Date(message.timestamp)}
          />
        )}
      </div>
    </div>
  );
}

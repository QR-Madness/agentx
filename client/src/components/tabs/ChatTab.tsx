import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Plus,
  Bot,
  User,
  Loader2,
  ChevronDown,
  ChevronRight,
  Zap,
  Brain,
  Settings2,
  RefreshCw,
  FileText,
  Radio,
  Database,
  Thermometer,
  History,
  Trash2,
  MessageSquare,
  Mic,
} from 'lucide-react';
import { api, ChatResponse, ReasoningStep, PromptProfile } from '../../lib/api';
import { MessageContent, ThinkingBubble, MessageActions } from '../chat';
import {
  getRecentChats,
  addRecentChat,
  deleteRecentChat,
  RecentChat,
  ChatMessage as StoredChatMessage,
} from '../../lib/storage';
import '../../styles/ChatTab.css';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  thinking?: string;  // Extracted thinking content
  reasoning?: ReasoningStep[];
  tokensUsed?: number;
  model?: string;
}

// Strip thinking tags from content (server also does this, but needed for streamed content)
function stripThinkingTags(content: string): string {
  return content
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/\[thinking\][\s\S]*?\[\/thinking\]/gi, '')
    .replace(/\[think\][\s\S]*?\[\/think\]/gi, '')
    .replace(/<internal_monologue>[\s\S]*?<\/internal_monologue>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

interface ProviderModels {
  provider: string;
  status: string;
  models: Array<{ id: string; name?: string; size?: number; parameter_size?: string }>;
}

export const ChatTab: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI assistant powered by AgentX. How can I help you today?',
      sender: 'assistant',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Model selection state
  const [availableModels, setAvailableModels] = useState<ProviderModels[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState(false);
  const [showModelSelector, setShowModelSelector] = useState(false);
  
  // Prompt profile state
  const [availableProfiles, setAvailableProfiles] = useState<PromptProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>('');
  const [showProfileSelector, setShowProfileSelector] = useState(false);
  
  // Streaming state
  const [useStreaming, setUseStreaming] = useState(true);
  const [streamingContent, setStreamingContent] = useState('');
  const streamAbortRef = useRef<{ abort: () => void } | null>(null);
  const streamingContentRef = useRef(''); // Track content synchronously

  // Advanced settings state
  const [temperature, setTemperature] = useState(0.7);
  const [useMemory, setUseMemory] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Recent chats state
  const [recentChats, setRecentChats] = useState<RecentChat[]>([]);
  const [showRecentChats, setShowRecentChats] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Fetch available models and profiles on mount
  useEffect(() => {
    fetchModels();
    fetchProfiles();
    loadRecentChats();
  }, []);

  // Save chat to recent chats when messages change (with session)
  useEffect(() => {
    if (sessionId && messages.length > 1) {
      const firstUserMessage = messages.find(m => m.sender === 'user');
      const title = firstUserMessage
        ? firstUserMessage.content.slice(0, 50) + (firstUserMessage.content.length > 50 ? '...' : '')
        : 'New Chat';
      const preview = firstUserMessage?.content.slice(0, 100) || '';

      const storedMessages: StoredChatMessage[] = messages.map(m => ({
        id: m.id,
        content: m.content,
        sender: m.sender,
        timestamp: m.timestamp.toISOString(),
        thinking: m.thinking,
        tokensUsed: m.tokensUsed,
        model: m.model,
      }));

      addRecentChat({
        sessionId,
        title,
        preview,
        messages: storedMessages,
        model: selectedModel,
        createdAt: messages[0]?.timestamp.toISOString() || new Date().toISOString(),
        lastMessageAt: new Date().toISOString(),
      });

      loadRecentChats();
    }
  }, [messages, sessionId]);

  const loadRecentChats = () => {
    setRecentChats(getRecentChats());
  };

  const loadChat = (chat: RecentChat) => {
    const loadedMessages: Message[] = chat.messages.map(m => ({
      ...m,
      timestamp: new Date(m.timestamp),
    }));
    setMessages(loadedMessages);
    setSessionId(chat.sessionId);
    if (chat.model) {
      setSelectedModel(chat.model);
    }
    setShowRecentChats(false);
  };

  const handleDeleteChat = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteRecentChat(chatId);
    loadRecentChats();
  };

  const fetchProfiles = async () => {
    try {
      const response = await api.listPromptProfiles();
      setAvailableProfiles(response.profiles);
      // Set default profile
      const defaultProfile = response.profiles.find(p => p.is_default);
      if (defaultProfile && !selectedProfile) {
        setSelectedProfile(defaultProfile.id);
      }
    } catch (error) {
      console.error('Failed to fetch profiles:', error);
    }
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await api.checkProvidersHealth();
      const providers: ProviderModels[] = [];
      
      // Response format: { status: string, providers: { [name]: { status, models, ... } } }
      const providersData = (response as Record<string, unknown>).providers || response;
      for (const [name, data] of Object.entries(providersData as Record<string, unknown>)) {
        if (typeof data !== 'object' || data === null) continue;
        const providerData = data as { status: string; models?: unknown[]; models_available?: number };
        if (providerData.models && Array.isArray(providerData.models)) {
          providers.push({
            provider: name,
            status: providerData.status,
            models: providerData.models.map((m: unknown) => {
              if (typeof m === 'string') {
                return { id: m };
              }
              return m as { id: string; name?: string; size?: number; parameter_size?: string };
            }),
          });
        }
      }
      
      setAvailableModels(providers);
      
      // Set default model if none selected
      if (!selectedModel && providers.length > 0) {
        const firstProvider = providers.find(p => p.status === 'healthy' && p.models.length > 0);
        if (firstProvider) {
          setSelectedModel(firstProvider.models[0].id || firstProvider.models[0].name || '');
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  const formatModelName = (model: { id: string; name?: string; parameter_size?: string }) => {
    const name = model.name || model.id;
    const size = model.parameter_size ? ` (${model.parameter_size})` : '';
    return `${name}${size}`;
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const messageText = input;
    setInput('');
    setIsTyping(true);

    if (useStreaming) {
      // Streaming mode
      setStreamingContent('');
      streamingContentRef.current = '';
      const messageId = (Date.now() + 1).toString();
      
      streamAbortRef.current = api.streamChat(
        {
          message: messageText,
          session_id: sessionId || undefined,
          model: selectedModel || undefined,
          profile_id: selectedProfile || undefined,
          temperature,
          use_memory: useMemory,
        },
        {
          onStart: (data) => {
            console.log('Stream started:', data);
          },
          onChunk: (content) => {
            streamingContentRef.current += content;
            setStreamingContent(streamingContentRef.current);
          },
          onDone: (data) => {
            // Capture content from ref (synchronous, no race condition)
            const finalContent = streamingContentRef.current;
            
            // Clear streaming state immediately
            streamingContentRef.current = '';
            setStreamingContent('');
            setIsTyping(false);
            
            // Add finalized message (only if we have content)
            // Strip thinking tags from content (they're extracted separately)
            const cleanContent = stripThinkingTags(finalContent);
            if (cleanContent) {
              setMessages(msgs => [...msgs, {
                id: messageId,
                content: cleanContent,
                sender: 'assistant' as const,
                timestamp: new Date(),
                thinking: data.thinking,
                model: selectedModel,
              }]);
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
            const errorMessage: Message = {
              id: messageId,
              content: `Sorry, I encountered an error: ${error}`,
              sender: 'assistant',
              timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
          },
        }
      );
    } else {
      // Non-streaming mode
      try {
        const response: ChatResponse = await api.chat({
          message: messageText,
          session_id: sessionId || undefined,
          model: selectedModel || undefined,
          show_reasoning: showReasoning,
          profile_id: selectedProfile || undefined,
          temperature,
          use_memory: useMemory,
        });

        if (response.session_id) {
          setSessionId(response.session_id);
        }

        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response.response,
          sender: 'assistant',
          timestamp: new Date(),
          thinking: response.thinking,
          reasoning: response.reasoning_trace,
          tokensUsed: response.tokens_used,
          model: selectedModel,
        };
        setMessages(prev => [...prev, aiMessage]);
      } catch (error) {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: 'Sorry, I encountered an error. Please check if the server is running and try again.',
          sender: 'assistant',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsTyping(false);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    // Abort any ongoing stream
    streamAbortRef.current?.abort();
    setStreamingContent('');
    setMessages([{
      id: Date.now().toString(),
      content: 'Hello! I\'m your AI assistant powered by AgentX. How can I help you today?',
      sender: 'assistant',
      timestamp: new Date(),
    }]);
    setSessionId(null);
  };

  const toggleReasoning = (messageId: string) => {
    setExpandedReasoning(prev => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleRegenerate = (messageIndex: number) => {
    // Find the last user message before this assistant message
    const userMessageIndex = messages.slice(0, messageIndex).reverse().findIndex(m => m.sender === 'user');
    if (userMessageIndex === -1) return;

    const actualUserIndex = messageIndex - 1 - userMessageIndex;
    const userMessage = messages[actualUserIndex];
    if (!userMessage) return;

    // Remove messages from this assistant message onwards
    setMessages(prev => prev.slice(0, messageIndex));

    // Set input to the user message and trigger send
    setInput(userMessage.content);
    // Use setTimeout to allow state to update before sending
    setTimeout(() => {
      const sendEvent = new CustomEvent('chat-regenerate');
      window.dispatchEvent(sendEvent);
    }, 50);
  };

  // Listen for regenerate events
  useEffect(() => {
    const handleRegenerateEvent = () => {
      if (input.trim()) {
        handleSend();
      }
    };
    window.addEventListener('chat-regenerate', handleRegenerateEvent);
    return () => window.removeEventListener('chat-regenerate', handleRegenerateEvent);
  }, [input]);

  return (
    <div className="chat-tab">
      {/* Header */}
      <div className="chat-header card">
        <div className="chat-title">
          <div className="chat-avatar">
            <Bot size={24} />
          </div>
          <div>
            <h2>AI Chat</h2>
            <p className="chat-status">
              <span className="status-dot online"></span>
              {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'Ready'}
            </p>
          </div>
        </div>
        <div className="chat-actions">
          <button
            className={`button-ghost ${useMemory ? 'active' : ''}`}
            onClick={() => setUseMemory(!useMemory)}
            title={useMemory ? 'Memory enabled' : 'Memory disabled'}
          >
            <Database size={18} />
          </button>
          <button
            className={`button-ghost ${useStreaming ? 'active' : ''}`}
            onClick={() => setUseStreaming(!useStreaming)}
            title={useStreaming ? 'Streaming enabled' : 'Streaming disabled'}
          >
            <Radio size={18} />
          </button>
          <button
            className={`button-ghost ${showProfileSelector ? 'active' : ''}`}
            onClick={() => { setShowProfileSelector(!showProfileSelector); setShowModelSelector(false); }}
            title="Select prompt profile"
          >
            <FileText size={18} />
          </button>
          <button 
            className={`button-ghost ${showReasoning ? 'active' : ''}`}
            onClick={() => setShowReasoning(!showReasoning)}
            title="Show reasoning traces"
          >
            <Brain size={18} />
          </button>
          <button
            className={`button-ghost ${showModelSelector ? 'active' : ''}`}
            onClick={() => { setShowModelSelector(!showModelSelector); setShowProfileSelector(false); }}
            title="Select model"
          >
            <Settings2 size={18} />
          </button>
          <button
            className={`button-ghost ${showRecentChats ? 'active' : ''}`}
            onClick={() => { setShowRecentChats(!showRecentChats); setShowModelSelector(false); setShowProfileSelector(false); }}
            title="Recent chats"
          >
            <History size={18} />
          </button>
          <button className="button-secondary" onClick={handleNewChat}>
            <Plus size={16} />
            New Chat
          </button>
        </div>
      </div>

      {/* Recent Chats Dropdown */}
      {showRecentChats && (
        <div className="recent-chats-panel card">
          <div className="recent-chats-header">
            <h3>Recent Chats</h3>
            <span className="chat-count">{recentChats.length} chats</span>
          </div>
          {recentChats.length === 0 ? (
            <div className="recent-chats-empty">
              <MessageSquare size={24} />
              <p>No recent chats yet</p>
            </div>
          ) : (
            <div className="recent-chats-list">
              {recentChats.map(chat => (
                <div
                  key={chat.id}
                  className={`recent-chat-item ${chat.sessionId === sessionId ? 'active' : ''}`}
                  onClick={() => loadChat(chat)}
                >
                  <div className="recent-chat-content">
                    <span className="recent-chat-title">{chat.title}</span>
                    <span className="recent-chat-preview">{chat.preview}</span>
                    <span className="recent-chat-time">
                      {new Date(chat.lastMessageAt).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    className="recent-chat-delete"
                    onClick={(e) => handleDeleteChat(chat.id, e)}
                    title="Delete chat"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Profile Selector Panel */}
      {showProfileSelector && (
        <div className="model-selector card">
          <div className="model-selector-header">
            <h3>Select Prompt Profile</h3>
            <button 
              className="button-ghost" 
              onClick={fetchProfiles}
              title="Refresh profiles"
            >
              <RefreshCw size={16} />
            </button>
          </div>
          
          {availableProfiles.length === 0 ? (
            <div className="model-empty">
              <p>No profiles available.</p>
            </div>
          ) : (
            <div className="model-list">
              <div className="provider-group">
                <div className="provider-header">
                  <span className="provider-name">Prompt Profiles</span>
                  <span className="model-count">{availableProfiles.length} profiles</span>
                </div>
                <div className="provider-models">
                  {availableProfiles.map(profile => (
                    <button
                      key={profile.id}
                      className={`model-option ${selectedProfile === profile.id ? 'selected' : ''}`}
                      onClick={() => {
                        setSelectedProfile(profile.id);
                        setShowProfileSelector(false);
                      }}
                    >
                      <div className="profile-option-content">
                        <span className="model-name">{profile.name}</span>
                        {profile.description && (
                          <span className="profile-description">{profile.description}</span>
                        )}
                      </div>
                      {profile.is_default && (
                        <span className="model-selected-badge">Default</span>
                      )}
                      {selectedProfile === profile.id && (
                        <span className="model-selected-badge">Active</span>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Selector Panel */}
      {showModelSelector && (
        <div className="model-selector card">
          <div className="model-selector-header">
            <h3>Select Model</h3>
            <button 
              className="button-ghost" 
              onClick={fetchModels}
              disabled={loadingModels}
              title="Refresh models"
            >
              <RefreshCw size={16} className={loadingModels ? 'spin' : ''} />
            </button>
          </div>
          
          {loadingModels ? (
            <div className="model-loading">
              <Loader2 size={20} className="spin" />
              <span>Loading models...</span>
            </div>
          ) : availableModels.length === 0 ? (
            <div className="model-empty">
              <p>No models available. Check your provider configuration.</p>
            </div>
          ) : (
            <div className="model-list">
              {availableModels.map(provider => (
                <div key={provider.provider} className="provider-group">
                  <div className="provider-header">
                    <span className={`status-dot ${provider.status === 'healthy' ? 'online' : 'offline'}`}></span>
                    <span className="provider-name">{provider.provider}</span>
                    <span className="model-count">{provider.models.length} models</span>
                  </div>
                  <div className="provider-models">
                    {provider.models.map(model => (
                      <button
                        key={model.id || model.name}
                        className={`model-option ${selectedModel === (model.id || model.name) ? 'selected' : ''}`}
                        onClick={() => {
                          setSelectedModel(model.id || model.name || '');
                          setShowModelSelector(false);
                        }}
                      >
                        <span className="model-name">{formatModelName(model)}</span>
                        {selectedModel === (model.id || model.name) && (
                          <span className="model-selected-badge">Active</span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Current Model Badge */}
      {(selectedModel || selectedProfile) && (
        <div className="current-model-badges">
          {selectedModel && (
            <div className="current-model-badge">
              <Bot size={14} />
              <span>{selectedModel}</span>
            </div>
          )}
          {selectedProfile && (
            <div className="current-model-badge profile">
              <FileText size={14} />
              <span>{availableProfiles.find(p => p.id === selectedProfile)?.name || selectedProfile}</span>
            </div>
          )}
        </div>
      )}

      {/* Advanced Settings Panel */}
      <div className="advanced-settings">
        <button
          className="advanced-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Thermometer size={14} />
          <span>Advanced Settings</span>
          <span className="advanced-value">Temp: {temperature.toFixed(1)}</span>
        </button>
        {showAdvanced && (
          <div className="advanced-panel card">
            <div className="setting-row">
              <label className="setting-label">
                <Thermometer size={14} />
                Temperature
                <span className="setting-value">{temperature.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="temperature-slider"
              />
              <div className="temperature-hints">
                <span>Precise</span>
                <span>Creative</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-avatar">
              {message.sender === 'user' ? <User size={18} /> : <Bot size={18} />}
            </div>
            <div className="message-content">
              {/* Thinking bubble for assistant messages */}
              {message.sender === 'assistant' && message.thinking && (
                <ThinkingBubble thinking={message.thinking} />
              )}
              
              {/* Message content - markdown for assistant, plain for user */}
              {message.sender === 'assistant' ? (
                <MessageContent content={message.content} />
              ) : (
                <div className="message-text">{message.content}</div>
              )}
              
              {/* Reasoning trace */}
              {message.reasoning && message.reasoning.length > 0 && (
                <div className="reasoning-section">
                  <button 
                    className="reasoning-toggle"
                    onClick={() => toggleReasoning(message.id)}
                  >
                    {expandedReasoning[message.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    <Brain size={14} />
                    <span>Reasoning ({message.reasoning.length} steps)</span>
                  </button>
                  {expandedReasoning[message.id] && (
                    <div className="reasoning-trace">
                      {message.reasoning.map((step, idx) => (
                        <div key={idx} className={`reasoning-step ${step.type}`}>
                          <span className="step-type">{step.type}</span>
                          <span className="step-content">{step.content}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              
              <div className="message-meta">
                <span className="message-time">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                {message.model && (
                  <span className="message-model" title={message.model}>
                    {message.model.split('/').pop()}
                  </span>
                )}
                {message.tokensUsed && (
                  <span className="message-tokens">
                    <Zap size={12} />
                    {message.tokensUsed}
                  </span>
                )}
              </div>
              <MessageActions
                content={message.content}
                isAssistant={message.sender === 'assistant'}
                timestamp={message.timestamp}
                onRegenerate={
                  message.sender === 'assistant'
                    ? () => handleRegenerate(messages.findIndex(m => m.id === message.id))
                    : undefined
                }
              />
            </div>
          </div>
        ))}

        {/* Streaming message or typing indicator */}
        {isTyping && (
          <div className="message assistant">
            <div className="message-avatar">
              <Bot size={18} />
            </div>
            <div className="message-content">
              {streamingContent ? (
                <div className="streaming-message">
                  {/* Show thinking bubble if we have thinking content during streaming */}
                  {(() => {
                    const thinkMatch = streamingContent.match(/<think(?:ing)?>([\s\S]*?)(?:<\/think(?:ing)?>|$)/i);
                    return thinkMatch ? <ThinkingBubble thinking={thinkMatch[1]} isStreaming /> : null;
                  })()}
                  <MessageContent content={stripThinkingTags(streamingContent)} />
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
      <div className="chat-input-wrapper">
        <div className="chat-input-container card">
          <button
            className="voice-button button-ghost"
            disabled
            title="Voice input coming soon"
          >
            <Mic size={18} />
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            onPaste={(e) => {
              // Check for image paste
              const items = e.clipboardData?.items;
              if (items) {
                for (let i = 0; i < items.length; i++) {
                  if (items[i].type.indexOf('image') !== -1) {
                    e.preventDefault();
                    // TODO: Show toast notification
                    console.log('Image paste detected - feature coming soon');
                    return;
                  }
                }
              }
            }}
            placeholder="Type your message... (Shift+Enter for new line)"
            className="chat-input"
            rows={1}
            disabled={isTyping}
          />
          <button
            className="send-button button-primary"
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
          >
            {isTyping ? <Loader2 size={20} className="spin" /> : <Send size={20} />}
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
};

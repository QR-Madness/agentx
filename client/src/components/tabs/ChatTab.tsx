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
  RefreshCw
} from 'lucide-react';
import { api, ChatResponse, ReasoningStep } from '../../lib/api';
import '../../styles/ChatTab.css';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  reasoning?: ReasoningStep[];
  tokensUsed?: number;
  model?: string;
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch available models on mount
  useEffect(() => {
    fetchModels();
  }, []);

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
    setInput('');
    setIsTyping(true);

    try {
      const response: ChatResponse = await api.chat({
        message: input,
        session_id: sessionId || undefined,
        model: selectedModel || undefined,
        show_reasoning: showReasoning,
      });

      if (response.session_id) {
        setSessionId(response.session_id);
      }

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        sender: 'assistant',
        timestamp: new Date(),
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
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
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
            className={`button-ghost ${showReasoning ? 'active' : ''}`}
            onClick={() => setShowReasoning(!showReasoning)}
            title="Show reasoning traces"
          >
            <Brain size={18} />
          </button>
          <button 
            className={`button-ghost ${showModelSelector ? 'active' : ''}`}
            onClick={() => setShowModelSelector(!showModelSelector)}
            title="Select model"
          >
            <Settings2 size={18} />
          </button>
          <button className="button-secondary" onClick={handleNewChat}>
            <Plus size={16} />
            New Chat
          </button>
        </div>
      </div>

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
      {selectedModel && (
        <div className="current-model-badge">
          <Bot size={14} />
          <span>{selectedModel}</span>
        </div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-avatar">
              {message.sender === 'user' ? <User size={18} /> : <Bot size={18} />}
            </div>
            <div className="message-content">
              <div className="message-text">{message.content}</div>
              
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
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="message assistant">
            <div className="message-avatar">
              <Bot size={18} />
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-container card">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
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
    </div>
  );
};

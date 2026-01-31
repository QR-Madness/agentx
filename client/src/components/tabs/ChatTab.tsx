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
  Brain
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
          <button className="button-secondary" onClick={handleNewChat}>
            <Plus size={16} />
            New Chat
          </button>
        </div>
      </div>

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

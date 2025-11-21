import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const ViewContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 24px;
`;

const ViewHeader = styled.div`
  margin-bottom: 24px;

  h2 {
    font-size: 24px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${({ theme }) => theme.colors.bgSecondary};
  border-radius: 12px;
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const Message = styled.div<{ $type: 'user' | 'assistant' | 'system' }>`
  padding: 12px 16px;
  border-radius: 8px;
  max-width: ${({ $type }) => ($type === 'system' ? '100%' : '80%')};
  background: ${({ $type, theme }) => {
    if ($type === 'user') return theme.colors.accentPrimary;
    if ($type === 'system') return theme.colors.bgTertiary;
    return theme.colors.bgTertiary;
  }};
  color: ${({ $type, theme }) =>
    $type === 'user' ? 'white' : theme.colors.textPrimary};
  align-self: ${({ $type }) => {
    if ($type === 'user') return 'flex-end';
    if ($type === 'system') return 'center';
    return 'flex-start';
  }};
  text-align: ${({ $type }) => ($type === 'system' ? 'center' : 'left')};
`;

const ChatInputArea = styled.div`
  display: flex;
  gap: 12px;
  padding: 16px;
  border-top: 1px solid ${({ theme }) => theme.colors.borderColor};
`;

const ChatInput = styled.input`
  flex: 1;
  padding: 12px 16px;
  background: ${({ theme }) => theme.colors.bgTertiary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.textPrimary};
  font-size: 15px;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.accentPrimary};
  }
`;

const SendButton = styled.button`
  padding: 12px 24px;
  background: ${({ theme }) => theme.colors.accentPrimary};
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.accentHover};
  }
`;

interface MessageType {
  text: string;
  type: 'user' | 'assistant' | 'system';
}

export const ChatView: React.FC = () => {
  const [messages, setMessages] = useState<MessageType[]>([
    { text: 'Welcome to AgentX! Start a conversation...', type: 'system' },
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    const message = input.trim();
    if (!message) return;

    setMessages((prev) => [...prev, { text: message, type: 'user' }]);
    setInput('');


    // Stub: Simulate AI response
    setTimeout(async () => {
      setMessages((prev) => [
        ...prev,
        {
          text: 'This is a placeholder response. AI integration coming soon! Message sentiment: ' + sentimentClassification.label + '.',
          type: 'assistant',
        },
      ]);
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <ViewContainer>
      <ViewHeader>
        <h2>Conversation</h2>
      </ViewHeader>
      <ChatContainer>
        <ChatMessages>
          {messages.map((msg, idx) => (
            <Message key={idx} $type={msg.type}>
              <p>{msg.text}</p>
            </Message>
          ))}
          <div ref={messagesEndRef} />
        </ChatMessages>
        <ChatInputArea>
          <ChatInput
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <SendButton onClick={handleSend}>Send</SendButton>
        </ChatInputArea>
      </ChatContainer>
    </ViewContainer>
  );
};

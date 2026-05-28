import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { UserBubble } from './UserBubble';
import type { ConversationMessage } from '../../../lib/messages';

const userMessage = {
  id: 'm1',
  type: 'user',
  content: 'How do I wire up the recall tool?',
  timestamp: new Date().toISOString(),
} as Extract<ConversationMessage, { type: 'user' }>;

describe('UserBubble (flat row)', () => {
  it('renders the "You" label and message content', () => {
    render(<UserBubble message={userMessage} />);
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('How do I wire up the recall tool?')).toBeInTheDocument();
  });

  it('uses the flat user-row structure (no right-aligned bubble class)', () => {
    const { container } = render(<UserBubble message={userMessage} />);
    const row = container.querySelector('.message-bubble.user');
    expect(row).not.toBeNull();
    // The flat-row redesign drops the legacy bright-blue bubble in favour of a
    // subtle tinted text block with a name header.
    expect(container.querySelector('.user-header .user-name')).not.toBeNull();
    expect(container.querySelector('.message-text')).not.toBeNull();
  });
});

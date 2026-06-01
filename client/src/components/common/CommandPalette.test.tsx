import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CommandPalette } from './CommandPalette';

vi.mock('../../contexts/ModalContext', () => ({
  useModal: () => ({ openModal: vi.fn() }),
}));
vi.mock('../../contexts/ConversationContext', () => ({
  useConversation: () => ({
    addTab: vi.fn(),
    closeTab: vi.fn(),
    activeTabId: 't1',
    activeTab: { sessionId: null, messages: [], noMemorization: false },
    updateTab: vi.fn(),
  }),
}));
vi.mock('../../contexts/UIChromeContext', () => ({
  useUIChrome: () => ({ focusMode: false, toggleFocusMode: vi.fn() }),
}));
vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ authRequired: false, isAuthenticated: false, logout: vi.fn() }),
}));

describe('CommandPalette', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <CommandPalette isOpen={false} onClose={vi.fn()} onNavigate={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('lists navigation + tool actions when open', () => {
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    expect(screen.getByText('Go to Chat')).toBeInTheDocument();
    expect(screen.getByText('Open Settings')).toBeInTheDocument();
    expect(screen.getByText('Open Translation')).toBeInTheDocument();
  });

  it('filters the list by query', () => {
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    fireEvent.change(screen.getByPlaceholderText(/type a command/i), {
      target: { value: 'translation' },
    });
    expect(screen.getByText('Open Translation')).toBeInTheDocument();
    expect(screen.queryByText('Go to Chat')).not.toBeInTheDocument();
  });

  it('shows an empty state for no matches', () => {
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    fireEvent.change(screen.getByPlaceholderText(/type a command/i), {
      target: { value: 'zzzznomatch' },
    });
    expect(screen.getByText('No matching commands')).toBeInTheDocument();
  });

  it('runs an action and closes on click', () => {
    const onNavigate = vi.fn();
    const onClose = vi.fn();
    render(<CommandPalette isOpen onClose={onClose} onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText('Go to Dashboard'));
    expect(onNavigate).toHaveBeenCalledWith('dashboard');
    expect(onClose).toHaveBeenCalled();
  });
});

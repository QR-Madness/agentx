import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CommandPalette } from './CommandPalette';

const setTheme = vi.fn();

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
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ preference: 'cosmic', setTheme }),
}));
vi.mock('../../contexts/AmbassadorDockContext', () => ({
  useAmbassadorDock: () => ({ dockCapable: false, open: false, setOpen: vi.fn() }),
}));

describe('CommandPalette', () => {
  beforeEach(() => {
    setTheme.mockClear();
    localStorage.clear();
  });

  it('renders nothing when closed', () => {
    const { container } = render(
      <CommandPalette isOpen={false} onClose={vi.fn()} onNavigate={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('lists navigation, workspace, and theme actions when open', () => {
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    expect(screen.getByText('Go to Chat')).toBeInTheDocument();
    expect(screen.getByText('Open Settings')).toBeInTheDocument();
    expect(screen.getByText('Open Logs')).toBeInTheDocument();
    expect(screen.getByText('Theme: Light')).toBeInTheDocument();
  });

  it('runs a navigation action and closes on click', () => {
    const onNavigate = vi.fn();
    const onClose = vi.fn();
    render(<CommandPalette isOpen onClose={onClose} onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText('Go to Dashboard'));
    expect(onNavigate).toHaveBeenCalledWith('dashboard');
    expect(onClose).toHaveBeenCalled();
  });

  it('switches theme from the Theme group', () => {
    const onClose = vi.fn();
    render(<CommandPalette isOpen onClose={onClose} onNavigate={vi.fn()} />);
    fireEvent.click(screen.getByText('Theme: Light'));
    expect(setTheme).toHaveBeenCalledWith('light');
    expect(onClose).toHaveBeenCalled();
  });

  it('offers the Solo/Team delegation toggle for the active conversation', () => {
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    // noDelegation is falsy on the mocked tab → the command offers to disable.
    expect(screen.getByText('Disable delegation for this chat')).toBeInTheDocument();
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    render(<CommandPalette isOpen onClose={onClose} onNavigate={vi.fn()} />);
    fireEvent.keyDown(screen.getByPlaceholderText(/type a command/i), { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  describe('settings section commands', () => {
    // Registry-driven, one per section — so they must not bury the handful of
    // commands worth seeing when the palette opens.
    it('stays out of the resting list', () => {
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      expect(screen.queryByText('Settings: Recall')).not.toBeInTheDocument();
      expect(screen.queryByText('Settings: Web Search')).not.toBeInTheDocument();
      // The plain entry point is still there.
      expect(screen.getByText('Open Settings')).toBeInTheDocument();
    });

    it('appears once the user searches', () => {
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      fireEvent.change(screen.getByPlaceholderText(/type a command/i), {
        target: { value: 'recall' },
      });
      expect(screen.getByText('Settings: Recall')).toBeInTheDocument();
    });

    it('is reachable by a section keyword, not just its label', () => {
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      fireEvent.change(screen.getByPlaceholderText(/type a command/i), {
        target: { value: 'tavily' },
      });
      expect(screen.getByText('Settings: Web Search')).toBeInTheDocument();
    });
  });
});

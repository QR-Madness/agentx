import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CommandPalette } from './CommandPalette';

const setTheme = vi.fn();
const modal = vi.hoisted(() => ({
  openModal: vi.fn(),
  closeModal: vi.fn(),
  openIds: new Set<string>(),
}));

vi.mock('../../contexts/ModalContext', () => ({
  useModal: () => ({
    openModal: modal.openModal,
    closeModal: modal.closeModal,
    isOpen: (id: string) => modal.openIds.has(id),
  }),
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
    modal.openModal.mockClear();
    modal.closeModal.mockClear();
    modal.openIds.clear();
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

  it('puts Open Organization first for "team", even with a chat open', () => {
    // The chat's delegation toggle also has a "team" keyword, and its group comes first.
    render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
    fireEvent.change(screen.getByPlaceholderText(/type a command/i), { target: { value: 'team' } });
    expect(screen.getAllByRole('option')[0]).toHaveTextContent('Open Organization');
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    render(<CommandPalette isOpen onClose={onClose} onNavigate={vi.fn()} />);
    fireEvent.keyDown(screen.getByPlaceholderText(/type a command/i), { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  // The palette is the top layer (lib/paletteLayer.ts). The surfaces, drawers and
  // the model picker beneath it listen for keys on window and document.
  describe('as the top layer', () => {
    const onWindow = vi.fn();
    const onDocument = vi.fn();
    beforeEach(() => {
      onWindow.mockClear();
      onDocument.mockClear();
      window.addEventListener('keydown', onWindow);
      document.addEventListener('keydown', onDocument);
    });
    afterEach(() => {
      window.removeEventListener('keydown', onWindow);
      document.removeEventListener('keydown', onDocument);
    });

    it('closes on Escape without the layers beneath hearing it', () => {
      const onClose = vi.fn();
      render(<CommandPalette isOpen onClose={onClose} onNavigate={vi.fn()} />);
      fireEvent.keyDown(screen.getByPlaceholderText(/type a command/i), { key: 'Escape' });
      expect(onClose).toHaveBeenCalled();
      expect(onDocument).not.toHaveBeenCalled();
      expect(onWindow).not.toHaveBeenCalled();
    });

    it('keeps typing and list navigation to itself', () => {
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      const input = screen.getByPlaceholderText(/type a command/i);
      for (const key of ['a', ' ', 'ArrowDown', 'ArrowUp', 'Tab']) fireEvent.keyDown(input, { key });
      expect(onDocument).not.toHaveBeenCalled();
      expect(onWindow).not.toHaveBeenCalled();
    });

    it('lets ⌘/Ctrl chords through to the app shortcuts on document', () => {
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      const input = screen.getByPlaceholderText(/type a command/i);
      fireEvent.keyDown(input, { key: 'k', ctrlKey: true });
      fireEvent.keyDown(input, { key: 'k', metaKey: true });
      expect(onDocument).toHaveBeenCalledTimes(2);
    });
  });

  describe('navigation', () => {
    // Deck, Memory and Projects are tab-surfaces: the palette opens over them, and
    // the TopBar's page pills already dismiss them on navigation.
    it('dismisses the open tab-surfaces, as the TopBar does', () => {
      modal.openIds = new Set(['memory', 'ambassador-deck', 'workspaces-drawer']);
      const onNavigate = vi.fn();
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={onNavigate} />);
      fireEvent.click(screen.getByText('Go to Dashboard'));
      expect(onNavigate).toHaveBeenCalledWith('dashboard');
      expect(modal.closeModal.mock.calls.map(([id]) => id).sort()).toEqual(
        ['ambassador-deck', 'memory', 'workspaces-drawer'],
      );
    });

    it('dismisses them for "New conversation" too, which lands on the chat page', () => {
      modal.openIds = new Set(['memory']);
      const onNavigate = vi.fn();
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={onNavigate} />);
      fireEvent.click(screen.getByText('New conversation'));
      expect(onNavigate).toHaveBeenCalledWith('agentx');
      expect(modal.closeModal).toHaveBeenCalledWith('memory');
    });

    it('leaves the modal surfaces open: they may hold unsaved work', () => {
      modal.openIds = new Set(['unified-settings', 'org-manager']);
      render(<CommandPalette isOpen onClose={vi.fn()} onNavigate={vi.fn()} />);
      fireEvent.click(screen.getByText('Go to Dashboard'));
      expect(modal.closeModal).not.toHaveBeenCalled();
    });
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

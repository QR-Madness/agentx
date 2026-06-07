/**
 * CommandPalette — ⌘K / Ctrl+K launcher for navigation + every drawer/modal.
 *
 * The single, primary command surface (the TopBar's Workspace overflow was
 * removed in favor of this). A thin `cmdk` renderer over the `useCommands()`
 * registry: cmdk owns fuzzy filtering, ranking, ↑↓/Enter, scroll-into-view and
 * ARIA; we add grouping, a "Recent" section (empty-query only), a theme group,
 * and a footer hint bar.
 *
 * Opened/closed by RootLayout (which owns page-navigation state + the global key
 * listener); the strip's search pill dispatches `agentx:toggle-command-palette`.
 */

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { Command } from 'cmdk';
import { Search, Check } from 'lucide-react';
import {
  useCommands,
  GROUP_ORDER,
  type Command as Cmd,
  type CommandGroup,
} from '../../hooks/useCommands';
import { getRecentCommandIds, pushRecentCommand } from '../../lib/recentCommands';
import { isMac } from '../../lib/platform';
import type { PageId } from '../../layouts/TopBar';
import './CommandPalette.css';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (page: PageId) => void;
}

function CommandRow({ cmd }: { cmd: Cmd }) {
  return (
    <>
      <span className="cmdk-item-icon">{cmd.icon}</span>
      <span className="cmdk-item-label">{cmd.label}</span>
      {cmd.isActive && <Check size={15} className="cmdk-item-check" />}
      {cmd.hint && <kbd className="cmdk-item-hint">{cmd.hint}</kbd>}
    </>
  );
}

export function CommandPalette({ isOpen, onClose, onNavigate }: CommandPaletteProps) {
  const [search, setSearch] = useState('');
  const commands = useCommands({ onNavigate, onClose });

  useEffect(() => {
    if (isOpen) setSearch('');
  }, [isOpen]);

  const byId = useMemo(() => new Map(commands.map(c => [c.id, c])), [commands]);

  // Recent is shown only when the query is empty (avoids double-matching an
  // action against both its Recent and canonical entry while searching).
  const recent = useMemo(() => {
    if (!isOpen || search.trim()) return [];
    return getRecentCommandIds()
      .map(id => byId.get(id))
      .filter((c): c is Cmd => !!c);
  }, [isOpen, search, byId]);

  const groups = useMemo<[CommandGroup, Cmd[]][]>(
    () =>
      GROUP_ORDER.map(g => [g, commands.filter(c => c.group === g)] as [CommandGroup, Cmd[]])
        .filter(([, items]) => items.length > 0),
    [commands],
  );

  if (!isOpen) return null;

  const run = (cmd: Cmd) => {
    pushRecentCommand(cmd.id);
    cmd.run();
  };

  return createPortal(
    <div className="cmdk-overlay" onClick={onClose} role="presentation">
      <div
        className="cmdk-panel"
        onClick={e => e.stopPropagation()}
        onKeyDown={e => {
          if (e.key === 'Escape') {
            e.preventDefault();
            onClose();
          }
        }}
      >
        <Command label="Command palette">
          <div className="cmdk-search">
            <Search size={16} className="cmdk-search-icon" />
            <Command.Input
              autoFocus
              value={search}
              onValueChange={setSearch}
              placeholder="Type a command or search…"
              className="cmdk-input"
            />
          </div>
          <Command.List className="cmdk-list">
            <Command.Empty className="cmdk-empty">No matching commands</Command.Empty>

            {recent.length > 0 && (
              <Command.Group heading="Recent" className="cmdk-group">
                {recent.map(cmd => (
                  <Command.Item
                    key={`recent:${cmd.id}`}
                    value={`recent:${cmd.id}`}
                    onSelect={() => run(cmd)}
                    className="cmdk-item"
                  >
                    <CommandRow cmd={cmd} />
                  </Command.Item>
                ))}
              </Command.Group>
            )}

            {groups.map(([group, items]) => (
              <Command.Group key={group} heading={group} className="cmdk-group">
                {items.map(cmd => (
                  <Command.Item
                    key={cmd.id}
                    value={cmd.id}
                    keywords={[cmd.label, ...(cmd.keywords ?? [])]}
                    onSelect={() => run(cmd)}
                    className="cmdk-item"
                  >
                    <CommandRow cmd={cmd} />
                  </Command.Item>
                ))}
              </Command.Group>
            ))}
          </Command.List>

          <div className="cmdk-footer">
            <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
            <span><kbd>↵</kbd> select</span>
            <span><kbd>esc</kbd> close</span>
            <span className="cmdk-footer-spacer" />
            <span className="cmdk-footer-brand"><kbd>{isMac ? '⌘' : 'Ctrl'}</kbd><kbd>K</kbd></span>
          </div>
        </Command>
      </div>
    </div>,
    document.body,
  );
}

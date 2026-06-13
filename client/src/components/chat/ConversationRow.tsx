/**
 * ConversationRow — one conversation entry in the list (open tab or past), with
 * the consolidated `⋯` actions menu (rename / pin / archive / icon / color /
 * move-to-group / select / delete), a multi-select checkbox, and a quick close
 * for open tabs.
 *
 * Memoized presentation: per-row *state* arrives as primitive props computed by
 * the parent (isActive/editing/busy/checked/selectionMode/draftTitle) and the
 * *handlers* arrive via the stable `RowHandlers` bundle from `useConversationList`.
 * That keeps it `React.memo`-skippable so streaming the active conversation only
 * re-renders the one row whose props actually changed — never the whole list.
 */

import { memo } from 'react';
import {
  MessageSquare, Download, Radio, Loader2, X, MoreHorizontal,
  Pin, PinOff, Archive, ArchiveRestore, Pencil, Trash2, Palette, Image, FolderInput, CheckSquare, Check,
} from 'lucide-react';
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
  DropdownMenuSeparator, DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuSubContent,
} from '../ui/DropdownMenu';
import { Checkbox } from '../ui/Checkbox';
import { getAvatarIcon } from '../../lib/avatars';
import { CONVERSATION_COLORS, conversationColorValue } from '../../lib/conversationColors';
import type { ConversationItem, RowHandlers } from '../../hooks/useConversationList';

interface ConversationRowProps {
  item: ConversationItem;
  handlers: RowHandlers;
  isActive: boolean;
  editing: boolean;
  busy: boolean;
  checked: boolean;
  selectionMode: boolean;
  /** Only supplied (and only changing) for the row being renamed. */
  draftTitle?: string;
  /** Flag a tab-backed row as currently open where the section doesn't imply it. */
  showOpenBadge?: boolean;
  onOpenIconPicker: (key: string) => void;
  onNewGroup: (key: string) => void;
}

function ConversationRowImpl({
  item, handlers: h, isActive, editing, busy, checked, selectionMode,
  draftTitle = '', showOpenBadge = false, onOpenIconPicker, onNewGroup,
}: ConversationRowProps) {
  const { meta } = item;
  const color = conversationColorValue(meta.color);

  const Icon = meta.icon ? getAvatarIcon(meta.icon)
    : item.isStreaming ? Radio
    : item.kind === 'server' ? Download
    : MessageSquare;

  const onClick = () => {
    if (editing) return;
    if (selectionMode) { h.toggleSelect(item.key); return; }
    h.openItem(item);
  };

  return (
    <div
      className={`history-item ${isActive ? 'active' : ''} ${checked ? 'selected' : ''} ${item.kind === 'server' ? 'history-item-server' : ''}`}
      onClick={onClick}
    >
      {selectionMode && (
        <Checkbox
          checked={checked}
          onCheckedChange={() => h.toggleSelect(item.key)}
          onClick={(e) => e.stopPropagation()}
          aria-label={`Select ${item.title}`}
        />
      )}

      <div className="history-item-icon" style={color ? { color, borderColor: color } : undefined}>
        <Icon size={14} className={item.isStreaming ? 'history-running-dot' : undefined} />
      </div>

      <div className="history-item-info">
        {editing ? (
          <input
            className="history-item-rename"
            value={draftTitle}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => h.setDraftTitle(e.target.value)}
            onBlur={() => h.commitRename(item)}
            onKeyDown={(e) => h.renameKeyDown(e, item)}
            aria-label="Rename conversation"
          />
        ) : (
          <span className="history-item-title" onDoubleClick={(e) => { e.stopPropagation(); h.startRename(item); }}>
            {meta.pinned && <Pin size={11} className="history-item-pinflag" />}
            {showOpenBadge && <span className="history-item-openflag" title="Open in a tab" aria-label="Open in a tab" />}
            {item.title}
          </span>
        )}
        <span className="history-item-meta">
          {item.messageCount} messages · {h.formatDate(item.lastMessageAt)}
        </span>
        {item.preview && <span className="history-item-preview">{item.preview}</span>}
      </div>

      <div className="history-item-actions">
        {busy ? (
          <div className="history-item-loading"><Loader2 size={12} className="spin" /></div>
        ) : selectionMode ? null : (
          <>
            {item.kind === 'tab' && (
              <button className="history-item-action" onClick={(e) => { e.stopPropagation(); h.closeOpenTab(item); }} title="Close (keeps on server)">
                <X size={12} />
              </button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="history-item-action" onClick={(e) => e.stopPropagation()} title="More" aria-label="Conversation actions">
                  <MoreHorizontal size={14} />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                onClick={(e) => e.stopPropagation()}
                // Don't yank focus back to the trigger on close — it would blur
                // (and dismiss) the inline "new group" input that opens on select.
                onCloseAutoFocus={(e) => e.preventDefault()}
              >
                <DropdownMenuItem onSelect={() => h.startRename(item)}>
                  <Pencil size={14} /> Rename
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => h.togglePin(item)}>
                  {meta.pinned ? <><PinOff size={14} /> Unpin</> : <><Pin size={14} /> Pin</>}
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => h.toggleArchive(item)}>
                  {meta.archived ? <><ArchiveRestore size={14} /> Unarchive</> : <><Archive size={14} /> Archive</>}
                </DropdownMenuItem>

                <DropdownMenuSeparator />

                <DropdownMenuItem onSelect={() => onOpenIconPicker(item.key)}>
                  <Image size={14} /> Set icon…
                </DropdownMenuItem>
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger><Palette size={14} /> Color</DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    <div className="conv-color-grid">
                      {CONVERSATION_COLORS.map(col => (
                        <button
                          key={col.key}
                          className={`conv-color-swatch ${meta.color === col.key ? 'selected' : ''}`}
                          style={{ background: col.value }}
                          title={col.label}
                          onClick={() => h.setColor(item.key, col.key)}
                        />
                      ))}
                    </div>
                    <DropdownMenuItem onSelect={() => h.setColor(item.key, undefined)}>Default</DropdownMenuItem>
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger><FolderInput size={14} /> Move to group</DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {h.existingGroups.map(g => (
                      <DropdownMenuItem key={g} onSelect={() => h.setGroup(item, g)}>
                        {meta.group === g && <Check size={13} />} {g}
                      </DropdownMenuItem>
                    ))}
                    {h.existingGroups.length > 0 && <DropdownMenuSeparator />}
                    <DropdownMenuItem onSelect={() => onNewGroup(item.key)}>New group…</DropdownMenuItem>
                    {meta.group && <DropdownMenuItem onSelect={() => h.setGroup(item, undefined)}>None</DropdownMenuItem>}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>

                <DropdownMenuSeparator />

                <DropdownMenuItem onSelect={() => h.enterSelection(item.key)}>
                  <CheckSquare size={14} /> Select
                </DropdownMenuItem>
                <DropdownMenuItem className="text-[var(--feedback-error)]" onSelect={() => h.deleteItem(item)}>
                  <Trash2 size={14} /> Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        )}
      </div>
    </div>
  );
}

export const ConversationRow = memo(ConversationRowImpl);

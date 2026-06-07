/**
 * ConversationRow — one conversation entry in the list (open tab or past), with
 * the consolidated `⋯` actions menu (rename / pin / archive / icon / color /
 * move-to-group / select / delete), a multi-select checkbox, and a quick close
 * for open tabs. Pure presentation over `useConversationList`'s handlers.
 */

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
import type { ConversationItem, useConversationList } from '../../hooks/useConversationList';

type ListApi = ReturnType<typeof useConversationList>;

interface ConversationRowProps {
  item: ConversationItem;
  c: ListApi;
  onOpenIconPicker: (key: string) => void;
  onNewGroup: (key: string) => void;
}

export function ConversationRow({ item, c, onOpenIconPicker, onNewGroup }: ConversationRowProps) {
  const { meta } = item;
  const color = conversationColorValue(meta.color);
  const editing = c.editingKey === item.key;
  const busy = c.deletingId === item.key || c.restoringId === item.key;
  const isActive = item.kind === 'tab' && item.tabId === c.activeTabId;
  const checked = c.selected.has(item.key);

  const Icon = meta.icon ? getAvatarIcon(meta.icon)
    : item.isStreaming ? Radio
    : item.kind === 'server' ? Download
    : MessageSquare;

  const onClick = () => {
    if (editing) return;
    if (c.selectionMode) { c.toggleSelect(item.key); return; }
    c.openItem(item);
  };

  return (
    <div
      className={`history-item ${isActive ? 'active' : ''} ${checked ? 'selected' : ''} ${item.kind === 'server' ? 'history-item-server' : ''}`}
      onClick={onClick}
    >
      {c.selectionMode && (
        <Checkbox
          checked={checked}
          onCheckedChange={() => c.toggleSelect(item.key)}
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
            value={c.draftTitle}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => c.setDraftTitle(e.target.value)}
            onBlur={() => c.commitRename(item)}
            onKeyDown={(e) => c.renameKeyDown(e, item)}
            aria-label="Rename conversation"
          />
        ) : (
          <span className="history-item-title" onDoubleClick={(e) => { e.stopPropagation(); c.startRename(item); }}>
            {meta.pinned && <Pin size={11} className="history-item-pinflag" />}
            {item.title}
          </span>
        )}
        <span className="history-item-meta">
          {item.messageCount} messages · {c.formatDate(item.lastMessageAt)}
        </span>
        {item.preview && <span className="history-item-preview">{item.preview}</span>}
      </div>

      <div className="history-item-actions">
        {busy ? (
          <div className="history-item-loading"><Loader2 size={12} className="spin" /></div>
        ) : c.selectionMode ? null : (
          <>
            {item.kind === 'tab' && (
              <button className="history-item-action" onClick={(e) => { e.stopPropagation(); c.closeOpenTab(item); }} title="Close (keeps on server)">
                <X size={12} />
              </button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="history-item-action" onClick={(e) => e.stopPropagation()} title="More" aria-label="Conversation actions">
                  <MoreHorizontal size={14} />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                <DropdownMenuItem onSelect={() => c.startRename(item)}>
                  <Pencil size={14} /> Rename
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => c.togglePin(item)}>
                  {meta.pinned ? <><PinOff size={14} /> Unpin</> : <><Pin size={14} /> Pin</>}
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => c.toggleArchive(item)}>
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
                          onClick={() => c.setColor(item.key, col.key)}
                        />
                      ))}
                    </div>
                    <DropdownMenuItem onSelect={() => c.setColor(item.key, undefined)}>Default</DropdownMenuItem>
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger><FolderInput size={14} /> Move to group</DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {c.existingGroups.map(g => (
                      <DropdownMenuItem key={g} onSelect={() => c.setGroup(item, g)}>
                        {meta.group === g && <Check size={13} />} {g}
                      </DropdownMenuItem>
                    ))}
                    {c.existingGroups.length > 0 && <DropdownMenuSeparator />}
                    <DropdownMenuItem onSelect={() => onNewGroup(item.key)}>New group…</DropdownMenuItem>
                    {meta.group && <DropdownMenuItem onSelect={() => c.setGroup(item, undefined)}>None</DropdownMenuItem>}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>

                <DropdownMenuSeparator />

                <DropdownMenuItem onSelect={() => c.enterSelection(item.key)}>
                  <CheckSquare size={14} /> Select
                </DropdownMenuItem>
                <DropdownMenuItem className="text-[var(--feedback-error)]" onSelect={() => c.deleteItem(item)}>
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

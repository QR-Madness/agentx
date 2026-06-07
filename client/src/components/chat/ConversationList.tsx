/**
 * ConversationList — the searchable, meta-driven conversation list shared by the
 * desktop Conversations sidebar and the mobile drawer. Renders sections (Pinned /
 * custom groups / Resume Running / Open / Past-by-recency / Archived), a
 * multi-select bulk bar, and hosts the single icon picker + inline new-group
 * input. All logic lives in `useConversationList`.
 */

import { useState } from 'react';
import {
  Search, X, Loader2, Radio, ChevronDown, ChevronRight,
  Pin, Archive, Trash2, CheckSquare, Star,
} from 'lucide-react';
import { useConversationList } from '../../hooks/useConversationList';
import { ConversationRow } from './ConversationRow';
import { AvatarPicker } from '../common/AvatarPicker';
import { getMeta } from '../../lib/conversationMeta';
import './ConversationList.css';

interface ConversationListProps {
  onActivated: () => void;
  autoFocusSearch?: boolean;
}

export function ConversationList({ onActivated, autoFocusSearch = true }: ConversationListProps) {
  const c = useConversationList({ onActivated, autoFocusSearch });
  const [iconPickerKey, setIconPickerKey] = useState<string | null>(null);
  const [newGroupKey, setNewGroupKey] = useState<string | null>(null);
  const [newGroupName, setNewGroupName] = useState('');

  const rowProps = { c, onOpenIconPicker: setIconPickerKey, onNewGroup: (key: string) => { setNewGroupKey(key); setNewGroupName(''); } };

  const commitNewGroup = () => {
    const name = newGroupName.trim();
    if (name && newGroupKey) c.setGroupByKey(newGroupKey, name);
    setNewGroupKey(null);
  };

  return (
    <>
      {c.selectionMode ? (
        <div className="conv-bulk-bar">
          <span className="conv-bulk-count">{c.selected.size} selected</span>
          <div className="conv-bulk-actions">
            <button className="history-item-action" title="Pin" onClick={c.bulkPin}><Pin size={15} /></button>
            <button className="history-item-action" title="Archive" onClick={c.bulkArchive}><Archive size={15} /></button>
            <button className="history-item-action text-[var(--feedback-error)]" title="Delete" onClick={c.bulkDelete}><Trash2 size={15} /></button>
            <button className="history-item-action" title="Done" onClick={c.clearSelection}><X size={15} /></button>
          </div>
        </div>
      ) : (
        <div className="history-search">
          <Search size={14} />
          <input
            ref={c.searchRef}
            type="text"
            value={c.searchQuery}
            onChange={(e) => c.setSearchQuery(e.target.value)}
            placeholder="Search conversations..."
          />
          {c.searchQuery && (
            <button className="clear-search" onClick={() => c.setSearchQuery('')}><X size={12} /></button>
          )}
        </div>
      )}

      {newGroupKey && (
        <div className="conv-newgroup">
          <input
            autoFocus
            value={newGroupName}
            placeholder="New group name…"
            onChange={(e) => setNewGroupName(e.target.value)}
            onBlur={commitNewGroup}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitNewGroup(); } else if (e.key === 'Escape') setNewGroupKey(null); }}
            aria-label="New group name"
          />
        </div>
      )}

      <div className="history-list">
        {/* Pinned */}
        {c.pinned.length > 0 && (
          <>
            <div className="history-section-label"><Star size={11} /> Pinned</div>
            {c.pinned.map(it => <ConversationRow key={it.key} item={it} {...rowProps} />)}
          </>
        )}

        {/* Custom groups */}
        {c.groups.map(([group, convs]) => {
          const collapsed = c.groupsCollapsed.has(group);
          return (
            <div key={group}>
              <button className="history-section-label history-section-toggle" onClick={() => c.toggleGroupCollapse(group)}>
                {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />} {group} <span className="history-section-count">{convs.length}</span>
              </button>
              {!collapsed && convs.map(it => <ConversationRow key={it.key} item={it} {...rowProps} />)}
            </div>
          );
        })}

        {/* Resume Running */}
        {c.filteredLiveRuns.length > 0 && (
          <>
            <div className="history-section-label">Resume Running</div>
            {c.filteredLiveRuns.map(run => (
              <div key={run.run_id} className="history-item history-item-running" onClick={() => c.handleResume(run)} title="Reopen this run and continue streaming">
                <div className="history-item-icon">
                  {c.restoringId === run.run_id ? <Loader2 size={14} className="spin" /> : <Radio size={14} className="history-running-dot" />}
                </div>
                <div className="history-item-info">
                  <span className="history-item-title">{run.message.slice(0, 60) || 'Running conversation'}</span>
                  <span className="history-item-meta">running · {c.formatDate(run.updated_at)}</span>
                </div>
              </div>
            ))}
          </>
        )}

        {/* Open */}
        {c.openItems.length > 0 && (
          <>
            <div className="history-section-label">Open</div>
            {c.openItems.map(it => <ConversationRow key={it.key} item={it} {...rowProps} />)}
          </>
        )}

        {/* Past by recency */}
        {c.pastByBucket.map(([bucket, convs]) => (
          <div key={bucket}>
            <div className="history-section-label">{bucket}</div>
            {convs.map(it => <ConversationRow key={it.key} item={it} {...rowProps} />)}
          </div>
        ))}

        {/* Archived */}
        {c.archived.length > 0 && (
          <>
            <button className="history-section-label history-section-toggle" onClick={c.toggleArchivedCollapse}>
              {c.archivedCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />} Archived <span className="history-section-count">{c.archived.length}</span>
            </button>
            {!c.archivedCollapsed && c.archived.map(it => <ConversationRow key={it.key} item={it} {...rowProps} />)}
          </>
        )}

        {c.isLoadingHistory && c.totalCount === 0 && (
          <div className="history-empty"><Loader2 size={16} className="spin" /><span>Loading history...</span></div>
        )}
        {!c.isLoadingHistory && c.totalCount === 0 && (
          <div className="history-empty">{c.searchQuery ? <>No conversations match &quot;{c.searchQuery}&quot;</> : <>No conversations yet</>}</div>
        )}
      </div>

      {!c.selectionMode && (
        <div className="history-footer">
          <button className="conv-select-toggle" onClick={() => c.enterSelection()}>
            <CheckSquare size={12} /> Select
          </button>
          <span>{c.openCount} open</span>
        </div>
      )}

      {/* Single controlled icon picker for the targeted conversation */}
      {iconPickerKey && (
        <AvatarPicker
          hideTrigger
          open
          value={getMeta(iconPickerKey).icon ?? ''}
          onChange={(id) => { c.setIcon(iconPickerKey, id); setIconPickerKey(null); }}
          onOpenChange={(o) => { if (!o) setIconPickerKey(null); }}
        />
      )}
    </>
  );
}

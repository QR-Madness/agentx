/**
 * ConversationList — the searchable conversation list body, shared by the
 * desktop Conversations sidebar (ConversationSidebar) and the mobile
 * Conversations drawer (ConversationsDrawerContent). All logic lives in
 * `useConversationList`; this is presentation only.
 *
 * Extension seam (conversation-management tools, next): rows are factored so a
 * selection checkbox / pin / archive affordance can be added per-row, and a
 * bulk-action bar can replace the footer — without touching the data layer.
 */

import { Search, Trash2, MessageSquare, X, Download, Loader2, Radio, Pencil } from 'lucide-react';
import { useConversationList } from '../../hooks/useConversationList';
import './ConversationList.css';

interface ConversationListProps {
  /** Called after a conversation is switched/restored/resumed so the host dismisses. */
  onActivated: () => void;
  autoFocusSearch?: boolean;
}

export function ConversationList({ onActivated, autoFocusSearch = true }: ConversationListProps) {
  const c = useConversationList({ onActivated, autoFocusSearch });

  return (
    <>
      <div className="history-search">
        <Search size={14} />
        <input
          ref={c.searchRef}
          type="text"
          value={c.searchQuery}
          onChange={(e) => c.setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            // Enter opens the first visible result (open tab, else past conv).
            if (e.key !== 'Enter') return;
            if (c.filteredTabs.length > 0) c.handleSelect(c.filteredTabs[0].id);
            else if (c.pastByBucket.length > 0) c.handleRestore(e as unknown as React.MouseEvent, c.pastByBucket[0][1][0].conversation_id);
          }}
          placeholder="Search conversations..."
        />
        {c.searchQuery && (
          <button className="clear-search" onClick={() => c.setSearchQuery('')}>
            <X size={12} />
          </button>
        )}
      </div>

      <div className="history-list">
        {/* Resume Running — runs whose owning tab was closed */}
        {c.filteredLiveRuns.length > 0 && (
          <>
            <div className="history-section-label">Resume Running</div>
            {c.filteredLiveRuns.map(run => (
              <div
                key={run.run_id}
                className="history-item history-item-running"
                onClick={(e) => c.handleResume(e, run)}
                title="Reopen this run and continue streaming"
              >
                <div className="history-item-icon">
                  {c.restoringId === run.run_id
                    ? <Loader2 size={14} className="spin" />
                    : <Radio size={14} className="history-running-dot" />}
                </div>
                <div className="history-item-info">
                  <span className="history-item-title">{run.message.slice(0, 60) || 'Running conversation'}</span>
                  <span className="history-item-meta">running · {c.formatDate(run.updated_at)}</span>
                </div>
              </div>
            ))}
          </>
        )}

        {/* Open Tabs */}
        {c.filteredTabs.length > 0 && (
          <>
            <div className="history-section-label">Open</div>
            {c.filteredTabs.map(tab => (
              <div
                key={tab.id}
                className={`history-item ${tab.id === c.activeTabId ? 'active' : ''}`}
                onClick={() => c.editingId === tab.id ? undefined : c.handleSelect(tab.id)}
              >
                <div className="history-item-icon">
                  {tab.isStreaming ? <Radio size={14} className="history-running-dot" /> : <MessageSquare size={14} />}
                </div>
                <div className="history-item-info">
                  {c.editingId === tab.id ? (
                    <input
                      className="history-item-rename"
                      value={c.draftTitle}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => c.setDraftTitle(e.target.value)}
                      onBlur={() => c.commitRename('tab')}
                      onKeyDown={(e) => c.renameKeyDown(e, 'tab')}
                      aria-label="Rename conversation"
                    />
                  ) : (
                    <span className="history-item-title" onDoubleClick={(e) => c.startRename(e, tab.id, tab.title)}>
                      {tab.title}
                    </span>
                  )}
                  <span className="history-item-meta">
                    {tab.messages.length} messages · {c.formatDate(tab.lastMessageAt)}
                  </span>
                </div>
                <div className="history-item-actions">
                  {c.deletingId === tab.id ? (
                    <div className="history-item-loading"><Loader2 size={12} className="spin" /></div>
                  ) : (
                    <>
                      <button className="history-item-action" onClick={(e) => c.startRename(e, tab.id, tab.title)} title="Rename">
                        <Pencil size={12} />
                      </button>
                      <button className="history-item-action" onClick={(e) => c.handleClose(e, tab.id)} title="Close (keeps on server)">
                        <X size={12} />
                      </button>
                      <button className="history-item-action history-item-action-danger" onClick={(e) => c.handleDeleteTab(e, tab.id, tab.title)} title="Delete conversation">
                        <Trash2 size={12} />
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </>
        )}

        {/* Past Conversations (server), grouped by recency */}
        {c.pastByBucket.map(([bucket, convs]) => (
          <div key={bucket}>
            <div className="history-section-label">{bucket}</div>
            {convs.map(conv => {
              const displayTitle = c.getDisplayTitle(conv.conversation_id, conv.title);
              return (
                <div
                  key={conv.conversation_id}
                  className="history-item history-item-server"
                  onClick={(e) => c.editingId === conv.conversation_id ? undefined : c.handleRestore(e, conv.conversation_id)}
                >
                  <div className="history-item-icon"><Download size={14} /></div>
                  <div className="history-item-info">
                    {c.editingId === conv.conversation_id ? (
                      <input
                        className="history-item-rename"
                        value={c.draftTitle}
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => c.setDraftTitle(e.target.value)}
                        onBlur={() => c.commitRename('server')}
                        onKeyDown={(e) => c.renameKeyDown(e, 'server')}
                        aria-label="Rename conversation"
                      />
                    ) : (
                      <span className="history-item-title" onDoubleClick={(e) => c.startRename(e, conv.conversation_id, displayTitle)}>
                        {displayTitle}
                      </span>
                    )}
                    <span className="history-item-meta">
                      {conv.message_count} messages · {c.formatDate(conv.last_message_at)}
                    </span>
                    {conv.preview && <span className="history-item-preview">{conv.preview}</span>}
                  </div>
                  <div className="history-item-actions">
                    {c.restoringId === conv.conversation_id || c.deletingId === conv.conversation_id ? (
                      <div className="history-item-loading"><Loader2 size={12} className="spin" /></div>
                    ) : (
                      <>
                        <button className="history-item-action" onClick={(e) => c.startRename(e, conv.conversation_id, displayTitle)} title="Rename">
                          <Pencil size={12} />
                        </button>
                        <button className="history-item-action history-item-action-danger" onClick={(e) => c.handleDeleteServerConversation(e, conv.conversation_id, displayTitle)} title="Delete conversation">
                          <Trash2 size={12} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}

        {c.isLoadingHistory && c.totalCount === 0 && (
          <div className="history-empty"><Loader2 size={16} className="spin" /><span>Loading history...</span></div>
        )}

        {!c.isLoadingHistory && c.totalCount === 0 && (
          <div className="history-empty">
            {c.searchQuery ? <>No conversations match &quot;{c.searchQuery}&quot;</> : <>No conversations yet</>}
          </div>
        )}
      </div>

      <div className="history-footer">
        <span>{c.openCount} open{c.pastCount > 0 ? ` · ${c.pastCount} past` : ''}</span>
      </div>
    </>
  );
}

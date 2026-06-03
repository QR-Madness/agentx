/**
 * SourcesPanel — the conversation Bibliography: a static, numbered, deduped list
 * of every source cited in the active conversation (derived live from its
 * `citation` exhibits via `buildBibliography`). Opened as a right-side drawer.
 */

import { useMemo } from 'react';
import { BookMarked, ExternalLink } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { buildBibliography } from '../../lib/bibliography';
import { isHttpUrl, urlHost } from '../../lib/links';

export function SourcesPanel() {
  const { activeTab } = useConversation();
  const entries = useMemo(
    () => buildBibliography(activeTab?.messages ?? []),
    [activeTab?.messages],
  );

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <BookMarked size={18} className="text-accent" />
        <h2 className="text-base font-semibold text-fg">Sources</h2>
        <span className="text-sm text-fg-muted">({entries.length})</span>
      </div>

      {entries.length === 0 ? (
        <p className="text-sm text-fg-muted">
          No sources cited in this conversation yet. They appear here as the agent searches the web
          or cites references.
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {entries.map((e) => (
            <li
              key={e.n}
              className="flex items-start gap-2 rounded-md border border-line bg-surface-raised p-2.5 text-sm"
            >
              <span className="shrink-0 font-mono text-xs text-fg-muted">[{e.n}]</span>
              <div className="flex min-w-0 flex-1 flex-col">
                {isHttpUrl(e.url) ? (
                  <a
                    href={e.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 truncate font-medium text-accent hover:underline"
                  >
                    <span className="truncate">{e.label}</span>
                    <ExternalLink size={12} className="shrink-0" />
                  </a>
                ) : (
                  <span className="truncate font-medium text-fg">{e.label}</span>
                )}
                {isHttpUrl(e.url) && (
                  <span className="truncate text-xs text-fg-muted">{urlHost(e.url)}</span>
                )}
              </div>
              {e.kind === 'active' && (
                <span className="shrink-0 rounded bg-accent-tertiary px-1.5 py-0.5 text-[10px] font-medium text-accent">
                  active
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

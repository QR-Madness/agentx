/**
 * CitationElement — agent-cited sources. The card always carries a "Sources"
 * header (so it's legible even when every source is passive — the common case
 * for auto-captured web_search results). Active sources fold out (icon + label,
 * expanding to a quote + link); passive sources are record-keeping: shown as an
 * open list when there are no active sources, else tucked under a disclosure.
 * Long passive lists collapse to the first few with a "+N more" toggle. URLs are
 * linked only when http(s) (see lib/links).
 */

import { useState } from 'react';
import { Globe, Database, FileText, Link2, BookMarked, type LucideIcon } from 'lucide-react';
import { isHttpUrl, urlHost } from '../../../lib/links';
import type { CitationSource } from '../../../lib/exhibits';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

const SOURCE_ICON: Record<NonNullable<CitationSource['source_type']>, LucideIcon> = {
  web: Globe,
  memory: Database,
  doc: FileText,
};

/** How many passive sources to show before the "+N more" toggle. */
const PASSIVE_PREVIEW = 5;

function sourceIcon(s: CitationSource): LucideIcon {
  return (s.source_type && SOURCE_ICON[s.source_type]) || Link2;
}

/** A link when http(s), else inert text. */
function SourceLabel({ source }: { source: CitationSource }) {
  if (isHttpUrl(source.url)) {
    return (
      <a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-accent hover:underline"
      >
        {source.label}
      </a>
    );
  }
  return <span>{source.label}</span>;
}

function ActiveSource({ source }: { source: CitationSource }) {
  const Icon = sourceIcon(source);
  return (
    <details className="rounded-md border border-line bg-surface-sunken">
      <summary className="flex cursor-pointer list-none items-center gap-2 p-2 text-sm">
        <Icon size={14} className="shrink-0 text-fg-muted" />
        <span className="min-w-0 flex-1 truncate font-medium text-fg">{source.label}</span>
        {isHttpUrl(source.url) && (
          <span className="shrink-0 text-xs text-fg-muted">{urlHost(source.url)}</span>
        )}
      </summary>
      <div className="flex flex-col gap-2 px-3 pb-2.5 text-sm">
        {source.quote && (
          <blockquote className="border-l-2 border-line pl-2 text-fg-secondary italic">
            {source.quote}
          </blockquote>
        )}
        {isHttpUrl(source.url) && (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="truncate text-xs text-accent hover:underline"
          >
            {source.url}
          </a>
        )}
      </div>
    </details>
  );
}

/** Compact passive list, capped to PASSIVE_PREVIEW with a "+N more" expander. */
function PassiveSources({ sources }: { sources: CitationSource[] }) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? sources : sources.slice(0, PASSIVE_PREVIEW);
  const hidden = sources.length - shown.length;
  return (
    <ul className="flex flex-col gap-1 pl-1">
      {shown.map((s, i) => (
        <li key={i} className="flex min-w-0 items-center gap-1.5 text-sm text-fg-secondary">
          <Link2 size={12} className="shrink-0 text-fg-muted" />
          <SourceLabel source={s} />
          {isHttpUrl(s.url) && (
            <span className="truncate text-xs text-fg-muted">{urlHost(s.url)}</span>
          )}
        </li>
      ))}
      {hidden > 0 && (
        <li>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="text-xs font-medium text-accent hover:underline"
          >
            +{hidden} more
          </button>
        </li>
      )}
    </ul>
  );
}

function CitationElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'citation') return null;
  const active = element.sources.filter((s) => s.kind === 'active');
  const passive = element.sources.filter((s) => s.kind !== 'active');

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-sm font-medium text-fg">
        <BookMarked size={14} className="shrink-0 text-fg-muted" />
        <span>{element.title || 'Sources'}</span>
        <span className="text-xs font-normal text-fg-muted">({element.sources.length})</span>
      </div>

      {active.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {active.map((s, i) => (
            <ActiveSource key={i} source={s} />
          ))}
        </div>
      )}

      {passive.length > 0 &&
        (active.length > 0 ? (
          // With active sources present, keep passive record-keeping collapsed.
          <details className="text-sm">
            <summary className="cursor-pointer list-none text-xs font-medium text-fg-muted hover:text-fg">
              More sources ({passive.length})
            </summary>
            <div className="mt-1.5">
              <PassiveSources sources={passive} />
            </div>
          </details>
        ) : (
          // Passive-only (e.g. auto-captured web results): show them, don't bury.
          <PassiveSources sources={passive} />
        ))}
    </div>
  );
}

export const CitationElement = memoElement(CitationElementImpl);

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

/** Prefer the human page title; fall back to the host for bare-URL sources. */
function sourceText(s: CitationSource): string {
  if (s.label && s.label.trim()) return s.label;
  if (isHttpUrl(s.url)) return urlHost(s.url);
  return s.url || 'source';
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

/** One passive source rendered as a compact numbered chip. */
function SourceChip({ source, n }: { source: CitationSource; n: number }) {
  const Icon = sourceIcon(source);
  const text = sourceText(source);
  const host = isHttpUrl(source.url) ? urlHost(source.url) : null;
  // Number + icon are decorative — keep them out of the link's accessible name.
  const inner = (
    <>
      <span aria-hidden="true" className="shrink-0 font-mono text-[10px] text-fg-muted tabular-nums">
        {n}
      </span>
      <Icon
        size={12}
        aria-hidden="true"
        className="shrink-0 text-fg-muted transition-colors group-hover:text-accent"
      />
      <span className="truncate">{text}</span>
    </>
  );
  const cls =
    'group inline-flex max-w-[14rem] items-center gap-1.5 rounded-full border border-line ' +
    'bg-surface-sunken px-2.5 py-1 text-xs text-fg-secondary transition-colors ' +
    'hover:border-line-strong hover:text-fg';
  if (isHttpUrl(source.url)) {
    return (
      <a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className={cls}
        title={`${text}${host ? ` — ${host}` : ''}\n${source.url}`}
      >
        {inner}
      </a>
    );
  }
  return <span className={cls} title={text}>{inner}</span>;
}

/** Passive sources as a wrapped row of chips, capped with a "+N more" toggle. */
function PassiveSources({ sources }: { sources: CitationSource[] }) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? sources : sources.slice(0, PASSIVE_PREVIEW);
  const hidden = sources.length - shown.length;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {shown.map((s, i) => (
        <SourceChip key={i} source={s} n={i + 1} />
      ))}
      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="rounded-full border border-dashed border-line px-2.5 py-1 text-xs font-medium text-fg-muted transition-colors hover:border-line-strong hover:text-fg"
        >
          +{hidden} more
        </button>
      )}
    </div>
  );
}

function CitationElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'citation') return null;
  const active = element.sources.filter((s) => s.kind === 'active');
  const passive = element.sources.filter((s) => s.kind !== 'active');

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-fg-muted">
        <BookMarked size={12} className="shrink-0" />
        <span>{element.title || 'Sources'}</span>
        <span className="font-normal normal-case tracking-normal">· {element.sources.length}</span>
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

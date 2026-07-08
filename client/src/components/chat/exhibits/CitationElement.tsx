/**
 * CitationElement — agent-cited sources, rendered like a tool call: a slim,
 * collapsed-by-default header row (type icon + "Sources · N" + a host teaser +
 * chevron) that expands to the source list. Keeps citations from stacking as
 * disruptive full-width cards in the transcript; the Sources drawer
 * (`SourcesPanel`) owns the canonical numbered bibliography, so the inline chips
 * carry no numbers. Active sources fold out (icon + label → quote + link);
 * passive sources are record-keeping (open list when alone, else a disclosure,
 * capped with "+N more"). URLs link only when http(s) (see lib/links).
 */

import { useId, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { isHttpUrl, urlHost } from '../../../lib/links';
import { sourceIcon, sourceText, hostTeaser, dominantIcon } from '../../../lib/sources';
import type { CitationSource } from '../../../lib/exhibits';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

/** How many passive sources to show before the "+N more" toggle. */
const PASSIVE_PREVIEW = 5;

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

/** One passive source rendered as a compact chip (no number — the drawer owns numbering). */
function SourceChip({ source }: { source: CitationSource }) {
  const Icon = sourceIcon(source);
  const text = sourceText(source);
  const host = isHttpUrl(source.url) ? urlHost(source.url) : null;
  const inner = (
    <>
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
        <SourceChip key={i} source={s} />
      ))}
      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="rounded-full border border-dashed border-line bg-transparent px-2.5 py-1 text-xs font-medium text-fg-muted transition-colors hover:border-line-strong hover:text-fg"
        >
          +{hidden} more
        </button>
      )}
    </div>
  );
}

function CitationElementImpl({ element, containerTitle }: ElementRenderProps) {
  const [expanded, setExpanded] = useState(false);
  const bodyId = useId();
  if (element.type !== 'citation') return null;

  const sources = element.sources;
  const active = sources.filter((s) => s.kind === 'active');
  const passive = sources.filter((s) => s.kind !== 'active');
  const title = element.title || containerTitle || 'Sources';
  const HeaderIcon = dominantIcon(sources);
  const teaser = hostTeaser(sources);
  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-controls={bodyId}
        className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left transition-colors hover:bg-surface-hover"
      >
        <HeaderIcon size={14} className="shrink-0 text-fg-muted" />
        <span className="shrink-0 text-xs font-semibold uppercase tracking-wide text-fg-muted">
          {title}
        </span>
        <span className="shrink-0 text-xs text-fg-muted">· {sources.length}</span>
        {teaser && (
          <span className="min-w-0 flex-1 truncate text-xs text-fg-muted/80">{teaser}</span>
        )}
        <Chevron size={14} className="ml-auto shrink-0 text-fg-muted" />
      </button>

      {expanded && (
        <div id={bodyId} className="mt-1.5 flex flex-col gap-2 px-1.5">
          {active.length > 0 && (
            <div className="flex flex-col gap-1.5">
              {active.map((s, i) => (
                <ActiveSource key={i} source={s} />
              ))}
            </div>
          )}
          {passive.length > 0 &&
            (active.length > 0 ? (
              <details className="text-sm">
                <summary className="cursor-pointer list-none text-xs font-medium text-fg-muted hover:text-fg">
                  More sources ({passive.length})
                </summary>
                <div className="mt-1.5">
                  <PassiveSources sources={passive} />
                </div>
              </details>
            ) : (
              <PassiveSources sources={passive} />
            ))}
        </div>
      )}
    </div>
  );
}

export const CitationElement = memoElement(CitationElementImpl);

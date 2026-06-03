/**
 * CitationElement — agent-cited sources. Active sources fold out (icon + label,
 * expanding to a quote + link); passive sources sit under a compact "Sources (N)"
 * disclosure (record-keeping). URLs are linked only when http(s) (see lib/links).
 */

import { Globe, Database, FileText, Link2, type LucideIcon } from 'lucide-react';
import { isHttpUrl, urlHost } from '../../../lib/links';
import type { CitationSource } from '../../../lib/exhibits';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

const SOURCE_ICON: Record<NonNullable<CitationSource['source_type']>, LucideIcon> = {
  web: Globe,
  memory: Database,
  doc: FileText,
};

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

function CitationElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'citation') return null;
  const active = element.sources.filter((s) => s.kind === 'active');
  const passive = element.sources.filter((s) => s.kind !== 'active');

  return (
    <div className="flex flex-col gap-2">
      {element.title && <div className="text-sm font-medium text-fg">{element.title}</div>}

      {active.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {active.map((s, i) => (
            <ActiveSource key={i} source={s} />
          ))}
        </div>
      )}

      {passive.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer list-none text-xs font-medium text-fg-muted hover:text-fg">
            Sources ({passive.length})
          </summary>
          <ul className="mt-1.5 flex flex-col gap-1 pl-1">
            {passive.map((s, i) => (
              <li key={i} className="flex items-center gap-1.5 text-fg-secondary">
                <Link2 size={12} className="shrink-0 text-fg-muted" />
                <SourceLabel source={s} />
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export const CitationElement = memoElement(CitationElementImpl);

/**
 * Source presentation helpers — shared by the inline citation block
 * (`CitationElement`) and the Sources drawer (`SourcesPanel`) so the two can't
 * drift on icons, labels, or the collapsed host teaser. Pure + display-only.
 */

import { Globe, Database, FileText, Link2, BookMarked, type LucideIcon } from 'lucide-react';
import { isHttpUrl, urlHost } from './links';
import type { CitationSource } from './exhibits';

const SOURCE_ICON: Record<NonNullable<CitationSource['source_type']>, LucideIcon> = {
  web: Globe,
  memory: Database,
  doc: FileText,
};

/** How many hosts the collapsed preview names before folding into "+N". */
const TEASER_HOSTS = 2;

/** The type icon for a source; bare links fall back to a generic chain. */
export function sourceIcon(s: Pick<CitationSource, 'source_type'>): LucideIcon {
  return (s.source_type && SOURCE_ICON[s.source_type]) || Link2;
}

/** Prefer the human page title; fall back to the host for bare-URL sources. */
export function sourceText(s: CitationSource): string {
  if (s.label && s.label.trim()) return s.label;
  if (isHttpUrl(s.url)) return urlHost(s.url);
  return s.url || 'source';
}

/** A short host label for teasers: the URL host, else the source-type word. */
function hostLabel(s: CitationSource): string {
  if (isHttpUrl(s.url)) return urlHost(s.url);
  return s.source_type || 'source';
}

/**
 * Collapsed-preview teaser: the first couple distinct hosts joined by ", ",
 * with "+N" for the remainder. Empty string when there's nothing to hint.
 */
export function hostTeaser(sources: CitationSource[]): string {
  const hosts: string[] = [];
  const seen = new Set<string>();
  for (const s of sources) {
    const h = hostLabel(s);
    const key = h.toLowerCase();
    if (!h || seen.has(key)) continue;
    seen.add(key);
    hosts.push(h);
  }
  if (hosts.length === 0) return '';
  const shown = hosts.slice(0, TEASER_HOSTS);
  const hidden = hosts.length - shown.length;
  return hidden > 0 ? `${shown.join(', ')} +${hidden}` : shown.join(', ');
}

/**
 * Header glyph for a set of sources: the shared type icon when every source is
 * the same type, else the neutral "sources" book icon.
 */
export function dominantIcon(sources: CitationSource[]): LucideIcon {
  const types = new Set(sources.map((s) => s.source_type));
  if (types.size === 1) {
    const only = sources[0]?.source_type;
    if (only && SOURCE_ICON[only]) return SOURCE_ICON[only];
  }
  return BookMarked;
}

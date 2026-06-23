/**
 * Exhibits — typed, declarative agent-authored content.
 *
 * An **Exhibit** is one presented unit (a tree of typed **Elements** arranged
 * by a `layout`) the agent builds by calling the `present_exhibit` tool. A
 * conversation accumulates exhibits into a **Gallery**. Slice 1 ships the
 * `mermaid` element + `stack` layout; new element types are a registry entry
 * (see `components/chat/exhibits/elementRegistry.ts`).
 *
 * The wire shape ({@link ExhibitWire}) uses snake_case `schema_version`; the
 * UI shape ({@link Exhibit}) is camelCase. {@link exhibitFromWire} bridges them
 * for both the live stream and the history-restore path.
 */

/** A mermaid diagram element (rendered client-side to SVG). */
export interface MermaidElement {
  type: 'mermaid';
  content: string;
  title?: string;
}

/** An interactive choice — the user picks an option, fed back as their next turn. */
export interface ChoiceElement {
  type: 'choice';
  prompt?: string;
  options: string[];
  title?: string;
}

/** A structured table (sortable / scrollable / responsive / expandable). */
export interface TableElement {
  type: 'table';
  columns: string[];
  rows: string[][];
  caption?: string;
  title?: string;
}

/** One cited source. `active` folds out (with a quote); `passive` is record-keeping. */
export interface CitationSource {
  label: string;
  url?: string;
  quote?: string;
  kind: 'active' | 'passive';
  source_type?: 'web' | 'memory' | 'doc';
}

/** A set of cited sources. */
export interface CitationElement {
  type: 'citation';
  sources: CitationSource[];
  title?: string;
}

export interface ImageElement {
  type: 'image';
  /** A served-blob path (…/documents/{doc}/raw); resolved to an authed object URL at render. */
  url: string;
  alt?: string;
  title?: string;
}

/** Union of element kinds. Widen as new element types ship. */
export type ExhibitElement =
  | MermaidElement
  | ChoiceElement
  | TableElement
  | CitationElement
  | ImageElement;

/** UI-shape exhibit. */
export interface Exhibit {
  schemaVersion: number;
  id: string;
  title?: string;
  layout: 'stack';
  elements: ExhibitElement[];
}

/** One raw element on the wire (all element fields are optional per type). */
export interface ExhibitWireElement {
  type: string;
  content?: string;
  options?: string[];
  prompt?: string;
  columns?: string[];
  rows?: unknown[][];
  caption?: string;
  sources?: Array<{
    label?: string;
    url?: string;
    quote?: string;
    kind?: string;
    source_type?: string;
  }>;
  url?: string;
  alt?: string;
  title?: string;
}

/** Raw wire shape emitted by the backend `exhibit` SSE event. */
export interface ExhibitWire {
  schema_version?: number;
  id: string;
  title?: string;
  layout?: string;
  elements: ExhibitWireElement[];
}

/** Element types the client can render. Unknown types fall back to source-as-code. */
const KNOWN_ELEMENT_TYPES = new Set<ExhibitElement['type']>([
  'mermaid',
  'choice',
  'table',
  'citation',
  'image',
]);

export function isKnownElementType(type: string): type is ExhibitElement['type'] {
  return KNOWN_ELEMENT_TYPES.has(type as ExhibitElement['type']);
}

/** Map one wire element into the typed UI {@link ExhibitElement} shape. */
function elementFromWire(el: ExhibitWireElement): ExhibitElement {
  if (el.type === 'choice') {
    return { type: 'choice', prompt: el.prompt, options: el.options ?? [], title: el.title };
  }
  if (el.type === 'table') {
    return {
      type: 'table',
      columns: el.columns ?? [],
      rows: (el.rows ?? []).map((row) => (row ?? []).map((cell) => String(cell ?? ''))),
      caption: el.caption,
      title: el.title,
    };
  }
  if (el.type === 'citation') {
    return {
      type: 'citation',
      sources: (el.sources ?? []).map((s) => ({
        label: s.label ?? '',
        url: s.url,
        quote: s.quote,
        kind: s.kind === 'active' ? 'active' : 'passive',
        source_type:
          s.source_type === 'web' || s.source_type === 'memory' || s.source_type === 'doc'
            ? s.source_type
            : undefined,
      })),
      title: el.title,
    };
  }
  if (el.type === 'image') {
    return { type: 'image', url: el.url ?? '', alt: el.alt, title: el.title };
  }
  // mermaid (and any unknown type) — unknown types survive at runtime; the
  // element registry misses and ExhibitBubble shows a source-as-code fallback.
  return { type: el.type as 'mermaid', content: el.content ?? '', title: el.title };
}

/** Map a wire exhibit (snake_case) into the UI {@link Exhibit} shape. */
export function exhibitFromWire(w: ExhibitWire): Exhibit {
  return {
    schemaVersion: w.schema_version ?? 1,
    id: w.id,
    title: w.title,
    layout: 'stack',
    elements: (w.elements ?? []).map(elementFromWire),
  };
}

/** Max sources kept when auto-capturing (mirrors backend MAX_CITATION_SOURCES). */
const MAX_CITATION_SOURCES = 50;

/**
 * Build a passive `citation` exhibit from `web_search` results — the client
 * mirror of the backend `citation_exhibit_from_web_search`. Dedupes by URL,
 * caps, returns `null` when there's nothing to show. Used by the history-restore
 * path so a searched-then-cited turn looks the same on reload as it did live
 * (where the backend emits the exhibit directly).
 */
export function citationExhibitFromWebSearch(results: unknown, id: string): Exhibit | null {
  if (!Array.isArray(results)) return null;
  const sources: CitationSource[] = [];
  const seen = new Set<string>();
  for (const r of results) {
    if (!r || typeof r !== 'object') continue;
    const row = r as Record<string, unknown>;
    const url = typeof row.url === 'string' ? row.url.trim() : '';
    const title = typeof row.title === 'string' ? row.title.trim() : '';
    const label = title || url;
    if (!label) continue;
    const key = url || label;
    if (seen.has(key)) continue;
    seen.add(key);
    sources.push({ label, url: url || undefined, kind: 'passive', source_type: 'web' });
    if (sources.length >= MAX_CITATION_SOURCES) break;
  }
  if (sources.length === 0) return null;
  return {
    schemaVersion: 1,
    id,
    layout: 'stack',
    elements: [{ type: 'citation', sources }],
  };
}

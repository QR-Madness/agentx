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

/** Union of element kinds. Widen as new element types ship. */
export type ExhibitElement = MermaidElement;

/** UI-shape exhibit. */
export interface Exhibit {
  schemaVersion: number;
  id: string;
  title?: string;
  layout: 'stack';
  elements: ExhibitElement[];
}

/** Raw wire shape emitted by the backend `exhibit` SSE event. */
export interface ExhibitWire {
  schema_version?: number;
  id: string;
  title?: string;
  layout?: string;
  elements: Array<{ type: string; content: string; title?: string }>;
}

/** Element types the client can render. Unknown types fall back to source-as-code. */
const KNOWN_ELEMENT_TYPES = new Set<ExhibitElement['type']>(['mermaid']);

export function isKnownElementType(type: string): type is ExhibitElement['type'] {
  return KNOWN_ELEMENT_TYPES.has(type as ExhibitElement['type']);
}

/** Map a wire exhibit (snake_case) into the UI {@link Exhibit} shape. */
export function exhibitFromWire(w: ExhibitWire): Exhibit {
  return {
    schemaVersion: w.schema_version ?? 1,
    id: w.id,
    title: w.title,
    layout: 'stack',
    elements: (w.elements ?? []).map((el) => ({
      // Unknown types survive at runtime (registry lookup misses → safe
      // source-as-code fallback); the cast keeps the UI type narrow.
      type: el.type as ExhibitElement['type'],
      content: el.content,
      title: el.title,
    })),
  };
}

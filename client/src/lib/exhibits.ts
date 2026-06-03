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

/** Union of element kinds. Widen as new element types ship. */
export type ExhibitElement = MermaidElement | ChoiceElement;

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
const KNOWN_ELEMENT_TYPES = new Set<ExhibitElement['type']>(['mermaid', 'choice']);

export function isKnownElementType(type: string): type is ExhibitElement['type'] {
  return KNOWN_ELEMENT_TYPES.has(type as ExhibitElement['type']);
}

/** Map one wire element into the typed UI {@link ExhibitElement} shape. */
function elementFromWire(el: ExhibitWireElement): ExhibitElement {
  if (el.type === 'choice') {
    return { type: 'choice', prompt: el.prompt, options: el.options ?? [], title: el.title };
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

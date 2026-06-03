/**
 * MermaidElement — renders an agent-authored Mermaid diagram to SVG.
 *
 * `mermaid` is heavy (~500KB) so it is dynamic-imported on first use and
 * initialized once with `securityLevel: 'strict'` (sanitizes diagram-embedded
 * HTML/links — the trusted, sanitized SVG is what we inject). A failed render
 * never throws: it falls back to the raw source as plain text + an error chip.
 */

import { memo, useEffect, useId, useRef, useState } from 'react';
import type { ElementRenderProps } from './types';

type MermaidApi = typeof import('mermaid')['default'];

let mermaidPromise: Promise<MermaidApi> | null = null;
let initialized = false;

async function getMermaid(): Promise<MermaidApi> {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then((m) => m.default);
  }
  const mermaid = await mermaidPromise;
  if (!initialized) {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'dark',
      fontFamily: 'inherit',
    });
    initialized = true;
  }
  return mermaid;
}

function MermaidElementImpl({ element }: ElementRenderProps) {
  // The registry only routes `mermaid` here; read its fields defensively so the
  // hooks below stay unconditional (rules of hooks).
  const content = element.type === 'mermaid' ? element.content : '';
  const title = element.type === 'mermaid' ? element.title : undefined;
  // useId is stable across renders and unique per instance; mermaid needs a
  // DOM-id-safe render target (no leading digit / colons).
  const baseId = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const renderSeq = useRef(0);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const seq = ++renderSeq.current;
    (async () => {
      try {
        const mermaid = await getMermaid();
        const { svg: rendered } = await mermaid.render(`mermaid-${baseId}-${seq}`, content);
        if (cancelled || seq !== renderSeq.current) return;
        setSvg(rendered);
        setError(null);
      } catch (e) {
        if (cancelled || seq !== renderSeq.current) return;
        setSvg(null);
        setError(e instanceof Error ? e.message : 'Failed to render diagram');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [content, baseId]);

  return (
    <figure className="m-0 flex flex-col gap-2">
      {svg && !error ? (
        <div
          className="exhibit-mermaid-svg overflow-x-auto rounded-md bg-surface-sunken p-3 [&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-w-full"
          // mermaid output is sanitized via securityLevel:'strict'; this is the
          // sanctioned way to mount its SVG.
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : error ? (
        <div className="flex flex-col gap-1.5 rounded-md border border-line bg-surface-sunken p-3">
          <span className="text-xs font-medium text-warning">Diagram failed to render</span>
          <pre className="overflow-x-auto text-xs text-fg-muted">
            <code>{content}</code>
          </pre>
        </div>
      ) : (
        <div className="rounded-md bg-surface-sunken p-3 text-xs text-fg-muted">
          Rendering diagram…
        </div>
      )}
      {title && (
        <figcaption className="text-center text-xs text-fg-muted">{title}</figcaption>
      )}
    </figure>
  );
}

// Memo on element identity only — the shared render contract also carries
// volatile choice callbacks/flags that must not trigger a (heavy) re-render.
// `element` identity is stable within a message and changes only on amend.
export const MermaidElement = memo(
  MermaidElementImpl,
  (a, b) => a.element === b.element,
);

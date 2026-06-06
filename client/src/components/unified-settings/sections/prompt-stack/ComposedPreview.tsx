/**
 * ComposedPreview — the live, read-only view of the fully composed system prompt
 * (what the agent actually receives), with an approximate token count.
 */

import { estimateTokens } from '../../../../lib/promptStack';
import { splitPlaceholders } from '../../../../lib/promptPlaceholders';
import './ComposedPreview.css';

interface ComposedPreviewProps {
  composed: string;
}

/** Render text with whitelisted {placeholder} tokens highlighted. */
export function HighlightedPrompt({ text }: { text: string }) {
  return (
    <>
      {splitPlaceholders(text).map((seg, i) =>
        seg.placeholder ? (
          <span key={i} className="prompt-placeholder-token">{seg.text}</span>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </>
  );
}

export function ComposedPreview({ composed }: ComposedPreviewProps) {
  return (
    <div className="prompt-stack__preview">
      <div className="prompt-stack__preview-head">
        <span className="prompt-stack__preview-title">Composed preview</span>
        <span className="prompt-stack__preview-count" title="Approximate">
          ~{estimateTokens(composed)} tokens
        </span>
      </div>
      <div className="prompt-stack__preview-body">
        {composed.trim() ? (
          <HighlightedPrompt text={composed} />
        ) : (
          <span className="prompt-stack__preview-empty">
            Every layer is disabled — the agent would receive no global system prompt.
          </span>
        )}
      </div>
    </div>
  );
}

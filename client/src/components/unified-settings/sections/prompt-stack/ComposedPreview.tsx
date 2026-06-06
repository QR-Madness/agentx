/**
 * ComposedPreview — the live, read-only view of the fully composed system prompt
 * (what the agent actually receives), with an approximate token count.
 */

import { estimateTokens } from '../../../../lib/promptStack';

interface ComposedPreviewProps {
  composed: string;
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
          composed
        ) : (
          <span className="prompt-stack__preview-empty">
            Every layer is disabled — the agent would receive no global system prompt.
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * EffectivePromptPreview — collapsible read-only preview of the *core* system
 * prompt an agent actually receives, composed in the exact backend order of
 * `PromptConfig.compose_system_prompt`: "Your name is {name}." → agent prompt →
 * global layer stack. (MCP tools + prompt-profile sections are appended at
 * runtime; this shows the authored core.)
 */

import { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { api } from '../../lib/api';
import { composeStack, estimateTokens } from '../../lib/promptStack';
import type { PromptLayer } from '../../lib/api/types';
import './EffectivePromptPreview.css';

interface EffectivePromptPreviewProps {
  name: string;
  agentPrompt: string;
}

export function EffectivePromptPreview({ name, agentPrompt }: EffectivePromptPreviewProps) {
  const [open, setOpen] = useState(false);
  const [globalStack, setGlobalStack] = useState<string | null>(null);

  // Lazy, non-blocking: fetch the global layers once when first expanded.
  useEffect(() => {
    if (!open || globalStack !== null) return;
    let alive = true;
    api
      .listPromptLayers()
      .then(({ layers }: { layers: PromptLayer[] }) => {
        if (alive) setGlobalStack(composeStack(layers));
      })
      .catch(() => {
        if (alive) setGlobalStack('');
      });
    return () => {
      alive = false;
    };
  }, [open, globalStack]);

  const parts = [
    name.trim() ? `Your name is ${name.trim()}.` : '',
    agentPrompt.trim(),
    globalStack ?? '',
  ].filter((p) => p.trim());
  const composed = parts.join('\n\n');

  return (
    <div className="effective-preview">
      <button
        type="button"
        className="effective-preview__toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <ChevronDown size={14} className="effective-preview__chevron" data-open={open || undefined} />
        Effective prompt preview
        {open && globalStack !== null && (
          <span className="effective-preview__count" title="Approximate">
            ~{estimateTokens(composed)} tokens
          </span>
        )}
      </button>
      {open && (
        <div className="effective-preview__body">
          {globalStack === null ? (
            <span className="effective-preview__muted">Loading…</span>
          ) : (
            <>
              <div className="effective-preview__note">Core prompt — tools &amp; sections are added at runtime.</div>
              <pre className="effective-preview__text">{composed || '(empty)'}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Prompt Stack helpers — client-side mirror of the backend layer composition
 * (`api/agentx_ai/prompts/layers.py::LayerStore.compose`) so the editor can show
 * an instant live preview without a round-trip after every keystroke.
 */

import type { PromptLayer } from './api/types';

/** A layer's effective content: override if set, else the shipped default. */
export function effectiveContent(layer: Pick<PromptLayer, 'override' | 'default'>): string {
  if (layer.override !== null && layer.override !== undefined) return layer.override;
  return layer.default ?? '';
}

/**
 * Compose the global stack exactly as the backend does: enabled layers' effective
 * content, in ascending `order`, joined by a blank line. Must stay in lockstep with
 * `LayerStore.compose()` (covered by promptStack.test.ts).
 */
export function composeStack(layers: PromptLayer[]): string {
  return [...layers]
    .sort((a, b) => a.order - b.order)
    .filter((layer) => layer.enabled && effectiveContent(layer).trim() !== '')
    .map(effectiveContent)
    .join('\n\n');
}

/**
 * Whether the user has an override that diverges from the shipped default — drives
 * the "● modified" dot instantly (locally), matching the backend `modified` rule.
 */
export function isModified(layer: Pick<PromptLayer, 'override' | 'default'>): boolean {
  return layer.override !== null && layer.override !== undefined && layer.override !== (layer.default ?? '');
}

/**
 * Rough token estimate for the live counter (~chars/4). Always surfaced with a
 * "~" so it reads as approximate, not authoritative.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

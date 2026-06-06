/**
 * Prompt placeholders — the whitelisted `{token}`s the backend substitutes when a
 * system prompt is composed (mirror of `api/agentx_ai/prompts/placeholders.py`;
 * keep in sync). Used by the editor's "Insert placeholder" affordance and by the
 * preview highlighter.
 */

export interface PromptPlaceholder {
  token: string;
  label: string;
  description: string;
}

export const PROMPT_PLACEHOLDERS: PromptPlaceholder[] = [
  { token: '{agent_name}', label: 'Agent name', description: "This agent's display name" },
  { token: '{date}', label: 'Date', description: "Today's date (YYYY-MM-DD)" },
  { token: '{time}', label: 'Time', description: 'Current time (HH:MM, 24h)' },
];

const TOKENS = new Set(PROMPT_PLACEHOLDERS.map((p) => p.token));

/**
 * Split text into segments, flagging the whitelisted placeholder tokens so a
 * renderer can highlight them. Unknown `{...}` stays plain text.
 */
export function splitPlaceholders(text: string): { text: string; placeholder: boolean }[] {
  if (!text || !text.includes('{')) return [{ text, placeholder: false }];
  // Split on any {...} run, keeping the delimiters.
  const parts = text.split(/(\{[a-z_]+\})/gi);
  return parts
    .filter((p) => p !== '')
    .map((p) => ({ text: p, placeholder: TOKENS.has(p) }));
}

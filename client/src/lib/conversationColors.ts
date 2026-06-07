/**
 * A small, theme-safe palette for per-conversation color tags. Hand-picked hues
 * that read on Cosmic / Light / Professional. (Distinct from `agentAccent`, which
 * deterministically generates a color per agent — this is a user-chosen tag.)
 */

export interface ConversationColor {
  key: string;
  label: string;
  value: string; // CSS color
}

export const CONVERSATION_COLORS: ConversationColor[] = [
  { key: 'slate', label: 'Slate', value: '#64748b' },
  { key: 'red', label: 'Red', value: '#ef4444' },
  { key: 'orange', label: 'Orange', value: '#f59e0b' },
  { key: 'green', label: 'Green', value: '#22c55e' },
  { key: 'teal', label: 'Teal', value: '#14b8a6' },
  { key: 'blue', label: 'Blue', value: '#3b82f6' },
  { key: 'violet', label: 'Violet', value: '#8b5cf6' },
  { key: 'pink', label: 'Pink', value: '#ec4899' },
];

export function conversationColorValue(key: string | undefined): string | undefined {
  if (!key) return undefined;
  return CONVERSATION_COLORS.find(c => c.key === key)?.value;
}

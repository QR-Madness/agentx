/**
 * Friendly, present-tense labels for the read-only tools the ambassador calls
 * while answering — shown as live chips ("reading the conversation…") so you can
 * see it reading/surveying rather than guessing. Mirrors the backend tool belt
 * (`agent/ambassador_tools.py`).
 */

export function toolChipLabel(tool: string, args?: Record<string, unknown>): string {
  const topic = typeof args?.topic === 'string' ? args.topic.trim() : '';
  switch (tool) {
    case 'summarize_conversation':
      return 'reading the conversation';
    case 'explore_conversation':
      return topic ? `digging into "${topic}"` : 'digging deeper';
    case 'read_conversation':
      return 'reading a conversation';
    case 'list_conversations':
      return 'looking across your conversations';
    default:
      return tool.replace(/_/g, ' ');
  }
}

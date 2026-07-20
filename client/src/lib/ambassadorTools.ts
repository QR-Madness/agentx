/**
 * Friendly, present-tense labels for the tools the ambassador calls while
 * answering — shown as live chips ("reading the conversation…") so you can
 * see it reading/surveying rather than guessing. Mirrors the backend tool belt
 * (`agent/ambassador_tools.py`).
 *
 * Also home to the confirmed-write proposal *resolution* store: which
 * rename/archive/delete proposals the person already confirmed or dismissed,
 * kept client-side (localStorage) so a thread replay never resurrects a live
 * confirm strip for an already-actioned proposal.
 */

export function toolChipLabel(tool: string, args?: Record<string, unknown>): string {
  const topic = typeof args?.topic === 'string' ? args.topic.trim() : '';
  const query = typeof args?.query === 'string' ? args.query.trim() : '';
  switch (tool) {
    case 'summarize_conversation':
      return 'reading the conversation';
    case 'explore_conversation':
      return topic ? `digging into "${topic}"` : 'digging deeper';
    case 'read_conversation':
      return 'reading a conversation';
    case 'list_conversations':
      return 'looking across your conversations';
    case 'survey_conversations':
      return 'surveying your conversations';
    case 'list_agents':
      return 'checking the agent roster';
    case 'rename_inquiry':
      return 'titling this Inquiry';
    case 'read_conversation_state':
      return "checking the conversation's state";
    case 'read_conversation_results':
      return 'checking what it produced';
    case 'search_conversations':
      return query ? `searching for "${query}"` : 'searching your conversations';
    case 'list_active_runs':
      return 'checking what your agents are doing';
    case 'usage_report':
      return 'tallying usage and spend';
    case 'recall_memory':
      return query ? `recalling "${query}"` : 'recalling memory';
    // Confirmed writes — proposals, sharply distinct from the Inquiry rename:
    case 'rename_conversation':
      return 'proposing a new conversation title';
    case 'archive_conversation':
      return args?.unarchive ? 'proposing a restore' : 'proposing an archive';
    case 'delete_conversation':
      return 'proposing a deletion';
    case 'dispatch_task':
      return 'proposing a dispatch';
    default:
      return tool.replace(/_/g, ' ');
  }
}

/** Human sentence for a confirmed-write proposal (panel strip + voice strip). */
export function proposalSentence(p: {
  action: 'rename' | 'archive' | 'unarchive' | 'delete' | 'dispatch';
  title?: string;
  agent_name?: string;
  task?: string;
  conversation_id?: string;
}): string {
  switch (p.action) {
    case 'rename':
      return `Rename this conversation to "${p.title ?? ''}"?`;
    case 'archive':
      return 'Archive this conversation? It leaves recents and surveys, but is never deleted.';
    case 'unarchive':
      return 'Restore this conversation from the archive?';
    case 'delete':
      return 'Permanently delete this conversation and its stored turns?';
    case 'dispatch': {
      const task = (p.task ?? '').trim();
      const clipped = task.length > 120 ? task.slice(0, 120).trimEnd() + '…' : task;
      const where = p.conversation_id ? ' (in an existing conversation)' : '';
      return `Dispatch to ${p.agent_name ?? 'a worker'}${where}: "${clipped}"?`;
    }
  }
}

// --- Proposal resolution (client-side, per proposal_id) ----------------------

export type ProposalResolution = 'confirmed' | 'dismissed';

const RESOLUTION_KEY = 'agentx:ambassador-proposal-resolutions';
const RESOLUTION_CAP = 200;

function readResolutions(): Record<string, ProposalResolution> {
  try {
    const raw = localStorage.getItem(RESOLUTION_KEY);
    return raw ? (JSON.parse(raw) as Record<string, ProposalResolution>) : {};
  } catch {
    return {};
  }
}

export function getProposalResolution(proposalId: string): ProposalResolution | undefined {
  return readResolutions()[proposalId];
}

export function setProposalResolution(proposalId: string, resolution: ProposalResolution): void {
  try {
    const map = readResolutions();
    map[proposalId] = resolution;
    // Cap: drop oldest insertions (object key order) so the map stays lean.
    const keys = Object.keys(map);
    if (keys.length > RESOLUTION_CAP) {
      for (const k of keys.slice(0, keys.length - RESOLUTION_CAP)) delete map[k];
    }
    localStorage.setItem(RESOLUTION_KEY, JSON.stringify(map));
  } catch {
    /* best-effort */
  }
}

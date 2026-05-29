import { User, Edit3 } from 'lucide-react';
import type { BubbleProps } from './types';
import { useAgentProfileOptional } from '../../../contexts/AgentProfileContext';
import { resolveMentionToken } from '../../../lib/mentions';
import type { AgentProfile } from '../../../lib/api/types';

const MENTION_RE = /(^|[^\w/@])@([\w-]+)/g;

/** Render content with resolvable @mentions emphasized as @DisplayName. */
function renderWithMentions(content: string, profiles: AgentProfile[]) {
  if (!content.includes('@') || profiles.length === 0) return content;

  const nodes: (string | React.ReactNode)[] = [];
  let last = 0;
  let key = 0;
  for (const m of content.matchAll(MENTION_RE)) {
    const token = m[2];
    const agentId = resolveMentionToken(token, profiles);
    if (!agentId) continue;
    const lead = m[1]; // the boundary char captured before '@' (may be '')
    const at = (m.index ?? 0) + lead.length;
    const profile = profiles.find(p => p.agentId === agentId)!;
    if (at > last) nodes.push(content.slice(last, at));
    nodes.push(
      <span key={`m${key++}`} className="mention">@{profile.name}</span>,
    );
    last = at + 1 + token.length; // past '@' + token
  }
  if (nodes.length === 0) return content;
  if (last < content.length) nodes.push(content.slice(last));
  return nodes;
}

export function UserBubble({ message, onEdit }: BubbleProps<'user'>) {
  const profiles = useAgentProfileOptional()?.profiles ?? [];
  return (
    <div className="message-bubble user">
      <div className="message-avatar user-avatar">
        <User size={16} />
      </div>
      <div className="message-body">
        <div className="user-header">
          <span className="user-name">You</span>
        </div>
        <div className="message-text">{renderWithMentions(message.content, profiles)}</div>
        <div className="message-meta">
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {message.editedAt && <span className="message-edited">(edited)</span>}
        </div>
        {onEdit && (
          <button
            className="edit-button"
            onClick={() => onEdit(message.content)}
            title="Edit message"
          >
            <Edit3 size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

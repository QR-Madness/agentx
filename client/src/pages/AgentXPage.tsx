/**
 * AgentXPage — Main chat workspace: a left Conversations rail + the chat panel.
 */

import { ChatPanel } from '../components/chat';
import { ConversationSidebar } from '../components/chat/ConversationSidebar';
import './AgentXPage.css';

export function AgentXPage() {
  return (
    <div className="agentx-layout">
      <ConversationSidebar />
      <div className="agentx-chat">
        <ChatPanel />
      </div>
    </div>
  );
}

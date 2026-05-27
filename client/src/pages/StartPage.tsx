/**
 * StartPage — Welcome screen with agent greeting and quick actions
 */

import { MessageSquarePlus, LayoutDashboard } from 'lucide-react';
import { useAgentProfile } from '../contexts/AgentProfileContext';
import { getAvatarIcon } from '../lib/avatars';
import type { PageId } from '../layouts/TopBar';
import './StartPage.css';
import { Button } from '../components/ui';

interface StartPageProps {
  onNavigate: (page: PageId) => void;
}

export function StartPage({ onNavigate }: StartPageProps) {
  const { activeProfile, getAgentName } = useAgentProfile();
  const agentName = getAgentName();
  const AvatarIcon = getAvatarIcon(activeProfile?.avatar);

  const handleNewConversation = () => {
    onNavigate('agentx');
    // Could also trigger new conversation creation here if needed
  };

  const handleOpenDashboard = () => {
    onNavigate('dashboard');
  };

  return (
    <div className="start-page">
      <div className="start-hero">
        <div className="start-logo">
          <AvatarIcon size={48} />
        </div>
        <h1 className="start-title">Hello, I'm {agentName}</h1>
        <p className="start-subtitle">
          How can I assist you today?
        </p>
        <div className="start-actions">
          <Button
            variant="primary" className="start-cta"
            onClick={handleNewConversation}
          >
            <MessageSquarePlus size={18} />
            <span>New Conversation</span>
          </Button>
          <Button
            variant="secondary" className="start-cta-secondary"
            onClick={handleOpenDashboard}
          >
            <LayoutDashboard size={18} />
            <span>Open Dashboard</span>
          </Button>
        </div>
      </div>
    </div>
  );
}

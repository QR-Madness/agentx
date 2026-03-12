/**
 * StartPage — Welcome placeholder (will be expanded in later phases)
 */

import { Sparkles, MessageSquarePlus } from 'lucide-react';
import './StartPage.css';

export function StartPage() {
  return (
    <div className="start-page">
      <div className="start-hero">
        <div className="start-logo">
          <Sparkles size={48} />
        </div>
        <h1 className="start-title">Welcome to AgentX</h1>
        <p className="start-subtitle">
          Your AI-powered assistant for intelligent conversations
        </p>
        <div className="start-actions">
          <button className="start-cta button-primary">
            <MessageSquarePlus size={18} />
            <span>Start a Conversation</span>
          </button>
        </div>
      </div>
    </div>
  );
}

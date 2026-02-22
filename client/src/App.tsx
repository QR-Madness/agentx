import { useState } from 'react';
import { TabBar, Tab, TabId, TabIcons } from './components/TabBar';
import { ServerProvider } from './contexts/ServerContext';
import { DashboardTab } from './components/tabs/DashboardTab';
import { AgentTab } from './components/tabs/AgentTab';
import { TranslationTab } from './components/tabs/TranslationTab';
import { ChatTab } from './components/tabs/ChatTab';
import { ToolsTab } from './components/tabs/ToolsTab';
import { MemoryTab } from './components/tabs/MemoryTab';
import { SettingsTab } from './components/tabs/SettingsTab';
import './App.css';

function AppContent() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');

  const tabs: Tab[] = [
    { id: 'dashboard', label: 'Dashboard', icon: TabIcons.dashboard },
    { id: 'agent', label: 'Agent', icon: TabIcons.agent },
    { id: 'translation', label: 'Translation', icon: TabIcons.translation },
    { id: 'chat', label: 'Chat', icon: TabIcons.chat },
    { id: 'tools', label: 'Tools', icon: TabIcons.tools },
    { id: 'memory', label: 'Memory', icon: TabIcons.memory },
    { id: 'settings', label: 'Settings', icon: TabIcons.settings },
  ];

  return (
    <div className="app-container">
      <TabBar tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />

      <main style={{ flex: 1, overflow: 'hidden', position: 'relative', zIndex: 1 }}>
        {/* All tabs are always mounted to preserve state, but only one is visible */}
        <div style={{ display: activeTab === 'dashboard' ? 'block' : 'none', height: '100%' }}>
          <DashboardTab />
        </div>
        <div style={{ display: activeTab === 'agent' ? 'block' : 'none', height: '100%' }}>
          <AgentTab />
        </div>
        <div style={{ display: activeTab === 'translation' ? 'block' : 'none', height: '100%' }}>
          <TranslationTab />
        </div>
        <div style={{ display: activeTab === 'chat' ? 'block' : 'none', height: '100%' }}>
          <ChatTab />
        </div>
        <div style={{ display: activeTab === 'tools' ? 'block' : 'none', height: '100%' }}>
          <ToolsTab />
        </div>
        <div style={{ display: activeTab === 'memory' ? 'block' : 'none', height: '100%' }}>
          <MemoryTab />
        </div>
        <div style={{ display: activeTab === 'settings' ? 'block' : 'none', height: '100%' }}>
          <SettingsTab />
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <ServerProvider>
      <AppContent />
    </ServerProvider>
  );
}

export default App;

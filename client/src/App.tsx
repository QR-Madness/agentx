import { useState } from 'react';
import { TabBar, Tab, TabId } from './components/TabBar';
import { DashboardTab } from './components/tabs/DashboardTab';
import { TranslationTab } from './components/tabs/TranslationTab';
import { ChatTab } from './components/tabs/ChatTab';
import { ToolsTab } from './components/tabs/ToolsTab';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');

  const tabs: Tab[] = [
    { id: 'dashboard', label: 'Dashboard', icon: '🏠' },
    { id: 'translation', label: 'Translation', icon: '🌐' },
    { id: 'chat', label: 'Chat', icon: '💬' },
    { id: 'tools', label: 'Tools', icon: '🔧' },
  ];

  return (
    <div className="app-container">
      <TabBar tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />

      <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {/* All tabs are always mounted to preserve state, but only one is visible */}
        <div style={{ display: activeTab === 'dashboard' ? 'block' : 'none', height: '100%' }}>
          <DashboardTab />
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
      </main>
    </div>
  );
}

export default App;

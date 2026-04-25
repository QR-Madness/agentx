/**
 * MemorySettingsSection — Memory system configuration (consolidation, recall)
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Settings } from 'lucide-react';

export default function MemorySettingsSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Settings size={20} className="section-title-icon" />
            Memory Settings
          </h2>
          <p className="section-description">
            Configure consolidation and recall layer settings
          </p>
        </div>
      </div>

      <div className="card">
        <p>Memory settings coming soon. Access via Memory panel in the meantime.</p>
      </div>
    </div>
  );
}

/**
 * ModelsSection — Model context limits configuration
 * TODO: Full extraction from SettingsPanel.tsx lines 550-653
 */

import { Layers } from 'lucide-react';

export default function ModelsSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Layers size={20} className="section-title-icon" />
            Model Context Limits
          </h2>
          <p className="section-description">
            Configure context window and output token limits for local models
          </p>
        </div>
      </div>

      <div className="card">
        <p>Model limits configuration coming soon. Access via legacy settings panel in the meantime.</p>
      </div>
    </div>
  );
}

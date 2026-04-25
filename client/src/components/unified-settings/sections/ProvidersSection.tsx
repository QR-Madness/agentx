/**
 * ProvidersSection — Model provider configuration (API keys, URLs)
 * TODO: Full extraction from SettingsPanel.tsx lines 391-548
 */

import { Key } from 'lucide-react';

export default function ProvidersSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Key size={20} className="section-title-icon" />
            Model Providers
          </h2>
          <p className="section-description">
            Configure API keys and URLs for AI model providers
          </p>
        </div>
      </div>

      <div className="card">
        <p>Provider configuration coming soon. Access via legacy settings panel in the meantime.</p>
      </div>
    </div>
  );
}

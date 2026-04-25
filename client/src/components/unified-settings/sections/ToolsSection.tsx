/**
 * ToolsSection — MCP tools browser
 * TODO: Full extraction from ToolsPanel.tsx
 */

import { Wrench } from 'lucide-react';

export default function ToolsSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Wrench size={20} className="section-title-icon" />
            MCP Tools
          </h2>
          <p className="section-description">
            Browse available MCP tools and capabilities
          </p>
        </div>
      </div>

      <div className="card">
        <p>Tools browser coming soon. Access via Tools panel in the meantime.</p>
      </div>
    </div>
  );
}

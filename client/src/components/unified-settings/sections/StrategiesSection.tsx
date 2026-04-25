/**
 * StrategiesSection — Procedural strategies browser
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Zap } from 'lucide-react';

export default function StrategiesSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Zap size={20} className="section-title-icon" />
            Strategies
          </h2>
          <p className="section-description">
            Browse procedural strategies learned from tool usage
          </p>
        </div>
      </div>

      <div className="card">
        <p>Strategies browser coming soon. Access via Memory panel in the meantime.</p>
      </div>
    </div>
  );
}

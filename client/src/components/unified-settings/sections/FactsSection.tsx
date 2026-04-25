/**
 * FactsSection — Facts browser
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { FileText } from 'lucide-react';

export default function FactsSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <FileText size={20} className="section-title-icon" />
            Facts
          </h2>
          <p className="section-description">
            Browse factual knowledge stored in semantic memory
          </p>
        </div>
      </div>

      <div className="card">
        <p>Facts browser coming soon. Access via Memory panel in the meantime.</p>
      </div>
    </div>
  );
}

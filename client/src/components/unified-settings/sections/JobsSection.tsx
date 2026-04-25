/**
 * JobsSection — Background jobs monitor
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Clock } from 'lucide-react';

export default function JobsSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Clock size={20} className="section-title-icon" />
            Background Jobs
          </h2>
          <p className="section-description">
            Monitor consolidation and maintenance jobs
          </p>
        </div>
      </div>

      <div className="card">
        <p>Jobs monitor coming soon. Access via Memory panel in the meantime.</p>
      </div>
    </div>
  );
}

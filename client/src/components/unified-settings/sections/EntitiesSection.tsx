/**
 * EntitiesSection — Entity explorer
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Users } from 'lucide-react';

export default function EntitiesSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Users size={20} className="section-title-icon" />
            Entities
          </h2>
          <p className="section-description">
            Browse entities in semantic memory (people, organizations, concepts)
          </p>
        </div>
      </div>

      <div className="card">
        <p>Entity explorer coming soon. Access via Memory panel in the meantime.</p>
      </div>
    </div>
  );
}

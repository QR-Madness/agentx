/**
 * MemoryOverviewSection — Memory & storage overview
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Database } from 'lucide-react';
import { Card, SectionHeader } from '../../ui';

export default function MemoryOverviewSection() {
  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Database size={20} />}
        title="Memory & Storage"
        description="Configure agent memory and data retention"
      />

      <Card className="memory-info">
        <div className="info-row">
          <span className="info-label">Session Storage</span>
          <span className="info-value">Local (Browser)</span>
        </div>
        <div className="info-row">
          <span className="info-label">Server Data</span>
          <span className="info-value">PostgreSQL + Neo4j</span>
        </div>
        <div className="info-row">
          <span className="info-label">Cache</span>
          <span className="info-value">Redis</span>
        </div>
      </Card>
    </div>
  );
}

/**
 * MemoryOverviewSection — Memory & storage overview
 * TODO: Full extraction from MemoryPanel.tsx
 */

import { Database } from 'lucide-react';

export default function MemoryOverviewSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Database size={20} className="section-title-icon" />
            Memory & Storage
          </h2>
          <p className="section-description">
            Configure agent memory and data retention
          </p>
        </div>
      </div>

      <div className="memory-info card">
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
      </div>
    </div>
  );
}

/**
 * TranslationSection — Translation interface
 * TODO: Full extraction from TranslationPanel.tsx
 */

import { Languages } from 'lucide-react';

export default function TranslationSection() {
  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Languages size={20} className="section-title-icon" />
            Translation
          </h2>
          <p className="section-description">
            Translate text between 200+ languages using NLLB-200
          </p>
        </div>
      </div>

      <div className="card">
        <p>Translation interface coming soon. Access via Translation modal in the meantime.</p>
      </div>
    </div>
  );
}

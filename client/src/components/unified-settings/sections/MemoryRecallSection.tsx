/**
 * MemoryRecallSection — recall/retrieval settings (memory nav group).
 * Thin wrapper: the form lives in RecallSettingsPanel.
 */

import { RecallSettingsPanel } from '../../memory-settings';
import '../../../styles/MemoryPanel.css';

export default function MemoryRecallSection() {
  return (
    <div className="settings-section fade-in">
      <RecallSettingsPanel />
    </div>
  );
}

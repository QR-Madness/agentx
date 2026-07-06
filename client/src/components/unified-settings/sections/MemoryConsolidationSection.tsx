/**
 * MemoryConsolidationSection — consolidation pipeline settings (memory nav
 * group). Thin wrapper: the form lives in ConsolidationSettingsPanel.
 */

import { useConsolidate } from '../../../lib/hooks';
import { ConsolidationSettingsPanel } from '../../memory-settings';
import '../../../styles/MemoryPanel.css';

export default function MemoryConsolidationSection() {
  const { consolidate } = useConsolidate();

  return (
    <div className="settings-section fade-in">
      <ConsolidationSettingsPanel onConsolidate={async jobs => { await consolidate(jobs); }} />
    </div>
  );
}

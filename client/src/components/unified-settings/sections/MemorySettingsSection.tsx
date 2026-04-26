import { useConsolidate } from '../../../lib/hooks';
import { ConsolidationSettingsPanel, RecallSettingsPanel } from '../../panels/MemorySettingsPanels';
import '../../../styles/MemoryPanel.css';

export default function MemorySettingsSection() {
  const { consolidate } = useConsolidate();

  const handleConsolidate = async (jobs?: string[]) => {
    await consolidate(jobs);
  };

  return (
    <div className="settings-section fade-in">
      <RecallSettingsPanel />
      <ConsolidationSettingsPanel onConsolidate={handleConsolidate} />
    </div>
  );
}

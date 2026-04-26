import { ArrowLeft } from 'lucide-react';
import { PromptLibraryModal } from '../modals/PromptLibraryModal';

interface PromptLibraryPanelProps {
  mode: 'insert' | 'select';
  onBack: () => void;
  onInsert?: (content: string) => void;
  onSelectTemplate?: (templateId: string, content: string) => void;
}

export function PromptLibraryPanel({
  mode,
  onBack,
  onInsert,
  onSelectTemplate,
}: PromptLibraryPanelProps) {
  const handleInsert = (content: string) => {
    onInsert?.(content);
    onBack();
  };

  const handleSelect = (templateId: string, content: string) => {
    onSelectTemplate?.(templateId, content);
    onBack();
  };

  return (
    <div className="prompt-library-panel">
      <div className="library-panel-breadcrumb">
        <button className="library-panel-back-btn" onClick={onBack}>
          <ArrowLeft size={15} />
          <span>Back to Profile</span>
        </button>
      </div>
      <div className="library-panel-content">
        <PromptLibraryModal
          onClose={onBack}
          onInsert={mode === 'insert' ? handleInsert : undefined}
          onSelectTemplate={mode === 'select' ? handleSelect : undefined}
          mode={mode}
          variant="panel"
        />
      </div>
    </div>
  );
}

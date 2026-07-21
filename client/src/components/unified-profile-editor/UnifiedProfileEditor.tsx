import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useIsMobile } from '../../lib/hooks';
import { ParallaxBackground } from '../unified-settings/animations/ParallaxBackground';
import {
  backdropVariants,
  containerVariants,
} from '../unified-settings/animations/transitions';
import { ProfileNav } from './ProfileNav';
import { ProfileContent } from './ProfileContent';
import './UnifiedProfileEditor.css';

interface UnifiedProfileEditorProps {
  isOpen: boolean;
  onClose: () => void;
  initialProfileId?: string;
  isNew?: boolean;
}

export function UnifiedProfileEditor({
  isOpen,
  onClose,
  initialProfileId,
  isNew: startNew = false,
}: UnifiedProfileEditorProps) {
  const { profiles, refresh } = useAgentProfile();
  const isMobile = useIsMobile();

  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(
    initialProfileId ?? null
  );
  const [isCreatingNew, setIsCreatingNew] = useState(startNew);

  // Refresh the list whenever the editor opens — the profile cache is loaded
  // once per server, so a freshly-seeded/added profile would otherwise not
  // appear until something triggered a refetch (e.g. an edit).
  useEffect(() => {
    if (isOpen) void refresh();
  }, [isOpen, refresh]);

  // Default to the first profile if nothing is selected and we're not creating.
  // On mobile the list is shown first (master-detail), so don't auto-open one.
  useEffect(() => {
    if (!isMobile && !isCreatingNew && !selectedProfileId && profiles.length > 0) {
      setSelectedProfileId(profiles[0].id);
    }
  }, [profiles, selectedProfileId, isCreatingNew, isMobile]);

  // ESC key
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  // Body scroll lock
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const selectedProfile =
    !isCreatingNew && selectedProfileId
      ? profiles.find(p => p.id === selectedProfileId) ?? null
      : null;

  const handleSelectProfile = (id: string) => {
    setSelectedProfileId(id);
    setIsCreatingNew(false);
  };

  const handleCreateNew = () => {
    setIsCreatingNew(true);
    setSelectedProfileId(null);
  };

  const handleSaved = (savedId: string) => {
    // After save/create, select the saved profile in the nav
    setSelectedProfileId(savedId);
    setIsCreatingNew(false);
  };

  const handleDeleted = () => {
    // After deletion, fall back to the first remaining profile
    setIsCreatingNew(false);
    setSelectedProfileId(null); // useEffect above will pick the first
  };

  // Mobile master-detail: show the list OR the editor, never both. Selecting a
  // profile / creating one opens the editor; Back returns to the list.
  const handleBack = () => {
    setIsCreatingNew(false);
    setSelectedProfileId(null);
  };
  // The Prompt Library takes over the content area; collapse the nav so it gets
  // the full width (and so the profile list isn't competing with the browser).
  const [libraryOpen, setLibraryOpen] = useState(false);
  const showList = (!isMobile || (!isCreatingNew && !selectedProfileId)) && !libraryOpen;
  const showEditor = !isMobile || isCreatingNew || !!selectedProfileId;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="unified-profile-backdrop"
            variants={backdropVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
            onClick={onClose}
          />

          <motion.div
            className="unified-profile-container"
            variants={containerVariants}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <ParallaxBackground />

            <div className="unified-profile-header">
              <h1>Agent Profiles</h1>
              <button onClick={onClose} className="close-button" title="Close">
                <X size={20} />
              </button>
            </div>

            <div className={`unified-profile-layout${isMobile ? ' is-mobile' : ''}${libraryOpen ? ' is-library' : ''}`}>
              {showList && (
                <ProfileNav
                  selectedProfileId={selectedProfileId}
                  isCreatingNew={isCreatingNew}
                  onSelectProfile={handleSelectProfile}
                  onCreateNew={handleCreateNew}
                />
              )}
              {showEditor && (
                <ProfileContent
                  key={isCreatingNew ? '__new__' : (selectedProfileId ?? '__empty__')}
                  profile={selectedProfile}
                  onSaved={handleSaved}
                  onDeleted={handleDeleted}
                  onCancel={onClose}
                  onBack={isMobile ? handleBack : undefined}
                  onLibraryOpenChange={setLibraryOpen}
                  onSelectProfile={handleSelectProfile}
                />
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

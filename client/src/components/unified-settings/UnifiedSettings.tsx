/**
 * UnifiedSettings — Full-screen immersive settings interface
 *
 * Replaces fragmented drawer-based settings with a cohesive experience:
 * - Vertical sidebar navigation, grouped by category
 * - Enhanced glassmorphism with depth layers
 * - Smooth Framer Motion animations
 * - Parallax background effects
 *
 * Opens on Overview, which answers "where do I start" — what you've changed
 * from the defaults, and where everything lives. It used to open on Model
 * Providers, so the first thing anyone saw was an API-key admin page.
 */

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Menu } from 'lucide-react';
import { useSettingsNavigation } from './hooks/useSettingsNavigation';
import { SettingsNav } from './SettingsNav';
import { SettingsContent } from './SettingsContent';
import { SettingsManifestProvider } from './SettingsManifestContext';
import { SettingsSearchProvider } from './SettingsSearchContext';
import { ParallaxBackground } from './animations/ParallaxBackground';
import { backdropVariants, containerVariants } from './animations/transitions';
import './UnifiedSettings.css';

interface UnifiedSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  /** Open straight to a section (palette commands, deep links). */
  initialSection?: string;
  /** `store:key` of a setting to scroll to and flash once it renders. */
  focusSetting?: string;
}

export function UnifiedSettings({
  isOpen, onClose, initialSection, focusSetting,
}: UnifiedSettingsProps) {
  const { activeSection, navigateTo, focusSettingIn, focus } =
    useSettingsNavigation(initialSection, focusSetting);
  const [isNavOpen, setIsNavOpen] = useState(false);

  // ESC key handler
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

  // Prevent body scroll when open
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

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop with progressive blur */}
          <motion.div
            className="unified-settings-backdrop"
            variants={backdropVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
            onClick={onClose}
          />

          {/* Main container - spring physics */}
          <motion.div
            className="unified-settings-container"
            variants={containerVariants}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <ParallaxBackground />

            {/* Header with close button */}
            <div className="unified-settings-header">
              <div className="header-left">
                <button
                  className="nav-toggle-btn"
                  onClick={() => setIsNavOpen(true)}
                  title="Open navigation"
                >
                  <Menu size={20} />
                </button>
                <h1>Settings</h1>
              </div>
              <button onClick={onClose} className="close-button" title="Close settings">
                <X size={20} />
              </button>
            </div>

            {/* Mobile scrim — closes nav when tapped */}
            {isNavOpen && (
              <div
                className="nav-mobile-scrim"
                onClick={() => setIsNavOpen(false)}
              />
            )}

            {/* Two-column layout. One manifest fetch for the whole surface —
                the nav searches it, the sections render from it, and both work
                fine if it never arrives. */}
            <SettingsManifestProvider>
              <SettingsSearchProvider>
                <div className="unified-settings-layout">
                  <SettingsNav
                    activeSection={activeSection}
                    isOpen={isNavOpen}
                    onSectionChange={(id) => {
                      navigateTo(id);
                      setIsNavOpen(false);
                    }}
                    onSettingSelect={(sectionId, settingId) => {
                      focusSettingIn(sectionId, settingId);
                      setIsNavOpen(false);
                    }}
                    onClose={() => setIsNavOpen(false)}
                  />
                  <SettingsContent
                    activeSection={activeSection}
                    onNavigate={navigateTo}
                    onFocusSetting={focusSettingIn}
                    focus={focus}
                  />
                </div>
              </SettingsSearchProvider>
            </SettingsManifestProvider>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

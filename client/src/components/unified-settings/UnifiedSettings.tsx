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
import { ParallaxBackground } from './animations/ParallaxBackground';
import { backdropVariants, containerVariants } from './animations/transitions';
import './UnifiedSettings.css';

interface UnifiedSettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export function UnifiedSettings({ isOpen, onClose }: UnifiedSettingsProps) {
  const { activeSection, navigateTo } = useSettingsNavigation();
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

            {/* Two-column layout */}
            <div className="unified-settings-layout">
              <SettingsNav
                activeSection={activeSection}
                isOpen={isNavOpen}
                onSectionChange={(id) => {
                  navigateTo(id);
                  setIsNavOpen(false);
                }}
                onClose={() => setIsNavOpen(false)}
              />
              {/* One manifest fetch for the whole surface; sections read what
                  they need from it and work fine if it never arrives. */}
              <SettingsManifestProvider>
                <SettingsContent
                  activeSection={activeSection}
                  onNavigate={navigateTo}
                />
              </SettingsManifestProvider>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

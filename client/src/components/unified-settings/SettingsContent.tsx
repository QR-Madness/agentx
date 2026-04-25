/**
 * SettingsContent — Content area with section routing
 */

import { Suspense, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw } from 'lucide-react';
import { findSectionById } from './sections';
import { contentVariants } from './animations/transitions';

interface SettingsContentProps {
  activeSection: string;
}

function LoadingSpinner() {
  return (
    <div className="settings-loading">
      <RefreshCw size={32} className="spin" />
      <p>Loading section...</p>
    </div>
  );
}

export function SettingsContent({ activeSection }: SettingsContentProps) {
  const section = useMemo(() => findSectionById(activeSection), [activeSection]);

  if (!section) {
    return (
      <div className="settings-content-area">
        <div className="settings-error">
          <p>Section not found: {activeSection}</p>
        </div>
      </div>
    );
  }

  const Component = section.component;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        className="settings-content-area"
        key={activeSection}
        variants={contentVariants}
        initial="initial"
        animate="animate"
        exit="exit"
      >
        <Suspense fallback={<LoadingSpinner />}>
          <Component />
        </Suspense>
      </motion.div>
    </AnimatePresence>
  );
}

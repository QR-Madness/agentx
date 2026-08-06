/**
 * SettingsContent — Content area with section routing
 *
 * Each section renders inside its own error boundary, keyed by section id. A
 * section that throws during render used to escape all the way to the page-level
 * boundary and take the whole app with it — which is exactly what happened when
 * Web Search hit a Radix Select it couldn't render. Now the failure is confined
 * to the pane, the rest of settings stays usable, and switching sections clears
 * the caught error (that's what the key is for).
 *
 * Pending edits are safe across that unmount: useSettingsAutosave flushes on
 * unmount, so a debounced change can't be lost when a boundary swaps a section
 * out.
 */

import { Suspense, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw, TriangleAlert } from 'lucide-react';
import { findSectionById } from './sections';
import { contentVariants } from './animations/transitions';
import { ErrorBoundary } from '../ErrorBoundary';
import { Button } from '../ui';

interface SettingsContentProps {
  activeSection: string;
  /** Lets a section hand navigation back to the shell (Overview uses it). */
  onNavigate?: (sectionId: string) => void;
}

function LoadingSpinner() {
  return (
    <div className="settings-loading">
      <RefreshCw size={32} className="spin" />
      <p>Loading section...</p>
    </div>
  );
}

function SectionError({ label, error, reset }: {
  label: string;
  error: Error;
  reset: () => void;
}) {
  return (
    <div className="settings-error" role="alert">
      <span className="settings-error__icon"><TriangleAlert size={28} /></span>
      <h3>{label} couldn't be displayed</h3>
      <p>{error.message || 'This section failed to render.'}</p>
      <p className="settings-error__note">
        Your other settings are unaffected — pick another section from the list,
        or try again.
      </p>
      <Button variant="secondary" onClick={reset}>Try again</Button>
    </div>
  );
}

export function SettingsContent({ activeSection, onNavigate }: SettingsContentProps) {
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
        <ErrorBoundary
          key={activeSection}
          fallback={(error, reset) => (
            <SectionError label={section.label} error={error} reset={reset} />
          )}
        >
          <Suspense fallback={<LoadingSpinner />}>
            <Component onNavigate={onNavigate} />
          </Suspense>
        </ErrorBoundary>
      </motion.div>
    </AnimatePresence>
  );
}

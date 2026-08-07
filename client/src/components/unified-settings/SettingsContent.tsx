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

import { Suspense, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw, TriangleAlert } from 'lucide-react';
import { findSectionById } from './sections';
import { contentVariants } from './animations/transitions';
import { ErrorBoundary } from '../ErrorBoundary';
import { Button } from '../ui';
import { scrollToAnchor } from '../../lib/scrollToAnchor';
import type { SettingFocus } from './hooks/useSettingsNavigation';

interface SettingsContentProps {
  activeSection: string;
  /** Lets a section hand navigation back to the shell (Overview uses it). */
  onNavigate?: (sectionId: string) => void;
  /** Same, but landing on a specific control. */
  onFocusSetting?: (sectionId: string, settingId: string) => void;
  /** Setting to scroll to once this section has rendered (see SettingFocus). */
  focus?: SettingFocus;
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

export function SettingsContent({
  activeSection, onNavigate, onFocusSetting, focus,
}: SettingsContentProps) {
  const section = useMemo(() => findSectionById(activeSection), [activeSection]);
  const areaRef = useRef<HTMLDivElement>(null);

  // Deliver a deep-linked focus once the section's lazy chunk has rendered.
  // scrollToAnchor retries across a few frames, which covers the Suspense gap
  // and the autosave hook's first load without needing to observe either.
  //
  // `focus.seq` is in the deps so re-selecting the same setting flashes again;
  // nothing clears the focus here, because a state change during delivery would
  // re-run this effect and its cleanup would cancel the scroll mid-flight.
  const focusKey = focus?.key;
  const focusSeq = focus?.seq;
  useEffect(() => {
    if (!focusKey) return;
    return scrollToAnchor('setting', focusKey, {
      within: areaRef.current,
      flashClass: 'setting-flash',
      retries: 12,
    });
  }, [focusKey, focusSeq, activeSection]);

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
        ref={areaRef}
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
            <Component onNavigate={onNavigate} onFocusSetting={onFocusSetting} />
          </Suspense>
        </ErrorBoundary>
      </motion.div>
    </AnimatePresence>
  );
}

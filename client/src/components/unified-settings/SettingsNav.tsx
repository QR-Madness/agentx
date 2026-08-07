/**
 * SettingsNav — Vertical sidebar navigation for settings sections
 *
 * Searching here matches individual settings as well as sections: a hit takes
 * you to the control itself rather than dropping you at the top of the section
 * that happens to contain it.
 */

import { motion } from 'framer-motion';
import { Search, X, CornerDownRight } from 'lucide-react';
import { SECTION_HIERARCHY, getAllSections } from './sections';
import { useSettingsSearch } from './hooks/useSettingsSearch';
import { useSettingsManifest } from './SettingsManifestContext';
import { useSharedSettingsSearch } from './SettingsSearchContext';
import { navVariants } from './animations/transitions';

interface SettingsNavProps {
  activeSection: string;
  onSectionChange: (sectionId: string) => void;
  /** Navigate to a section and land on one of its settings. */
  onSettingSelect?: (sectionId: string, settingId: string) => void;
  isOpen?: boolean;
  onClose?: () => void;
}

export function SettingsNav({
  activeSection, onSectionChange, onSettingSelect, isOpen, onClose,
}: SettingsNavProps) {
  const allSections = getAllSections();
  const manifest = useSettingsManifest();
  // Prefer the shared search, so this box and the Overview's are one input.
  // The local instance is the fallback for rendering outside the provider —
  // the same null-safety contract the manifest context holds to.
  const local = useSettingsSearch(allSections, manifest?.entries.values());
  const {
    query, setQuery, filtered, settingHits, isSearching, hasResults,
  } = useSharedSettingsSearch() ?? local;
  const sectionLabels = new Map(allSections.map(s => [s.id, s.label]));

  return (
    <motion.nav
      className={`settings-nav${isOpen ? ' is-open' : ''}`}
      variants={navVariants}
      initial="initial"
      animate="animate"
    >
      {/* Mobile close button */}
      {onClose && (
        <button className="nav-mobile-close" onClick={onClose} title="Close navigation">
          <X size={18} />
        </button>
      )}

      {/* Search bar */}
      <div className="nav-search">
        <Search size={16} className="nav-search-icon" />
        <input
          type="text"
          className="nav-search-input"
          placeholder="Search settings..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {/* Category groups */}
      {Object.entries(SECTION_HIERARCHY).map(([key, category]) => {
        const visibleSections = query
          ? category.sections.filter(s => filtered.some(f => f.id === s.id))
          : category.sections;

        if (visibleSections.length === 0) return null;

        return (
          <div key={key} className="nav-category">
            <div className="category-header">
              {category.icon}
              <span>{category.label}</span>
            </div>

            {visibleSections.map(section => (
              <button
                key={section.id}
                className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
                // The active section reads as current, not merely highlighted.
                aria-current={activeSection === section.id ? 'page' : undefined}
                onClick={() => onSectionChange(section.id)}
              >
                {section.icon}
                <span>{section.label}</span>
              </button>
            ))}
          </div>
        );
      })}

      {/* Matching settings — the part section-only search could never do. */}
      {isSearching && settingHits.length > 0 && (
        <div className="nav-category nav-hits">
          <div className="category-header">
            <CornerDownRight size={16} />
            <span>Settings</span>
            <span className="nav-hits-count">{settingHits.length}</span>
          </div>
          {settingHits.map(hit => (
            <button
              key={hit.id}
              className="nav-hit"
              onClick={() => {
                if (hit.sectionId) onSettingSelect?.(hit.sectionId, hit.id);
              }}
              disabled={!hit.sectionId}
              title={hit.summary}
            >
              <span className="nav-hit-label">
                {hit.label}
                {hit.isModified && (
                  <span
                    className="setting-modified-dot"
                    aria-label="Changed from the default"
                  />
                )}
              </span>
              {hit.sectionId && (
                <span className="nav-hit-section">
                  {sectionLabels.get(hit.sectionId) ?? hit.sectionId}
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* No results message */}
      {isSearching && !hasResults && (
        <div className="nav-empty">
          <p>No settings found for "{query}"</p>
        </div>
      )}
    </motion.nav>
  );
}

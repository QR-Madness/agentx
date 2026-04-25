/**
 * SettingsNav — Vertical sidebar navigation for settings sections
 */

import { motion } from 'framer-motion';
import { Search } from 'lucide-react';
import { SECTION_HIERARCHY, getAllSections } from './sections';
import { useSettingsSearch } from './hooks/useSettingsSearch';
import { navVariants } from './animations/transitions';

interface SettingsNavProps {
  activeSection: string;
  onSectionChange: (sectionId: string) => void;
}

export function SettingsNav({ activeSection, onSectionChange }: SettingsNavProps) {
  const allSections = getAllSections();
  const { query, setQuery, filtered } = useSettingsSearch(allSections);

  return (
    <motion.nav
      className="settings-nav"
      variants={navVariants}
      initial="initial"
      animate="animate"
    >
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
                onClick={() => onSectionChange(section.id)}
              >
                {section.icon}
                <span>{section.label}</span>
              </button>
            ))}
          </div>
        );
      })}

      {/* No results message */}
      {query && filtered.length === 0 && (
        <div className="nav-empty">
          <p>No settings found for "{query}"</p>
        </div>
      )}
    </motion.nav>
  );
}

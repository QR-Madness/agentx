/**
 * AppearanceSection — Theme and visual customization.
 *
 * Registry-driven: the theme cards iterate `THEMES` (lib/theme.ts), so adding a
 * theme to the registry surfaces it here (and in the command palette) with no
 * picker edits. Each card's swatch previews the theme's own surface + accent
 * gradient straight from its token values.
 */

import { Palette, Monitor, Check } from 'lucide-react';
import { useTheme } from '../../../contexts/ThemeContext';
import { THEMES } from '../../../lib/theme';
import { THEME_ICONS } from '../../common/themeIcons';
import { Card, SectionHeader } from '../../ui';

export default function AppearanceSection() {
  const { preference, setTheme, currentTheme } = useTheme();

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Palette size={20} />}
        title="Appearance"
        description="Customize the look and feel of AgentX"
      />

      <Card>
        <h3 className="subsection-title">Theme</h3>
        <p className="subsection-description">
          Choose your preferred color scheme
        </p>

        <div className="theme-options">
          {Object.values(THEMES).map((t) => {
            const Icon = THEME_ICONS[t.icon];
            return (
              <button
                key={t.name}
                className={`theme-option ${preference === t.name ? 'active' : ''}`}
                onClick={() => setTheme(t.name as never)}
              >
                <div
                  className="theme-option-icon"
                  style={{
                    background: t.variables['--surface-raised'],
                    color: t.variables['--accent-primary'],
                    borderColor: t.variables['--border-emphasis'],
                  }}
                >
                  <span
                    className="theme-option-swatch"
                    style={{ background: t.variables['--accent-gradient'] }}
                  />
                  <Icon size={20} />
                </div>
                <div className="theme-option-info">
                  <span className="theme-option-name">{t.displayName}</span>
                  <span className="theme-option-desc">{t.description}</span>
                </div>
                {preference === t.name && <Check size={18} className="theme-check" />}
              </button>
            );
          })}

          <button
            className={`theme-option ${preference === 'system' ? 'active' : ''}`}
            onClick={() => setTheme('system')}
          >
            <div className="theme-option-icon system">
              <Monitor size={24} />
            </div>
            <div className="theme-option-info">
              <span className="theme-option-name">System</span>
              <span className="theme-option-desc">Follow your operating system preference</span>
            </div>
            {preference === 'system' && <Check size={18} className="theme-check" />}
          </button>
        </div>

        {preference === 'system' && (
          <p className="theme-system-hint">
            Currently using: <strong>{THEMES[currentTheme]?.displayName ?? currentTheme}</strong>
          </p>
        )}
      </Card>
    </div>
  );
}

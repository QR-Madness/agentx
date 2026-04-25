/**
 * AppearanceSection — Theme and visual customization
 * Extracted from SettingsPanel.tsx lines 850-923
 */

import {
  Palette,
  Moon,
  Sun,
  Monitor,
  Check,
} from 'lucide-react';
import { useTheme } from '../../../contexts/ThemeContext';

export default function AppearanceSection() {
  const { preference, setTheme, isDark } = useTheme();

  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Palette size={20} className="section-title-icon" />
            Appearance
          </h2>
          <p className="section-description">
            Customize the look and feel of AgentX
          </p>
        </div>
      </div>

      <div className="card">
        <h3 className="subsection-title">Theme</h3>
        <p className="subsection-description">
          Choose your preferred color scheme
        </p>

        <div className="theme-options">
          <button
            className={`theme-option ${preference === 'cosmic' ? 'active' : ''}`}
            onClick={() => setTheme('cosmic')}
          >
            <div className="theme-option-icon cosmic">
              <Moon size={24} />
            </div>
            <div className="theme-option-info">
              <span className="theme-option-name">Cosmic Dark</span>
              <span className="theme-option-desc">Deep space aesthetic with purple accents</span>
            </div>
            {preference === 'cosmic' && <Check size={18} className="theme-check" />}
          </button>

          <button
            className={`theme-option ${preference === 'light' ? 'active' : ''}`}
            onClick={() => setTheme('light')}
          >
            <div className="theme-option-icon light">
              <Sun size={24} />
            </div>
            <div className="theme-option-info">
              <span className="theme-option-name">Light</span>
              <span className="theme-option-desc">Clean, bright interface for daytime use</span>
            </div>
            {preference === 'light' && <Check size={18} className="theme-check" />}
          </button>

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
            Currently using: <strong>{isDark ? 'Cosmic Dark' : 'Light'}</strong>
          </p>
        )}
      </div>
    </div>
  );
}

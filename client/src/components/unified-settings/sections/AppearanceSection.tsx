/**
 * AppearanceSection — Theme and visual customization
 */

import {
  Palette,
  Moon,
  Sun,
  Monitor,
  Check,
  Square,
} from 'lucide-react';
import { useTheme } from '../../../contexts/ThemeContext';
import { Card, SectionHeader } from '../../ui';

export default function AppearanceSection() {
  const { preference, setTheme, isDark } = useTheme();

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
            className={`theme-option ${preference === 'professional' ? 'active' : ''}`}
            onClick={() => setTheme('professional')}
          >
            <div className="theme-option-icon professional">
              <Square size={24} />
            </div>
            <div className="theme-option-info">
              <span className="theme-option-name">Professional</span>
              <span className="theme-option-desc">Monochrome graphite — color only for emphasis</span>
            </div>
            {preference === 'professional' && <Check size={18} className="theme-check" />}
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
              <span className="theme-option-desc">Warm, neutral interface for daytime use</span>
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
      </Card>
    </div>
  );
}

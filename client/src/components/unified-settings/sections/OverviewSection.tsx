/**
 * OverviewSection — where settings open, and where to start.
 *
 * Settings used to land on Model Providers: an API-key admin page, which told
 * you nothing about the other ~200 knobs or which of them you'd already moved.
 * This answers the two questions someone actually arrives with.
 *
 * **What have I changed?** The manifest knows every setting's shipped default
 * and its current value, so the digest is computed, not curated — it can't fall
 * behind the settings it describes. Secrets are excluded (their values arrive
 * redacted, so "changed" is unanswerable and printing them would be worse).
 *
 * **Where does everything live?** Category tiles, straight off the same
 * SECTION_HIERARCHY the nav renders, so a new section appears here for free.
 *
 * Rows navigate to the owning section. Landing on the exact control — scroll,
 * focus, flash — is Wave 2; it needs anchors this deliberately doesn't invent
 * yet.
 */

import { Compass, RefreshCw, TriangleAlert } from 'lucide-react';
import { Button, Card, SectionHeader } from '../../ui';
import { SECTION_HIERARCHY, getAllSections } from './index';
import { useSettingsManifest } from '../SettingsManifestContext';
import type { SettingsManifestEntry } from '../../../lib/api';

interface OverviewSectionProps {
  onNavigate?: (sectionId: string) => void;
}

/** `recall_candidate_pool` → "Recall candidate pool"; `search.max_results` →
 *  "Search · Max results". Better than showing a raw key to someone who never
 *  reads the config file. */
function prettyKey(key: string): string {
  const parts = key.split('.');
  const humanize = (s: string) =>
    s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
  if (parts.length === 1) return humanize(parts[0]);
  return `${humanize(parts[0])} · ${humanize(parts.slice(1).join(' '))}`;
}

/**
 * Compact, non-committal rendering — this is a digest, not an editor.
 *
 * Empty strings read as "empty", never "default": several model keys ship with
 * the literal default `'inherit'`, so calling `''` the default would print
 * "inherit → default" and mean the opposite of what it says.
 */
function displayValue(value: unknown): string {
  if (value === null || value === undefined) return 'not set';
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (typeof value === 'string') return value === '' ? 'empty' : value;
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`;
  if (typeof value === 'object') return 'customised';
  return String(value);
}

export default function OverviewSection({ onNavigate }: OverviewSectionProps) {
  const manifest = useSettingsManifest();
  const sectionLabels = new Map(getAllSections().map(s => [s.id, s.label]));

  // Group the changed settings under the section that owns them, so the digest
  // reads like the nav rather than like a config dump.
  const grouped = new Map<string, SettingsManifestEntry[]>();
  for (const entry of manifest?.modified ?? []) {
    const target = entry.ui_section && sectionLabels.has(entry.ui_section)
      ? entry.ui_section
      : 'other';
    if (!grouped.has(target)) grouped.set(target, []);
    grouped.get(target)!.push(entry);
  }

  const changedCount = manifest?.modified.length ?? 0;

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Compass size={20} />}
        title="Overview"
        description="What you've changed, and where everything lives. Every setting keeps its shipped default — you can put any of them back."
      />

      {/* --- What you've changed --------------------------------------- */}
      <section className="settings-overview-block">
        <h3 className="settings-section-title">
          Changed from defaults
          {changedCount > 0 && (
            <span className="settings-overview-count">{changedCount}</span>
          )}
        </h3>

        {manifest?.loading && (
          <div className="settings-overview-state">
            <RefreshCw size={16} className="spin" />
            <span>Reading your settings…</span>
          </div>
        )}

        {!manifest?.loading && manifest?.error && (
          <Card className="settings-overview-state settings-overview-state--warn">
            <TriangleAlert size={16} />
            <div>
              <p>Couldn't load the settings registry.</p>
              <p className="settings-overview-note">
                Every section below still works — only this summary needs it.
              </p>
            </div>
            <Button variant="secondary" onClick={() => manifest.refresh()}>
              Retry
            </Button>
          </Card>
        )}

        {!manifest && (
          <p className="settings-overview-note">
            Settings summary is unavailable in this context.
          </p>
        )}

        {manifest && !manifest.loading && !manifest.error && changedCount === 0 && (
          <p className="settings-overview-note">
            Everything is on its shipped default. Anything you change will be
            listed here.
          </p>
        )}

        {[...grouped.entries()].map(([sectionId, entries]) => (
          <div key={sectionId} className="settings-overview-group">
            <h4 className="settings-overview-group-title">
              {sectionLabels.get(sectionId) ?? 'Set elsewhere in the app'}
            </h4>
            {!sectionLabels.has(sectionId) && (
              <p className="settings-overview-note">
                These aren't edited from Settings — they follow what you pick in
                the app itself, like the model chip on the composer.
              </p>
            )}
            <ul className="settings-overview-list">
              {entries.map(entry => (
                <li key={`${entry.store}:${entry.key}`} className="settings-overview-row">
                  <button
                    type="button"
                    className="settings-overview-row-btn"
                    onClick={() => sectionLabels.has(sectionId) && onNavigate?.(sectionId)}
                    disabled={!sectionLabels.has(sectionId)}
                  >
                    <span className="settings-overview-key">
                      {prettyKey(entry.key)}
                      {entry.help?.summary && (
                        <span className="settings-overview-summary">
                          {entry.help.summary}
                        </span>
                      )}
                    </span>
                    <span className="settings-overview-values">
                      <span className="settings-overview-default">
                        {displayValue(entry.default)}
                      </span>
                      <span aria-hidden="true">→</span>
                      <span className="settings-overview-current">
                        {displayValue(entry.value)}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      {/* --- Where everything lives ------------------------------------ */}
      <section className="settings-overview-block">
        <h3 className="settings-section-title">Everything else</h3>
        <div className="settings-overview-tiles">
          {Object.entries(SECTION_HIERARCHY)
            .filter(([id]) => id !== 'home')
            .map(([categoryId, category]) => (
              <Card key={categoryId} className="settings-overview-tile">
                <div className="settings-overview-tile-head">
                  {category.icon}
                  <span>{category.label}</span>
                </div>
                <ul className="settings-overview-tile-list">
                  {category.sections.map(section => (
                    <li key={section.id}>
                      <button
                        type="button"
                        className="settings-overview-link"
                        onClick={() => onNavigate?.(section.id)}
                      >
                        {section.label}
                      </button>
                    </li>
                  ))}
                </ul>
              </Card>
            ))}
        </div>
      </section>
    </div>
  );
}

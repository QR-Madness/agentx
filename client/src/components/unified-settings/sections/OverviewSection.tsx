/**
 * OverviewSection — where settings open, and where to start.
 *
 * Settings used to land on Model Providers: an API-key admin page, which told
 * you nothing about the other ~200 knobs or which of them you'd already moved.
 * This answers the three questions someone actually arrives with.
 *
 * **What am I looking for?** A search box driving the same shared query as the
 * nav's, with its results inline — because on mobile the nav is a full-screen
 * takeover, so a landing page whose only search lives in the nav isn't a
 * landing page at all.
 *
 * **Is this thing set up?** A short setup list for the handful of things
 * nothing else works without. Every row's state is read from the manifest — a
 * key is "configured" because the server says the value is non-empty, never
 * because the value itself was inspected (they arrive redacted, and should).
 *
 * **What have I changed, and where does everything live?** The digest is
 * computed from shipped defaults, so it can't fall behind the settings it
 * describes; tiles come off SECTION_HIERARCHY and carry each screen's authored
 * blurb and counts from the manifest's `sections` block.
 *
 * Rows navigate to the control and flash it. There is deliberately no reset
 * here: the control's own reset sits beside its help and its bounds, and a
 * second write path from a list — no context, no undo — is worse.
 */

import { Compass, RefreshCw, Search, TriangleAlert, Check, CornerDownRight } from 'lucide-react';
import { Button, Card, SectionHeader } from '../../ui';
import { SECTION_HIERARCHY, getAllSections } from './index';
import { useSettingsManifest } from '../SettingsManifestContext';
import { useSharedSettingsSearch } from '../SettingsSearchContext';
import { humanizeSettingKey } from '../settingLabel';
import type { SettingsManifestEntry } from '../../../lib/api';

interface OverviewSectionProps {
  onNavigate?: (sectionId: string) => void;
  /** Navigate to a section and land on the specific control. */
  onFocusSetting?: (sectionId: string, settingId: string) => void;
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

/**
 * Has this key been given a value?
 *
 * Secrets arrive redacted — `'***'` when set, `''` when not — so this is the
 * only honest question to ask about one, and it is asked without the value
 * ever being read.
 */
function isConfigured(entry?: SettingsManifestEntry): boolean {
  if (!entry) return false;
  const v = entry.value;
  if (v === null || v === undefined) return false;
  if (typeof v === 'string') return v.trim() !== '';
  if (typeof v === 'object') return Object.keys(v as object).length > 0;
  return true;
}

/** The few things a fresh install genuinely needs, in the order they matter. */
const SETUP_STEPS: {
  id: string;
  label: string;
  why: string;
  section: string;
  optional?: boolean;
  /** Any one of these being configured completes the step. */
  keys: string[];
}[] = [
  {
    id: 'provider',
    label: 'Connect a model provider',
    why: 'Nothing else works until at least one backend can be reached.',
    section: 'providers',
    keys: [
      'config:providers.anthropic.api_key',
      'config:providers.openai.api_key',
      'config:providers.openrouter.api_key',
      'config:providers.vercel.api_key',
      'config:providers.lmstudio.base_url',
      'config:providers.custom',
    ],
  },
  {
    id: 'roles',
    label: 'Pick a model for each role',
    why: 'Background work — summarising, quick utility calls, deep reasoning — follows these instead of naming a model in a dozen places.',
    section: 'model-roles',
    optional: true,
    keys: [
      'config:models.roles.fast_utility',
      'config:models.roles.deep_reasoning',
      'config:models.roles.summarizer',
    ],
  },
  {
    id: 'search',
    label: 'Add a web-search key',
    why: 'Only needed if you want agents to read the web. Without one, search tools return nothing.',
    section: 'search',
    optional: true,
    keys: ['config:search.tavily_api_key', 'config:search.brave_api_key'],
  },
];

export default function OverviewSection({ onNavigate, onFocusSetting }: OverviewSectionProps) {
  const manifest = useSettingsManifest();
  const search = useSharedSettingsSearch();
  const sectionLabels = new Map(getAllSections().map(s => [s.id, s.label]));

  const goTo = (sectionId: string, settingId?: string) => {
    if (settingId && onFocusSetting) onFocusSetting(sectionId, settingId);
    else onNavigate?.(sectionId);
  };

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
  const steps = SETUP_STEPS.map(step => ({
    ...step,
    done: step.keys.some(id => isConfigured(manifest?.entries.get(id))),
  }));
  // Once everything is in place the list has nothing to say, so it goes away
  // rather than becoming a permanent row of ticks.
  const showSetup = Boolean(manifest) && !manifest?.loading && steps.some(s => !s.done);
  const isSearching = Boolean(search?.isSearching);

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Compass size={20} />}
        title="Overview"
        description={
          manifest?.sections.get('overview')?.help?.summary
          ?? "What you've changed, and where everything lives."
        }
      />

      {/* --- Search ------------------------------------------------------ */}
      {search && (
        <div className="settings-overview-search">
          <Search size={16} className="settings-overview-search-icon" />
          <input
            type="search"
            className="settings-overview-search-input"
            placeholder="Search every setting…"
            aria-label="Search every setting"
            value={search.query}
            onChange={e => search.setQuery(e.target.value)}
          />
        </div>
      )}

      {isSearching && search && (
        <section className="settings-overview-block">
          {search.settingHits.length > 0 && (
            <>
              <h3 className="settings-section-title">
                Settings
                <span className="settings-overview-count">{search.settingHits.length}</span>
              </h3>
              <ul className="settings-overview-list">
                {search.settingHits.map(hit => (
                  <li key={hit.id}>
                    <button
                      type="button"
                      className="settings-overview-row-btn"
                      onClick={() => hit.sectionId && goTo(hit.sectionId, hit.id)}
                      disabled={!hit.sectionId}
                    >
                      <span className="settings-overview-key">
                        {hit.label}
                        {hit.summary && (
                          <span className="settings-overview-summary">{hit.summary}</span>
                        )}
                      </span>
                      {hit.sectionId && (
                        <span className="settings-overview-values">
                          <CornerDownRight size={12} aria-hidden="true" />
                          {sectionLabels.get(hit.sectionId) ?? hit.sectionId}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {search.filtered.length > 0 && (
            <>
              <h3 className="settings-section-title">Sections</h3>
              <ul className="settings-overview-list">
                {search.filtered.map(section => (
                  <li key={section.id}>
                    <button
                      type="button"
                      className="settings-overview-row-btn"
                      onClick={() => goTo(section.id)}
                    >
                      <span className="settings-overview-key">
                        {section.label}
                        {manifest?.sections.get(section.id)?.help?.summary && (
                          <span className="settings-overview-summary">
                            {manifest.sections.get(section.id)!.help!.summary}
                          </span>
                        )}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {!search.hasResults && (
            <p className="settings-overview-note">
              Nothing matches “{search.query}”. Search covers every setting's key,
              its name, and its description.
            </p>
          )}
        </section>
      )}

      {!isSearching && (
        <>
          {/* --- Getting set up ----------------------------------------- */}
          {showSetup && (
            <section className="settings-overview-block">
              <h3 className="settings-section-title">Getting set up</h3>
              <ul className="settings-overview-list">
                {steps.map(step => (
                  <li key={step.id}>
                    <button
                      type="button"
                      className="settings-overview-row-btn settings-overview-step"
                      onClick={() => goTo(step.section)}
                    >
                      <span
                        className={`settings-overview-step-mark${step.done ? ' is-done' : ''}`}
                        aria-hidden="true"
                      >
                        {step.done && <Check size={12} />}
                      </span>
                      <span className="settings-overview-key">
                        {step.label}
                        {step.optional && !step.done && (
                          <span className="settings-overview-optional"> · optional</span>
                        )}
                        <span className="settings-overview-summary">{step.why}</span>
                      </span>
                      <span className="settings-overview-values">
                        {step.done ? 'Configured' : sectionLabels.get(step.section) ?? step.section}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* --- What you've changed ------------------------------------ */}
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
                listed here, with a way back to it.
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
                        onClick={() => {
                          if (!sectionLabels.has(sectionId)) return;
                          goTo(sectionId, `${entry.store}:${entry.key}`);
                        }}
                        disabled={!sectionLabels.has(sectionId)}
                      >
                        <span className="settings-overview-key">
                          {humanizeSettingKey(entry.key)}
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

          {/* --- Where everything lives --------------------------------- */}
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
                      {category.sections.map(section => {
                        const meta = manifest?.sections.get(section.id);
                        const changed = manifest?.modifiedBySection.get(section.id) ?? 0;
                        return (
                          <li key={section.id}>
                            <button
                              type="button"
                              className="settings-overview-link"
                              onClick={() => onNavigate?.(section.id)}
                            >
                              <span className="settings-overview-link-head">
                                <span>{section.label}</span>
                                {changed > 0 && (
                                  <span
                                    className="settings-overview-tile-changed"
                                    title={`${changed} changed from default`}
                                  >
                                    {changed}
                                  </span>
                                )}
                              </span>
                              {meta?.help?.summary && (
                                <span className="settings-overview-summary">
                                  {meta.help.summary}
                                </span>
                              )}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </Card>
                ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

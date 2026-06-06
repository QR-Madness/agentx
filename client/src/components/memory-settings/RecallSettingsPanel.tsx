import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Save, Search } from 'lucide-react';
import { useRecallSettings } from '../../lib/hooks';
import { RecallSettings } from '../../lib/api';
import { ModelPickerField } from '../common/ModelPickerField';
import { useNotify } from '../../contexts/NotificationContext';
import { Button } from '../ui';
import { SettingsSection } from './fields/SettingsSection';
import { SliderField } from './fields/SliderField';
import { NumberField } from './fields/NumberField';
import { ToggleField } from './fields/ToggleField';

const oneDp = (v: number) => v.toFixed(1);

export function RecallSettingsPanel() {
  const { settings, loading, saving, error, updateSettings } = useRecallSettings();
  const { notifySuccess, notifyError } = useNotify();
  const [localSettings, setLocalSettings] = useState<Partial<RecallSettings>>({});
  const [autosaveState, setAutosaveState] = useState<'idle' | 'saving' | 'saved'>('idle');
  const baselineRef = useRef<string | null>(null);
  const autosaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
      baselineRef.current = JSON.stringify(settings);
      setAutosaveState('idle');
    }
  }, [settings]);

  const handleChange = <K extends keyof RecallSettings>(
    key: K,
    value: RecallSettings[K]
  ) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  // Autosave genuine edits after a short debounce (baseline-diff skips hydration).
  useEffect(() => {
    if (!settings || Object.keys(localSettings).length === 0) return;
    const snap = JSON.stringify(localSettings);
    if (baselineRef.current === null || snap === baselineRef.current) return;
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    autosaveTimer.current = setTimeout(async () => {
      setAutosaveState('saving');
      const ok = await updateSettings(localSettings);
      if (ok) {
        baselineRef.current = snap;
        setAutosaveState('saved');
      } else {
        setAutosaveState('idle');
      }
    }, 800);
    return () => {
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localSettings]);

  const handleSave = async () => {
    const success = await updateSettings(localSettings);
    if (success) {
      baselineRef.current = JSON.stringify(localSettings);
      setAutosaveState('saved');
      notifySuccess('Recall settings saved successfully');
    } else {
      notifyError('Failed to save recall settings');
    }
  };

  if (loading) {
    return (
      <div className="settings-panel">
        <div className="memory-loading">
          <RefreshCw size={24} className="spin" />
          <span>Loading recall settings...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="settings-panel">
        <div className="memory-error">
          Failed to load recall settings: {error.message}
        </div>
      </div>
    );
  }

  return (
    <div className="settings-panel recall-settings">
      <h2 className="settings-title">
        <Search size={20} />
        Recall Layer Settings
      </h2>
      <p className="settings-description">
        Configure enhanced retrieval techniques to improve memory recall accuracy.
        These techniques help bridge the semantic gap between questions and stored facts.
      </p>

      <SettingsSection title="Retrieval Techniques">
        <div className="settings-grid">
          <ToggleField
            label="Hybrid Search (BM25 + Vector)"
            title="Combine BM25 keyword matching with vector similarity using Reciprocal Rank Fusion"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Combines keyword matching with semantic similarity"
            checked={localSettings.recall_enable_hybrid ?? true}
            onChange={v => handleChange('recall_enable_hybrid', v)}
          />
          <ToggleField
            label="Entity-Centric Retrieval"
            title="Traverse entity relationships to find linked facts"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Finds facts via entity graph traversal"
            checked={localSettings.recall_enable_entity_centric ?? true}
            onChange={v => handleChange('recall_enable_entity_centric', v)}
          />
          <ToggleField
            label="Query Expansion"
            title="Transform questions to statement form for better matching"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint={'Transforms "When is my birthday?" → "birthday is"'}
            checked={localSettings.recall_enable_query_expansion ?? true}
            onChange={v => handleChange('recall_enable_query_expansion', v)}
          />
          <ToggleField
            label="HyDE (Hypothetical Document Embedding)"
            title="Generate hypothetical answer and search with that embedding (requires LLM)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="LLM generates hypothetical answer for better embedding match"
            checked={localSettings.recall_enable_hyde ?? false}
            onChange={v => handleChange('recall_enable_hyde', v)}
          />
          <ToggleField
            label="Self-Query (Filter Extraction)"
            title="LLM extracts structured filters from natural language (requires LLM)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="Extracts time filters, keywords from queries"
            checked={localSettings.recall_enable_self_query ?? false}
            onChange={v => handleChange('recall_enable_self_query', v)}
          />
        </div>
      </SettingsSection>

      {localSettings.recall_enable_hybrid && (
        <SettingsSection title="Hybrid Search Settings">
          <div className="settings-grid">
            <SliderField
              label="BM25 Weight"
              value={localSettings.recall_hybrid_bm25_weight ?? 0.3}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hybrid_bm25_weight', v)}
            />
            <SliderField
              label="Vector Weight"
              value={localSettings.recall_hybrid_vector_weight ?? 0.7}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hybrid_vector_weight', v)}
            />
          </div>
        </SettingsSection>
      )}

      {localSettings.recall_enable_entity_centric && (
        <SettingsSection title="Entity-Centric Settings">
          <div className="settings-grid">
            <SliderField
              label="Similarity Threshold"
              value={localSettings.recall_entity_similarity_threshold ?? 0.65}
              min={0.3} max={0.95} step={0.05}
              onChange={v => handleChange('recall_entity_similarity_threshold', v)}
            />
            <NumberField
              label="Max Entities"
              value={localSettings.recall_entity_max_entities ?? 5}
              min={1} max={20}
              onChange={v => handleChange('recall_entity_max_entities', v)}
            />
          </div>
        </SettingsSection>
      )}

      {localSettings.recall_enable_query_expansion && (
        <SettingsSection title="Query Expansion Settings">
          <div className="settings-grid">
            <NumberField
              label="Max Variants"
              value={localSettings.recall_expansion_max_variants ?? 3}
              min={1} max={10}
              onChange={v => handleChange('recall_expansion_max_variants', v)}
            />
          </div>
        </SettingsSection>
      )}

      {localSettings.recall_enable_hyde && (
        <SettingsSection title="HyDE Settings">
          <div className="settings-grid">
            <div className="setting-row">
              <ModelPickerField
                label="Model"
                value={localSettings.recall_hyde_model ?? ''}
                onChange={v => handleChange('recall_hyde_model', v)}
                onProviderChange={v => handleChange('recall_hyde_provider', v)}
                showDefault={false}
              />
            </div>
            <SliderField
              label="Temperature"
              value={localSettings.recall_hyde_temperature ?? 0.7}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hyde_temperature', v)}
            />
          </div>
        </SettingsSection>
      )}

      {localSettings.recall_enable_self_query && (
        <SettingsSection title="Self-Query Settings">
          <div className="settings-grid">
            <div className="setting-row">
              <ModelPickerField
                label="Model"
                value={localSettings.recall_self_query_model ?? ''}
                onChange={v => handleChange('recall_self_query_model', v)}
                onProviderChange={v => handleChange('recall_self_query_provider', v)}
                showDefault={false}
              />
            </div>
          </div>
        </SettingsSection>
      )}

      <div className="settings-actions">
        <span aria-live="polite" style={{ marginRight: 'auto', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {autosaveState === 'saving' ? 'Saving…' : autosaveState === 'saved' ? 'Saved ✓' : 'Autosaves'}
        </span>
        <Button onClick={handleSave} disabled={saving}>
          {saving ? (
            <><RefreshCw size={16} className="spin" /> Saving...</>
          ) : (
            <><Save size={16} /> Save Settings</>
          )}
        </Button>
      </div>
    </div>
  );
}

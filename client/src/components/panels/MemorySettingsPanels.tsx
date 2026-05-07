import { useState, useEffect } from 'react';
import {
  RefreshCw,
  Zap,
  X,
  RotateCcw,
  Save,
  Search
} from 'lucide-react';
import {
  useConsolidationSettings,
  useRecallSettings
} from '../../lib/hooks';
import { ConsolidationSettings, RecallSettings, api } from '../../lib/api';
import { ModelSelector } from '../common/ModelSelector';

// Consolidation Settings Panel Component
export function ConsolidationSettingsPanel({
  onConsolidate
}: {
  onConsolidate: (jobs?: string[]) => Promise<void>;
}) {
  const { settings, loading, saving, error, updateSettings, refresh } = useConsolidationSettings();
  const [localSettings, setLocalSettings] = useState<Partial<ConsolidationSettings>>({});
  const [consolidating, setConsolidating] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [consolidateJobs, setConsolidateJobs] = useState<string[]>(['consolidate']);
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
    }
  }, [settings]);

  const handleChange = <K extends keyof ConsolidationSettings>(
    key: K,
    value: ConsolidationSettings[K]
  ) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    const success = await updateSettings(localSettings);
    if (success) {
      setSaveMessage({ type: 'success', text: 'Settings saved successfully' });
    } else {
      setSaveMessage({ type: 'error', text: 'Failed to save settings' });
    }
    setTimeout(() => setSaveMessage(null), 3000);
  };

  const handleReset = () => {
    if (settings) {
      setLocalSettings({
        ...settings,
        extraction_system_prompt: '',
        relevance_filter_prompt: ''
      });
    }
  };

  const handleConsolidate = async () => {
    setConsolidating(true);
    try {
      await onConsolidate(consolidateJobs.length > 0 ? consolidateJobs : undefined);
    } finally {
      setConsolidating(false);
    }
  };

  const handleReinitMemory = async () => {
    if (!confirm('This will reset consolidation for ALL conversations, allowing them to be reprocessed. Continue?')) {
      return;
    }

    const deleteMemories = confirm(
      'Also DELETE all existing entities, facts, and strategies?\n\n' +
      'Click OK to delete and rebuild from scratch.\n' +
      'Click Cancel to keep existing memories and just reprocess conversations.'
    );

    setResetting(true);
    try {
      const result = await api.resetMemory(deleteMemories);
      let message = `Reset ${result.conversations_reset} conversations for reprocessing`;
      if (deleteMemories && result.memories_deleted !== undefined) {
        message += `, deleted ${result.memories_deleted} memories`;
      }
      setSaveMessage({ type: 'success', text: message });
    } catch (err) {
      setSaveMessage({
        type: 'error',
        text: `Reset failed: ${(err as Error).message}`
      });
    } finally {
      setResetting(false);
      setTimeout(() => setSaveMessage(null), 5000);
    }
  };

  const handleClearStuckJobs = async () => {
    try {
      const result = await api.clearStuckJobs();
      setSaveMessage({
        type: 'success',
        text: result.message
      });
    } catch (err) {
      setSaveMessage({
        type: 'error',
        text: `Failed to clear stuck jobs: ${(err as Error).message}`
      });
    }
    setTimeout(() => setSaveMessage(null), 5000);
  };

  const toggleJob = (job: string) => {
    setConsolidateJobs(prev =>
      prev.includes(job) ? prev.filter(j => j !== job) : [...prev, job]
    );
  };

  if (loading) {
    return (
      <div className="settings-panel">
        <div className="memory-loading">
          <RefreshCw size={24} className="spin" />
          <p>Loading settings...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="settings-panel">
        <div className="memory-error">
          <p>Failed to load settings: {error.message}</p>
          <button className="button-ghost" onClick={refresh}>
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      {/* Force Consolidate Section */}
      <div className="settings-section">
        <h3 className="settings-section-title">
          <Zap size={18} />
          Force Consolidate
        </h3>
        <p className="settings-description">
          Run consolidation immediately to extract entities and facts from recent conversations.
        </p>
        <div className="consolidate-jobs">
          <label className="job-checkbox">
            <input
              type="checkbox"
              checked={consolidateJobs.includes('consolidate')}
              onChange={() => toggleJob('consolidate')}
            />
            <span>Extract (entities, facts, relationships)</span>
          </label>
          <label className="job-checkbox">
            <input
              type="checkbox"
              checked={consolidateJobs.includes('patterns')}
              onChange={() => toggleJob('patterns')}
            />
            <span>Patterns (procedural memory)</span>
          </label>
          <label className="job-checkbox">
            <input
              type="checkbox"
              checked={consolidateJobs.includes('promote')}
              onChange={() => toggleJob('promote')}
            />
            <span>Promote (cross-channel)</span>
          </label>
        </div>
        <div className="consolidate-buttons">
          <button
            className="button-primary consolidate-now-btn"
            onClick={handleConsolidate}
            disabled={consolidating || consolidateJobs.length === 0}
          >
            {consolidating ? (
              <><RefreshCw size={16} className="spin" /> Consolidating...</>
            ) : (
              <><Zap size={16} /> Run Now</>
            )}
          </button>
          <button
            className="button-ghost reinit-btn"
            onClick={handleReinitMemory}
            disabled={resetting}
            title="Reset all conversations for reprocessing"
          >
            {resetting ? (
              <><RefreshCw size={16} className="spin" /> Resetting...</>
            ) : (
              <><RotateCcw size={16} /> Reinit Memory</>
            )}
          </button>
          <button
            className="button-ghost clear-stuck-btn"
            onClick={handleClearStuckJobs}
            title="Clear any jobs stuck in 'running' state"
          >
            <X size={16} /> Clear Stuck
          </button>
        </div>
      </div>

      {/* Extraction Settings */}
      <div className="settings-section">
        <h3 className="settings-section-title">Extraction</h3>
        <div className="settings-grid">
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.extraction_model || ''}
              onChange={v => handleChange('extraction_model', v)}
              onProviderChange={v => handleChange('extraction_provider', v)}
              showDefault={false}
              compact
            />
          </div>
          <div className="setting-row">
            <label>Temperature: {(localSettings.extraction_temperature ?? 0.2).toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.extraction_temperature ?? 0.2}
              onChange={e => handleChange('extraction_temperature', parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Max Tokens</label>
            <input
              type="number"
              value={localSettings.extraction_max_tokens ?? 2000}
              onChange={e => handleChange('extraction_max_tokens', parseInt(e.target.value) || 2000)}
              min={100}
              max={8000}
            />
          </div>
          <div className="setting-row checkbox">
            <label>
              <input
                type="checkbox"
                checked={localSettings.extraction_condense_facts ?? true}
                onChange={e => handleChange('extraction_condense_facts', e.target.checked)}
              />
              Condense facts into atomic statements
            </label>
          </div>
        </div>
        <div className="setting-textarea">
          <div className="textarea-header">
            <label>System Prompt</label>
            <button
              className="button-ghost reset-prompt-btn"
              onClick={() => handleChange('extraction_system_prompt', '')}
              title="Reset to default"
            >
              <RotateCcw size={14} />
            </button>
          </div>
          <textarea
            value={localSettings.extraction_system_prompt || ''}
            onChange={e => handleChange('extraction_system_prompt', e.target.value)}
            placeholder={settings?.default_extraction_prompt || 'Default system prompt will be used...'}
            rows={8}
          />
          {!localSettings.extraction_system_prompt && (
            <p className="prompt-hint">Leave empty to use default prompt</p>
          )}
        </div>
      </div>

      {/* Relevance Filter Settings */}
      <div className="settings-section">
        <h3 className="settings-section-title">Relevance Filter</h3>
        <div className="settings-grid">
          <div className="setting-row checkbox">
            <label>
              <input
                type="checkbox"
                checked={localSettings.relevance_filter_enabled ?? true}
                onChange={e => handleChange('relevance_filter_enabled', e.target.checked)}
              />
              Enable relevance filter (skip non-informative turns)
            </label>
          </div>
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.relevance_filter_model || ''}
              onChange={v => handleChange('relevance_filter_model', v)}
              showDefault={false}
              compact
            />
          </div>
          <div className="setting-row">
            <label>Max Tokens</label>
            <input
              type="number"
              value={localSettings.relevance_filter_max_tokens ?? 500}
              onChange={e => handleChange('relevance_filter_max_tokens', parseInt(e.target.value) || 500)}
              min={10}
              max={2000}
              title="Reasoning models need more tokens (500+)"
            />
          </div>
        </div>
        <div className="setting-textarea">
          <div className="textarea-header">
            <label>Relevance Prompt</label>
            <button
              className="button-ghost reset-prompt-btn"
              onClick={() => handleChange('relevance_filter_prompt', '')}
              title="Reset to default"
            >
              <RotateCcw size={14} />
            </button>
          </div>
          <textarea
            value={localSettings.relevance_filter_prompt || ''}
            onChange={e => handleChange('relevance_filter_prompt', e.target.value)}
            placeholder={settings?.default_relevance_prompt || 'Default relevance prompt will be used...'}
            rows={6}
          />
          {!localSettings.relevance_filter_prompt && (
            <p className="prompt-hint">Leave empty to use default prompt</p>
          )}
        </div>
      </div>

      {/* Combined Extraction Settings */}
      <div className="settings-section">
        <h3 className="settings-section-title">Combined Extraction</h3>
        <p className="settings-section-hint">
          Single-pass relevance + extraction. Handles ~75% of consolidation traffic when enabled.
        </p>
        <div className="settings-grid">
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.combined_extraction_model || ''}
              onChange={v => handleChange('combined_extraction_model', v)}
              showDefault={false}
              compact
            />
          </div>
          <div className="setting-row">
            <label>Temperature: {(localSettings.combined_extraction_temperature ?? 0.3).toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.combined_extraction_temperature ?? 0.3}
              onChange={e => handleChange('combined_extraction_temperature', parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Max Tokens</label>
            <input
              type="number"
              value={localSettings.combined_extraction_max_tokens ?? 2000}
              onChange={e => handleChange('combined_extraction_max_tokens', parseInt(e.target.value) || 2000)}
              min={100}
              max={8000}
            />
          </div>
        </div>
      </div>

      {/* Entity Linking Settings */}
      <div className="settings-section">
        <h3 className="settings-section-title">Entity Linking</h3>
        <div className="settings-grid">
          <div className="setting-row checkbox">
            <label>
              <input
                type="checkbox"
                checked={localSettings.entity_linking_enabled ?? true}
                onChange={e => handleChange('entity_linking_enabled', e.target.checked)}
              />
              Enable entity linking (connect facts to entities)
            </label>
          </div>
          <div className="setting-row">
            <label>Similarity Threshold: {((localSettings.entity_linking_similarity_threshold ?? 0.75) * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={localSettings.entity_linking_similarity_threshold ?? 0.75}
              onChange={e => handleChange('entity_linking_similarity_threshold', parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Quality Thresholds */}
      <div className="settings-section">
        <h3 className="settings-section-title">Quality Thresholds</h3>
        <div className="settings-grid">
          <div className="setting-row">
            <label>Min Fact Confidence: {((localSettings.fact_confidence_threshold ?? 0.7) * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.fact_confidence_threshold ?? 0.7}
              onChange={e => handleChange('fact_confidence_threshold', parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Min Promotion Confidence: {((localSettings.promotion_min_confidence ?? 0.85) * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.promotion_min_confidence ?? 0.85}
              onChange={e => handleChange('promotion_min_confidence', parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Job Scheduling */}
      <div className="settings-section">
        <h3 className="settings-section-title">Job Intervals (minutes)</h3>
        <div className="settings-grid intervals">
          <div className="setting-row">
            <label>Consolidation</label>
            <input
              type="number"
              value={localSettings.job_consolidate_interval ?? 15}
              onChange={e => handleChange('job_consolidate_interval', parseInt(e.target.value) || 15)}
              min={1}
            />
          </div>
          <div className="setting-row">
            <label>Promotion</label>
            <input
              type="number"
              value={localSettings.job_promote_interval ?? 60}
              onChange={e => handleChange('job_promote_interval', parseInt(e.target.value) || 60)}
              min={1}
            />
          </div>
          <div className="setting-row">
            <label>Entity Linking</label>
            <input
              type="number"
              value={localSettings.job_entity_linking_interval ?? 30}
              onChange={e => handleChange('job_entity_linking_interval', parseInt(e.target.value) || 30)}
              min={1}
            />
          </div>
        </div>
      </div>

      {/* Experimental */}
      <div className="settings-section experimental">
        <h3 className="settings-section-title">Experimental</h3>
        <div className="settings-grid">
          <div className="setting-row checkbox">
            <label>
              <input
                type="checkbox"
                checked={localSettings.contradiction_detection_enabled ?? false}
                onChange={e => handleChange('contradiction_detection_enabled', e.target.checked)}
              />
              Enable contradiction detection
            </label>
          </div>
          <div className="setting-row checkbox">
            <label>
              <input
                type="checkbox"
                checked={localSettings.correction_detection_enabled ?? false}
                onChange={e => handleChange('correction_detection_enabled', e.target.checked)}
              />
              Enable user correction handling
            </label>
          </div>
        </div>
      </div>

      {saveMessage && (
        <div className={`save-message ${saveMessage.type}`}>
          {saveMessage.text}
        </div>
      )}

      <div className="settings-actions">
        <button className="button-ghost" onClick={handleReset} disabled={saving}>
          <RotateCcw size={16} />
          Reset Prompts
        </button>
        <button className="button-primary" onClick={handleSave} disabled={saving}>
          {saving ? (
            <><RefreshCw size={16} className="spin" /> Saving...</>
          ) : (
            <><Save size={16} /> Save Settings</>
          )}
        </button>
      </div>
    </div>
  );
}

// RecallLayer Settings Panel Component
export function RecallSettingsPanel() {
  const { settings, loading, saving, error, updateSettings } = useRecallSettings();
  const [localSettings, setLocalSettings] = useState<Partial<RecallSettings>>({});
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
    }
  }, [settings]);

  const handleChange = <K extends keyof RecallSettings>(
    key: K,
    value: RecallSettings[K]
  ) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    const success = await updateSettings(localSettings);
    if (success) {
      setSaveMessage({ type: 'success', text: 'Recall settings saved successfully' });
    } else {
      setSaveMessage({ type: 'error', text: 'Failed to save recall settings' });
    }
    setTimeout(() => setSaveMessage(null), 3000);
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

      <div className="settings-section">
        <h3 className="settings-section-title">Retrieval Techniques</h3>
        <div className="settings-grid">
          <div className="setting-row checkbox">
            <label title="Combine BM25 keyword matching with vector similarity using Reciprocal Rank Fusion">
              <input
                type="checkbox"
                checked={localSettings.recall_enable_hybrid ?? true}
                onChange={e => handleChange('recall_enable_hybrid', e.target.checked)}
              />
              <span className="setting-label">
                Hybrid Search (BM25 + Vector)
                <span className="setting-badge recommended">Recommended</span>
              </span>
            </label>
            <span className="setting-hint">Combines keyword matching with semantic similarity</span>
          </div>

          <div className="setting-row checkbox">
            <label title="Traverse entity relationships to find linked facts">
              <input
                type="checkbox"
                checked={localSettings.recall_enable_entity_centric ?? true}
                onChange={e => handleChange('recall_enable_entity_centric', e.target.checked)}
              />
              <span className="setting-label">
                Entity-Centric Retrieval
                <span className="setting-badge recommended">Recommended</span>
              </span>
            </label>
            <span className="setting-hint">Finds facts via entity graph traversal</span>
          </div>

          <div className="setting-row checkbox">
            <label title="Transform questions to statement form for better matching">
              <input
                type="checkbox"
                checked={localSettings.recall_enable_query_expansion ?? true}
                onChange={e => handleChange('recall_enable_query_expansion', e.target.checked)}
              />
              <span className="setting-label">
                Query Expansion
                <span className="setting-badge recommended">Recommended</span>
              </span>
            </label>
            <span className="setting-hint">Transforms "When is my birthday?" → "birthday is"</span>
          </div>

          <div className="setting-row checkbox">
            <label title="Generate hypothetical answer and search with that embedding (requires LLM)">
              <input
                type="checkbox"
                checked={localSettings.recall_enable_hyde ?? false}
                onChange={e => handleChange('recall_enable_hyde', e.target.checked)}
              />
              <span className="setting-label">
                HyDE (Hypothetical Document Embedding)
                <span className="setting-badge expensive">LLM Required</span>
              </span>
            </label>
            <span className="setting-hint">LLM generates hypothetical answer for better embedding match</span>
          </div>

          <div className="setting-row checkbox">
            <label title="LLM extracts structured filters from natural language (requires LLM)">
              <input
                type="checkbox"
                checked={localSettings.recall_enable_self_query ?? false}
                onChange={e => handleChange('recall_enable_self_query', e.target.checked)}
              />
              <span className="setting-label">
                Self-Query (Filter Extraction)
                <span className="setting-badge expensive">LLM Required</span>
              </span>
            </label>
            <span className="setting-hint">Extracts time filters, keywords from queries</span>
          </div>
        </div>
      </div>

      {localSettings.recall_enable_hybrid && (
        <div className="settings-section">
          <h3 className="settings-section-title">Hybrid Search Settings</h3>
          <div className="settings-grid">
            <div className="setting-row">
              <label>BM25 Weight</label>
              <div className="setting-input-group">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings.recall_hybrid_bm25_weight ?? 0.3}
                  onChange={e => handleChange('recall_hybrid_bm25_weight', parseFloat(e.target.value))}
                />
                <span className="setting-value">{(localSettings.recall_hybrid_bm25_weight ?? 0.3).toFixed(1)}</span>
              </div>
            </div>
            <div className="setting-row">
              <label>Vector Weight</label>
              <div className="setting-input-group">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings.recall_hybrid_vector_weight ?? 0.7}
                  onChange={e => handleChange('recall_hybrid_vector_weight', parseFloat(e.target.value))}
                />
                <span className="setting-value">{(localSettings.recall_hybrid_vector_weight ?? 0.7).toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {localSettings.recall_enable_entity_centric && (
        <div className="settings-section">
          <h3 className="settings-section-title">Entity-Centric Settings</h3>
          <div className="settings-grid">
            <div className="setting-row">
              <label>Similarity Threshold</label>
              <div className="setting-input-group">
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={localSettings.recall_entity_similarity_threshold ?? 0.65}
                  onChange={e => handleChange('recall_entity_similarity_threshold', parseFloat(e.target.value))}
                />
                <span className="setting-value">{(localSettings.recall_entity_similarity_threshold ?? 0.65).toFixed(2)}</span>
              </div>
            </div>
            <div className="setting-row">
              <label>Max Entities</label>
              <input
                type="number"
                min="1"
                max="20"
                value={localSettings.recall_entity_max_entities ?? 5}
                onChange={e => handleChange('recall_entity_max_entities', parseInt(e.target.value))}
              />
            </div>
          </div>
        </div>
      )}

      {localSettings.recall_enable_query_expansion && (
        <div className="settings-section">
          <h3 className="settings-section-title">Query Expansion Settings</h3>
          <div className="settings-grid">
            <div className="setting-row">
              <label>Max Variants</label>
              <input
                type="number"
                min="1"
                max="10"
                value={localSettings.recall_expansion_max_variants ?? 3}
                onChange={e => handleChange('recall_expansion_max_variants', parseInt(e.target.value))}
              />
            </div>
          </div>
        </div>
      )}

      {localSettings.recall_enable_hyde && (
        <div className="settings-section">
          <h3 className="settings-section-title">HyDE Settings</h3>
          <div className="settings-grid">
            <div className="setting-row">
              <ModelSelector
                label="Model"
                value={localSettings.recall_hyde_model ?? ''}
                onChange={v => handleChange('recall_hyde_model', v)}
                onProviderChange={v => handleChange('recall_hyde_provider', v)}
                showDefault={false}
                compact
              />
            </div>
            <div className="setting-row">
              <label>Temperature</label>
              <div className="setting-input-group">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings.recall_hyde_temperature ?? 0.7}
                  onChange={e => handleChange('recall_hyde_temperature', parseFloat(e.target.value))}
                />
                <span className="setting-value">{(localSettings.recall_hyde_temperature ?? 0.7).toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {localSettings.recall_enable_self_query && (
        <div className="settings-section">
          <h3 className="settings-section-title">Self-Query Settings</h3>
          <div className="settings-grid">
            <div className="setting-row">
              <ModelSelector
                label="Model"
                value={localSettings.recall_self_query_model ?? ''}
                onChange={v => handleChange('recall_self_query_model', v)}
                onProviderChange={v => handleChange('recall_self_query_provider', v)}
                showDefault={false}
                compact
              />
            </div>
          </div>
        </div>
      )}

      {saveMessage && (
        <div className={`save-message ${saveMessage.type}`}>
          {saveMessage.text}
        </div>
      )}

      <div className="settings-actions">
        <button className="button-primary" onClick={handleSave} disabled={saving}>
          {saving ? (
            <><RefreshCw size={16} className="spin" /> Saving...</>
          ) : (
            <><Save size={16} /> Save Settings</>
          )}
        </button>
      </div>
    </div>
  );
}

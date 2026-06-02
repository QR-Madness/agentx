import { useState, useEffect, useId } from 'react';
import { RefreshCw, Zap, X, RotateCcw, Save } from 'lucide-react';
import { useConsolidationSettings } from '../../lib/hooks';
import { ConsolidationSettings, api } from '../../lib/api';
import { ModelSelector } from '../common/ModelSelector';
import { useNotify } from '../../contexts/NotificationContext';
import { Button, Checkbox, Label } from '../ui';
import { SettingsSection } from './fields/SettingsSection';
import { SliderField } from './fields/SliderField';
import { NumberField } from './fields/NumberField';
import { ToggleField } from './fields/ToggleField';
import { PromptField } from './fields/PromptField';

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

const CONSOLIDATE_JOBS: { id: string; label: string }[] = [
  { id: 'consolidate', label: 'Extract (entities, facts, relationships)' },
  { id: 'patterns', label: 'Patterns (procedural memory)' },
  { id: 'promote', label: 'Promote (cross-channel)' },
];

export function ConsolidationSettingsPanel({
  onConsolidate,
}: {
  onConsolidate: (jobs?: string[]) => Promise<void>;
}) {
  const { settings, loading, saving, error, updateSettings, refresh } = useConsolidationSettings();
  const { notifySuccess, notifyError } = useNotify();
  const [localSettings, setLocalSettings] = useState<Partial<ConsolidationSettings>>({});
  const [consolidating, setConsolidating] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [consolidateJobs, setConsolidateJobs] = useState<string[]>(['consolidate']);
  const jobIdPrefix = useId();

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
      notifySuccess('Settings saved successfully');
    } else {
      notifyError('Failed to save settings');
    }
  };

  const handleReset = () => {
    if (settings) {
      setLocalSettings({
        ...settings,
        extraction_system_prompt: '',
        relevance_filter_prompt: '',
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
      notifySuccess(message);
    } catch (err) {
      notifyError(err, 'Reset failed');
    } finally {
      setResetting(false);
    }
  };

  const handleClearStuckJobs = async () => {
    try {
      const result = await api.clearStuckJobs();
      notifySuccess(result.message);
    } catch (err) {
      notifyError(err, 'Failed to clear stuck jobs');
    }
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
          <Button variant="ghost" onClick={refresh}>
            <RefreshCw size={16} /> Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      <SettingsSection
        title="Force Consolidate"
        icon={<Zap size={18} />}
        description="Run consolidation immediately to extract entities and facts from recent conversations."
      >
        <div className="consolidate-jobs">
          {CONSOLIDATE_JOBS.map(job => {
            const id = `${jobIdPrefix}-${job.id}`;
            return (
              <label key={job.id} className="job-checkbox flex items-center gap-2">
                <Checkbox
                  id={id}
                  checked={consolidateJobs.includes(job.id)}
                  onCheckedChange={() => toggleJob(job.id)}
                />
                <Label htmlFor={id}>{job.label}</Label>
              </label>
            );
          })}
        </div>
        <div className="consolidate-buttons">
          <Button
            className="consolidate-now-btn"
            onClick={handleConsolidate}
            disabled={consolidating || consolidateJobs.length === 0}
          >
            {consolidating ? (
              <><RefreshCw size={16} className="spin" /> Consolidating...</>
            ) : (
              <><Zap size={16} /> Run Now</>
            )}
          </Button>
          <Button
            variant="ghost"
            className="reinit-btn"
            onClick={handleReinitMemory}
            disabled={resetting}
            title="Reset all conversations for reprocessing"
          >
            {resetting ? (
              <><RefreshCw size={16} className="spin" /> Resetting...</>
            ) : (
              <><RotateCcw size={16} /> Reinit Memory</>
            )}
          </Button>
          <Button
            variant="ghost"
            className="clear-stuck-btn"
            onClick={handleClearStuckJobs}
            title="Clear any jobs stuck in 'running' state"
          >
            <X size={16} /> Clear Stuck
          </Button>
        </div>
      </SettingsSection>

      <SettingsSection title="Extraction">
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
          <SliderField
            label="Temperature"
            value={localSettings.extraction_temperature ?? 0.2}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('extraction_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={localSettings.extraction_max_tokens ?? 2000}
            min={100} max={8000} fallback={2000}
            onChange={v => handleChange('extraction_max_tokens', v)}
          />
          <ToggleField
            label="Condense facts into atomic statements"
            checked={localSettings.extraction_condense_facts ?? true}
            onChange={v => handleChange('extraction_condense_facts', v)}
          />
        </div>
        <PromptField
          label="System Prompt"
          value={localSettings.extraction_system_prompt || ''}
          onChange={v => handleChange('extraction_system_prompt', v)}
          onReset={() => handleChange('extraction_system_prompt', '')}
          placeholder={settings?.default_extraction_prompt || 'Default system prompt will be used...'}
          rows={8}
        />
      </SettingsSection>

      <SettingsSection title="Relevance Filter">
        <div className="settings-grid">
          <ToggleField
            label="Enable relevance filter (skip non-informative turns)"
            checked={localSettings.relevance_filter_enabled ?? true}
            onChange={v => handleChange('relevance_filter_enabled', v)}
          />
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.relevance_filter_model || ''}
              onChange={v => handleChange('relevance_filter_model', v)}
              showDefault={false}
              compact
            />
          </div>
          <SliderField
            label="Temperature"
            value={localSettings.relevance_filter_temperature ?? 0.1}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('relevance_filter_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={localSettings.relevance_filter_max_tokens ?? 500}
            min={10} max={2000} fallback={500}
            title="Reasoning models need more tokens (500+)"
            onChange={v => handleChange('relevance_filter_max_tokens', v)}
          />
        </div>
        <PromptField
          label="Relevance Prompt"
          value={localSettings.relevance_filter_prompt || ''}
          onChange={v => handleChange('relevance_filter_prompt', v)}
          onReset={() => handleChange('relevance_filter_prompt', '')}
          placeholder={settings?.default_relevance_prompt || 'Default relevance prompt will be used...'}
          rows={6}
        />
      </SettingsSection>

      <SettingsSection
        title="Combined Extraction"
        description={
          <>
            Merges the separate <strong>Relevance Filter</strong> and{' '}
            <strong>Extraction</strong> calls into a single LLM pass — roughly
            75% fewer calls during consolidation. When a model is set here it{' '}
            <strong>overrides</strong> those two stages for the turns it handles;
            leave it empty to fall back to running them separately.
          </>
        }
      >
        <div className="settings-grid">
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.combined_extraction_model || ''}
              onChange={v => handleChange('combined_extraction_model', v)}
              showDefault={false}
              compact
            />
            <Button
              variant="ghost"
              size="sm"
              className="setting-reset-inline"
              onClick={() => handleChange('combined_extraction_model', '')}
              disabled={!localSettings.combined_extraction_model}
              title="Reset to default (run Relevance + Extraction separately)"
            >
              <RotateCcw size={13} />
              <span>Reset to default</span>
            </Button>
          </div>
          <SliderField
            label="Temperature"
            value={localSettings.combined_extraction_temperature ?? 0.3}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('combined_extraction_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={localSettings.combined_extraction_max_tokens ?? 2000}
            min={100} max={8000} fallback={2000}
            onChange={v => handleChange('combined_extraction_max_tokens', v)}
          />
        </div>
      </SettingsSection>

      <SettingsSection
        title="Trajectory Compression"
        description="Consolidates older tool-call rounds into a Knowledge block when context exceeds the threshold."
      >
        <div className="settings-grid">
          <ToggleField
            label="Enable trajectory compression"
            checked={localSettings.trajectory_compression_enabled ?? true}
            onChange={v => handleChange('trajectory_compression_enabled', v)}
          />
          <div className="setting-row">
            <ModelSelector
              label="Model"
              value={localSettings.trajectory_compression_model || ''}
              onChange={v => handleChange('trajectory_compression_model', v)}
              showDefault={false}
              compact
            />
          </div>
          <SliderField
            label="Temperature"
            value={localSettings.trajectory_compression_temperature ?? 0.2}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('trajectory_compression_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={localSettings.trajectory_compression_max_tokens ?? 1500}
            min={100} max={8000} fallback={1500}
            onChange={v => handleChange('trajectory_compression_max_tokens', v)}
          />
          <SliderField
            label="Trigger Threshold (% of context)"
            value={localSettings.trajectory_compression_threshold_ratio ?? 0.75}
            min={0.5} max={0.95} step={0.05}
            format={pct}
            onChange={v => handleChange('trajectory_compression_threshold_ratio', v)}
          />
          <NumberField
            label="Preserve Recent Rounds"
            value={localSettings.trajectory_compression_preserve_recent_rounds ?? 2}
            min={1} max={5} fallback={2}
            onChange={v => handleChange('trajectory_compression_preserve_recent_rounds', v)}
          />
        </div>
      </SettingsSection>

      <SettingsSection title="Entity Linking">
        <div className="settings-grid">
          <ToggleField
            label="Enable entity linking (connect facts to entities)"
            checked={localSettings.entity_linking_enabled ?? true}
            onChange={v => handleChange('entity_linking_enabled', v)}
          />
          <SliderField
            label="Similarity Threshold"
            value={localSettings.entity_linking_similarity_threshold ?? 0.75}
            min={0.5} max={1} step={0.05}
            format={pct}
            onChange={v => handleChange('entity_linking_similarity_threshold', v)}
          />
          <ToggleField
            label="Use LLM for ambiguous matches"
            checked={localSettings.entity_linking_use_llm_disambiguation ?? false}
            onChange={v => handleChange('entity_linking_use_llm_disambiguation', v)}
          />
          {localSettings.entity_linking_use_llm_disambiguation && (
            <div className="setting-row">
              <ModelSelector
                label="Disambiguation Model"
                value={localSettings.entity_linking_model || ''}
                onChange={v => handleChange('entity_linking_model', v)}
                showDefault={false}
                compact
              />
            </div>
          )}
        </div>
      </SettingsSection>

      <SettingsSection title="Quality Thresholds">
        <div className="settings-grid">
          <SliderField
            label="Min Fact Confidence"
            value={localSettings.fact_confidence_threshold ?? 0.7}
            min={0} max={1} step={0.05}
            format={pct}
            onChange={v => handleChange('fact_confidence_threshold', v)}
          />
          <SliderField
            label="Min Promotion Confidence"
            value={localSettings.promotion_min_confidence ?? 0.85}
            min={0} max={1} step={0.05}
            format={pct}
            onChange={v => handleChange('promotion_min_confidence', v)}
          />
        </div>
      </SettingsSection>

      <SettingsSection title="Job Intervals (minutes)">
        <div className="settings-grid intervals">
          <NumberField
            label="Consolidation"
            value={localSettings.job_consolidate_interval ?? 15}
            min={1} fallback={15}
            onChange={v => handleChange('job_consolidate_interval', v)}
          />
          <NumberField
            label="Promotion"
            value={localSettings.job_promote_interval ?? 60}
            min={1} fallback={60}
            onChange={v => handleChange('job_promote_interval', v)}
          />
          <NumberField
            label="Entity Linking"
            value={localSettings.job_entity_linking_interval ?? 30}
            min={1} fallback={30}
            onChange={v => handleChange('job_entity_linking_interval', v)}
          />
        </div>
      </SettingsSection>

      <div className="settings-section experimental">
        <h3 className="settings-section-title">Experimental</h3>
        <div className="settings-grid">
          <ToggleField
            label="Enable contradiction detection"
            checked={localSettings.contradiction_detection_enabled ?? false}
            onChange={v => handleChange('contradiction_detection_enabled', v)}
          />
          {localSettings.contradiction_detection_enabled && (
            <>
              <div className="setting-row">
                <ModelSelector
                  label="Contradiction Model"
                  value={localSettings.contradiction_model || ''}
                  onChange={v => handleChange('contradiction_model', v)}
                  showDefault={false}
                  compact
                />
              </div>
              <SliderField
                label="Temperature"
                value={localSettings.contradiction_temperature ?? 0.2}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('contradiction_temperature', v)}
              />
              <NumberField
                label="Max Tokens"
                value={localSettings.contradiction_max_tokens ?? 500}
                min={100} max={4000} fallback={500}
                onChange={v => handleChange('contradiction_max_tokens', v)}
              />
            </>
          )}
          <ToggleField
            label="Enable user correction handling"
            checked={localSettings.correction_detection_enabled ?? false}
            onChange={v => handleChange('correction_detection_enabled', v)}
          />
          {localSettings.correction_detection_enabled && (
            <>
              <div className="setting-row">
                <ModelSelector
                  label="Correction Model"
                  value={localSettings.correction_model || ''}
                  onChange={v => handleChange('correction_model', v)}
                  showDefault={false}
                  compact
                />
              </div>
              <SliderField
                label="Temperature"
                value={localSettings.correction_temperature ?? 0.2}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('correction_temperature', v)}
              />
              <NumberField
                label="Max Tokens"
                value={localSettings.correction_max_tokens ?? 500}
                min={100} max={4000} fallback={500}
                onChange={v => handleChange('correction_max_tokens', v)}
              />
            </>
          )}
        </div>
      </div>

      <div className="settings-actions">
        <Button variant="ghost" onClick={handleReset} disabled={saving}>
          <RotateCcw size={16} />
          Reset Prompts
        </Button>
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

import { useState, useEffect, useId } from 'react';
import { RefreshCw, Zap, X, RotateCcw } from 'lucide-react';
import { useApi, useSettingsAutosave } from '../../lib/hooks';
import { ConsolidationSettings, api, type ModelRoleMember, type ModelRoleName } from '../../lib/api';
import { ModelPickerField } from '../common/ModelPickerField';
import { useNotify } from '../../contexts/NotificationContext';
import { Badge, Button, Checkbox, Label } from '../ui';
import { useConfirm } from '../ui/ConfirmDialog';
import { SettingsSection, SliderField, NumberField, ToggleField, SaveStatusChip } from '../settings/fields';

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

const CONSOLIDATE_JOBS: { id: string; label: string }[] = [
  { id: 'consolidate', label: 'Extract (entities, facts, relationships)' },
  { id: 'patterns', label: 'Patterns (procedural memory)' },
  { id: 'promote', label: 'Promote (cross-channel)' },
];

const ROLE_LABELS: Record<ModelRoleName, string> = {
  fast_utility: 'Fast Utility',
  deep_reasoning: 'Deep Reasoning',
  summarizer: 'Summarizer',
};

/** Autosave draft shape (index signature required by useSettingsAutosave). */
type ConsolidationDraft = ConsolidationSettings & Record<string, unknown>;

/** Human-readable tail of a provider:model id. */
function shortModel(id: string): string {
  const idx = id.indexOf(':');
  return idx >= 0 ? id.slice(idx + 1) : id;
}

/**
 * Effective-model chip under a stage's model picker (GET /api/models/roles).
 * Explicit values render nothing — the picker already shows them; stages
 * following their role get an accent badge, unset stages note the fallback chain.
 */
function StageRoleChip({
  members,
  member,
}: {
  members: ModelRoleMember[] | undefined;
  member: string;
}) {
  const m = members?.find(x => x.member === member);
  if (!m || m.following === 'explicit') return null;
  return (
    <div className="setting-hint">
      {m.following === 'role' ? (
        <Badge variant="accent" size="sm" title={m.effective}>
          Following {ROLE_LABELS[m.role]} · {shortModel(m.effective)}
        </Badge>
      ) : (
        <span>Using fallback chain</span>
      )}
    </div>
  );
}

export function ConsolidationSettingsPanel({
  onConsolidate,
}: {
  onConsolidate: (jobs?: string[]) => Promise<void>;
}) {
  const { notifySuccess, notifyError } = useNotify();
  const confirm = useConfirm();
  const [consolidating, setConsolidating] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [consolidateJobs, setConsolidateJobs] = useState<string[]>(['consolidate']);
  const jobIdPrefix = useId();

  const { settings, loading, error, status, update, refresh } =
    useSettingsAutosave<ConsolidationDraft>({
      load: async () => (await api.getConsolidationSettings()) as ConsolidationDraft,
      save: changed => api.updateConsolidationSettings(changed),
      onError: err => notifyError(err, 'Consolidation settings'),
    });

  // Effective-model/role chips for the stage pickers; re-fetched after each
  // successful save so the chips track the just-saved stage models.
  const { data: modelRoles, refresh: refreshModelRoles } = useApi(() => api.getModelRoles(), []);
  const roleMembers = modelRoles?.members;
  useEffect(() => {
    if (status === 'saved') void refreshModelRoles();
  }, [status, refreshModelRoles]);

  const handleChange = <K extends keyof ConsolidationSettings>(
    key: K,
    value: ConsolidationSettings[K]
  ) => {
    update({ [key]: value } as Partial<ConsolidationDraft>);
  };

  // The stage models that inherit `feature_default_model` when left empty.
  const STAGE_MODEL_KEYS: (keyof ConsolidationSettings)[] = [
    'extraction_model',
    'relevance_filter_model',
    'combined_extraction_model',
    'trajectory_compression_model',
    'entity_linking_model',
    'contradiction_model',
    'correction_model',
  ];

  // "Apply to all stages": clear every stage model so they all inherit the bulk
  // default; the user can then override individual stages as needed.
  const applyDefaultToAllStages = () => {
    const patch: Record<string, unknown> = {};
    for (const k of STAGE_MODEL_KEYS) patch[k as string] = '';
    update(patch as Partial<ConsolidationDraft>);
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
    const proceed = await confirm({
      title: 'Reset consolidation?',
      body: 'This will reset consolidation for ALL conversations, allowing them to be reprocessed.',
      confirmLabel: 'Continue',
      danger: true,
    });
    if (!proceed) return;

    const deleteMemories = await confirm({
      title: 'Also delete existing memories?',
      body: 'Delete all existing entities, facts, and strategies, and rebuild from scratch? Choose "Keep memories" to keep existing memories and just reprocess conversations.',
      confirmLabel: 'Delete & rebuild',
      cancelLabel: 'Keep memories',
      danger: true,
    });

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

  if (!settings) {
    return (
      <div className="settings-panel">
        <div className="memory-error">
          <p>Failed to load settings{error ? `: ${error.message}` : ''}</p>
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

      <SettingsSection title="Default model for all memory stages">
        <div className="settings-grid">
          <div className="setting-row">
            <ModelPickerField
              label="Default model"
              value={settings.feature_default_model || ''}
              onChange={v => handleChange('feature_default_model', v)}
              showDefault={false}
            />
          </div>
          <p className="setting-hint">
            Stages left blank below inherit this model; if this is blank too, they inherit your
            default chat model. Set it once to point all of memory at one model.
          </p>
          <p className="setting-hint">
            Prefer Model Roles (Settings → Infrastructure → Model Roles) for workload-level
            control — a stage&apos;s assigned role takes precedence over this bulk default, which
            only applies to stages with no role model resolved.
          </p>
          <Button
            variant="secondary"
            onClick={applyDefaultToAllStages}
            title="Clear every stage below so they all use this default"
          >
            Apply to all stages
          </Button>
        </div>
      </SettingsSection>

      <SettingsSection title="Extraction">
        <div className="settings-grid">
          <div className="setting-row">
            <ModelPickerField
              label="Model (blank = inherit default)"
              value={settings.extraction_model || ''}
              onChange={v => handleChange('extraction_model', v)}
              showDefault={false}
            />
            <StageRoleChip members={roleMembers} member="extraction" />
          </div>
          <SliderField
            label="Temperature"
            value={settings.extraction_temperature ?? 0.2}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('extraction_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={settings.extraction_max_tokens ?? 2000}
            min={100} max={8000} fallback={2000}
            onChange={v => handleChange('extraction_max_tokens', v)}
          />
          <ToggleField
            label="Condense facts into atomic statements"
            checked={settings.extraction_condense_facts ?? true}
            onChange={v => handleChange('extraction_condense_facts', v)}
          />
        </div>
        <p className="setting-hint">
          The extraction system prompt now lives in Settings → Prompts → Feature Prompts.
        </p>
      </SettingsSection>

      <SettingsSection title="Relevance Filter">
        <div className="settings-grid">
          <ToggleField
            label="Enable relevance filter (skip non-informative turns)"
            checked={settings.relevance_filter_enabled ?? true}
            onChange={v => handleChange('relevance_filter_enabled', v)}
          />
          <div className="setting-row">
            <ModelPickerField
              label="Model"
              value={settings.relevance_filter_model || ''}
              onChange={v => handleChange('relevance_filter_model', v)}
              showDefault={false}
            />
            <StageRoleChip members={roleMembers} member="relevance_filter" />
          </div>
          <SliderField
            label="Temperature"
            value={settings.relevance_filter_temperature ?? 0.1}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('relevance_filter_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={settings.relevance_filter_max_tokens ?? 500}
            min={10} max={2000} fallback={500}
            title="Reasoning models need more tokens (500+)"
            onChange={v => handleChange('relevance_filter_max_tokens', v)}
          />
        </div>
        <p className="setting-hint">
          The relevance filter prompt now lives in Settings → Prompts → Feature Prompts.
        </p>
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
            <ModelPickerField
              label="Model"
              value={settings.combined_extraction_model || ''}
              onChange={v => handleChange('combined_extraction_model', v)}
              showDefault={false}
            />
            <Button
              variant="ghost"
              size="sm"
              className="setting-reset-inline"
              onClick={() => handleChange('combined_extraction_model', '')}
              disabled={!settings.combined_extraction_model}
              title="Reset to default (run Relevance + Extraction separately)"
            >
              <RotateCcw size={13} />
              <span>Reset to default</span>
            </Button>
            <StageRoleChip members={roleMembers} member="combined_extraction" />
          </div>
          <SliderField
            label="Temperature"
            value={settings.combined_extraction_temperature ?? 0.3}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('combined_extraction_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={settings.combined_extraction_max_tokens ?? 2000}
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
            checked={settings.trajectory_compression_enabled ?? true}
            onChange={v => handleChange('trajectory_compression_enabled', v)}
          />
          <div className="setting-row">
            <ModelPickerField
              label="Model"
              value={settings.trajectory_compression_model || ''}
              onChange={v => handleChange('trajectory_compression_model', v)}
              showDefault={false}
            />
          </div>
          <SliderField
            label="Temperature"
            value={settings.trajectory_compression_temperature ?? 0.2}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('trajectory_compression_temperature', v)}
          />
          <NumberField
            label="Max Tokens"
            value={settings.trajectory_compression_max_tokens ?? 1500}
            min={100} max={8000} fallback={1500}
            onChange={v => handleChange('trajectory_compression_max_tokens', v)}
          />
          <SliderField
            label="Trigger Threshold (% of context)"
            value={settings.trajectory_compression_threshold_ratio ?? 0.75}
            min={0.5} max={0.95} step={0.05}
            format={pct}
            onChange={v => handleChange('trajectory_compression_threshold_ratio', v)}
          />
          <NumberField
            label="Preserve Recent Rounds"
            value={settings.trajectory_compression_preserve_recent_rounds ?? 2}
            min={1} max={5} fallback={2}
            onChange={v => handleChange('trajectory_compression_preserve_recent_rounds', v)}
          />
        </div>
      </SettingsSection>

      <SettingsSection title="Entity Linking">
        <div className="settings-grid">
          <ToggleField
            label="Enable entity linking (connect facts to entities)"
            checked={settings.entity_linking_enabled ?? true}
            onChange={v => handleChange('entity_linking_enabled', v)}
          />
          <SliderField
            label="Similarity Threshold"
            value={settings.entity_linking_similarity_threshold ?? 0.75}
            min={0.5} max={1} step={0.05}
            format={pct}
            onChange={v => handleChange('entity_linking_similarity_threshold', v)}
          />
          <ToggleField
            label="Use LLM for ambiguous matches"
            checked={settings.entity_linking_use_llm_disambiguation ?? false}
            onChange={v => handleChange('entity_linking_use_llm_disambiguation', v)}
          />
          {settings.entity_linking_use_llm_disambiguation && (
            <div className="setting-row">
              <ModelPickerField
                label="Disambiguation Model"
                value={settings.entity_linking_model || ''}
                onChange={v => handleChange('entity_linking_model', v)}
                showDefault={false}
              />
              <StageRoleChip members={roleMembers} member="entity_linking" />
            </div>
          )}
        </div>
      </SettingsSection>

      <SettingsSection title="Quality Thresholds">
        <div className="settings-grid">
          <SliderField
            label="Min Fact Confidence"
            value={settings.fact_confidence_threshold ?? 0.7}
            min={0} max={1} step={0.05}
            format={pct}
            onChange={v => handleChange('fact_confidence_threshold', v)}
          />
          <SliderField
            label="Min Promotion Confidence"
            value={settings.promotion_min_confidence ?? 0.85}
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
            value={settings.job_consolidate_interval ?? 15}
            min={1} fallback={15}
            onChange={v => handleChange('job_consolidate_interval', v)}
          />
          <NumberField
            label="Promotion"
            value={settings.job_promote_interval ?? 60}
            min={1} fallback={60}
            onChange={v => handleChange('job_promote_interval', v)}
          />
          <NumberField
            label="Entity Linking"
            value={settings.job_entity_linking_interval ?? 30}
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
            checked={settings.contradiction_detection_enabled ?? false}
            onChange={v => handleChange('contradiction_detection_enabled', v)}
          />
          {settings.contradiction_detection_enabled && (
            <>
              <div className="setting-row">
                <ModelPickerField
                  label="Contradiction Model"
                  value={settings.contradiction_model || ''}
                  onChange={v => handleChange('contradiction_model', v)}
                  showDefault={false}
                />
                <StageRoleChip members={roleMembers} member="contradiction" />
              </div>
              <SliderField
                label="Temperature"
                value={settings.contradiction_temperature ?? 0.2}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('contradiction_temperature', v)}
              />
              <NumberField
                label="Max Tokens"
                value={settings.contradiction_max_tokens ?? 500}
                min={100} max={4000} fallback={500}
                onChange={v => handleChange('contradiction_max_tokens', v)}
              />
            </>
          )}
          <ToggleField
            label="Enable user correction handling"
            checked={settings.correction_detection_enabled ?? false}
            onChange={v => handleChange('correction_detection_enabled', v)}
          />
          {settings.correction_detection_enabled && (
            <>
              <div className="setting-row">
                <ModelPickerField
                  label="Correction Model"
                  value={settings.correction_model || ''}
                  onChange={v => handleChange('correction_model', v)}
                  showDefault={false}
                />
                <StageRoleChip members={roleMembers} member="correction" />
              </div>
              <SliderField
                label="Temperature"
                value={settings.correction_temperature ?? 0.2}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('correction_temperature', v)}
              />
              <NumberField
                label="Max Tokens"
                value={settings.correction_max_tokens ?? 500}
                min={100} max={4000} fallback={500}
                onChange={v => handleChange('correction_max_tokens', v)}
              />
            </>
          )}
        </div>
      </div>

      <div className="settings-actions">
        <SaveStatusChip status={status} />
      </div>
    </div>
  );
}

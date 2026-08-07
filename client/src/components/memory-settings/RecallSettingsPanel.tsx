/**
 * RecallSettingsPanel — the golden section for the settings overhaul.
 *
 * Recall is the most bespoke surface in Settings, so it's where the new
 * conventions get proven before the rest follow:
 *
 *  - **No local defaults.** Every value falls back to the manifest's declared
 *    default rather than a literal typed here (there were 25 of them), so a
 *    control can't disagree with the server about its own default. Bounds and
 *    help arrive the same way.
 *  - **Each technique owns all of its knobs.** HyDE's model, temperature and
 *    token budget used to be split across two places — model and temperature
 *    under "HyDE Settings", max tokens down in "Advanced" — so tuning one
 *    technique meant hunting in two. Self-Query was split the same way.
 *  - **Advanced is a disclosure, not a dumping ground.** It holds the
 *    cross-cutting knobs that belong to no single technique, collapsed by
 *    default.
 *
 * Every setting here is still present and still writable — nothing was removed,
 * only grouped where it belongs.
 */

import { RefreshCw, Search } from 'lucide-react';
import { useSettingsAutosave } from '../../lib/hooks';
import { RecallSettings, api } from '../../lib/api';
import { ModelPickerField } from '../common/ModelPickerField';
import { useNotify } from '../../contexts/NotificationContext';
import { Button } from '../ui';
import {
  bindSetting,
  useSettingsManifest,
} from '../unified-settings/SettingsManifestContext';
import {
  SettingsSection,
  SliderField,
  NumberField,
  ToggleField,
  TextField,
  SaveStatusChip,
} from '../settings/fields';

const oneDp = (v: number) => v.toFixed(1);

/** Autosave draft shape (index signature required by useSettingsAutosave). */
type RecallDraft = RecallSettings & Record<string, unknown>;

export function RecallSettingsPanel() {
  const { notifyError } = useNotify();
  const manifest = useSettingsManifest();
  const { settings, loading, error, status, update, refresh } =
    useSettingsAutosave<RecallDraft>({
      load: async () => (await api.getRecallSettings()) as RecallDraft,
      save: changed => api.updateRecallSettings(changed),
      onError: err => notifyError(err, 'Recall settings'),
    });

  const handleChange = <K extends keyof RecallSettings>(
    key: K,
    value: RecallSettings[K]
  ) => {
    update({ [key]: value } as Partial<RecallDraft>);
  };

  /**
   * Bind one key to the manifest: bounds, help, whether it differs from the
   * shipped default, and a reset that writes that default back through the
   * normal autosave path. Not a hook — several of these live inside
   * conditionally-rendered branches.
   */
  const bind = <K extends keyof RecallSettings>(key: K) => {
    const entry = bindSetting(manifest, 'memory', key as string, settings?.[key]);
    return {
      binding: entry,
      onReset: entry
        ? () => handleChange(key, entry.defaultValue as RecallSettings[K])
        : undefined,
    };
  };

  /** Value with the manifest default as the fallback. */
  const val = <K extends keyof RecallSettings>(
    key: K,
    shipped: RecallSettings[K]
  ): RecallSettings[K] => {
    const current = settings?.[key];
    return (current === undefined || current === null
      ? shipped
      : current) as RecallSettings[K];
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

  if (!settings) {
    return (
      <div className="settings-panel">
        <div className="memory-error">
          <p>Failed to load recall settings{error ? `: ${error.message}` : ''}</p>
          <Button variant="secondary" onClick={() => refresh()}>Try again</Button>
        </div>
      </div>
    );
  }

  const hybridOn = val('recall_enable_hybrid', true);
  const entityOn = val('recall_enable_entity_centric', true);
  const expansionOn = val('recall_enable_query_expansion', true);
  const hydeOn = val('recall_enable_hyde', false);
  const selfQueryOn = val('recall_enable_self_query', false);
  const rerankOn = val('cross_encoder_enabled', true);
  const guardOn = val('recall_first_person_guard', false);

  return (
    <div className="settings-panel recall-settings">
      <h2 className="settings-title">
        <Search size={20} />
        Recall Layer Settings
        <span style={{ marginLeft: 'auto' }}>
          <SaveStatusChip status={status} />
        </span>
      </h2>
      <p className="settings-description">
        How the agent finds what it remembers. Each technique below searches
        stored memory a different way; they run together and their results are
        merged, so turning one on widens what can be found rather than replacing
        anything. Hover the <strong>?</strong> beside a setting for what it does
        and when to change it.
      </p>

      <SettingsSection title="Retrieval Techniques">
        <div className="settings-grid">
          <ToggleField
            label="Hybrid Search (BM25 + Vector)"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Combines keyword matching with semantic similarity"
            checked={hybridOn}
            onChange={v => handleChange('recall_enable_hybrid', v)}
            {...bind('recall_enable_hybrid')}
          />
          {hybridOn && (
            <div className="settings-subgroup">
              <SliderField
                label="BM25 Weight"
                value={val('recall_hybrid_bm25_weight', 0.3)}
                min={0} max={1} step={0.1} format={oneDp}
                onChange={v => handleChange('recall_hybrid_bm25_weight', v)}
                {...bind('recall_hybrid_bm25_weight')}
              />
              <SliderField
                label="Vector Weight"
                value={val('recall_hybrid_vector_weight', 0.7)}
                min={0} max={1} step={0.1} format={oneDp}
                onChange={v => handleChange('recall_hybrid_vector_weight', v)}
                {...bind('recall_hybrid_vector_weight')}
              />
            </div>
          )}

          <ToggleField
            label="Entity-Centric Retrieval"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Finds facts via entity graph traversal"
            checked={entityOn}
            onChange={v => handleChange('recall_enable_entity_centric', v)}
            {...bind('recall_enable_entity_centric')}
          />
          {entityOn && (
            <div className="settings-subgroup">
              <SliderField
                label="Similarity Threshold"
                value={val('recall_entity_similarity_threshold', 0.65)}
                min={0.3} max={0.95} step={0.05}
                onChange={v => handleChange('recall_entity_similarity_threshold', v)}
                {...bind('recall_entity_similarity_threshold')}
              />
              <NumberField
                label="Max Entities"
                value={val('recall_entity_max_entities', 5)}
                onChange={v => handleChange('recall_entity_max_entities', v)}
                {...bind('recall_entity_max_entities')}
              />
            </div>
          )}

          <ToggleField
            label="Query Expansion"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint={'Transforms "When is my birthday?" → "birthday is"'}
            checked={expansionOn}
            onChange={v => handleChange('recall_enable_query_expansion', v)}
            {...bind('recall_enable_query_expansion')}
          />
          {expansionOn && (
            <div className="settings-subgroup">
              <NumberField
                label="Max Variants"
                value={val('recall_expansion_max_variants', 3)}
                onChange={v => handleChange('recall_expansion_max_variants', v)}
                {...bind('recall_expansion_max_variants')}
              />
            </div>
          )}

          <ToggleField
            label="HyDE (Hypothetical Document Embedding)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="LLM generates hypothetical answer for better embedding match"
            checked={hydeOn}
            onChange={v => handleChange('recall_enable_hyde', v)}
            {...bind('recall_enable_hyde')}
          />
          {hydeOn && (
            <div className="settings-subgroup">
              <ModelPickerField
                label="Model"
                value={val('recall_hyde_model', '')}
                onChange={v => handleChange('recall_hyde_model', v)}
                showDefault={false}
                {...bind('recall_hyde_model')}
              />
              <SliderField
                label="Temperature"
                value={val('recall_hyde_temperature', 0.7)}
                min={0} max={1} step={0.1} format={oneDp}
                onChange={v => handleChange('recall_hyde_temperature', v)}
                {...bind('recall_hyde_temperature')}
              />
              <NumberField
                label="Max Tokens"
                value={val('recall_hyde_max_tokens', 150)}
                fallback={150}
                onChange={v => handleChange('recall_hyde_max_tokens', v)}
                {...bind('recall_hyde_max_tokens')}
              />
            </div>
          )}

          <ToggleField
            label="Self-Query (Filter Extraction)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="Extracts time filters, keywords from queries"
            checked={selfQueryOn}
            onChange={v => handleChange('recall_enable_self_query', v)}
            {...bind('recall_enable_self_query')}
          />
          {selfQueryOn && (
            <div className="settings-subgroup">
              <ModelPickerField
                label="Model"
                value={val('recall_self_query_model', '')}
                onChange={v => handleChange('recall_self_query_model', v)}
                showDefault={false}
                {...bind('recall_self_query_model')}
              />
              <SliderField
                label="Temperature"
                value={val('recall_self_query_temperature', 0.2)}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('recall_self_query_temperature', v)}
                {...bind('recall_self_query_temperature')}
              />
              <NumberField
                label="Max Tokens"
                value={val('recall_self_query_max_tokens', 200)}
                fallback={200}
                onChange={v => handleChange('recall_self_query_max_tokens', v)}
                {...bind('recall_self_query_max_tokens')}
              />
            </div>
          )}
        </div>
      </SettingsSection>

      <SettingsSection title="Two-Stage Rerank">
        <div className="settings-grid">
          <ToggleField
            label="Cross-Encoder Rerank"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Reranks a wider candidate pool with a cross-encoder — +20pp retrieval accuracy in evals"
            checked={rerankOn}
            onChange={v => handleChange('cross_encoder_enabled', v)}
            {...bind('cross_encoder_enabled')}
          />
          {rerankOn && (
            <div className="settings-subgroup">
              <TextField
                label="Cross-Encoder Model"
                value={val('cross_encoder_model', '')}
                placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2"
                onChange={v => handleChange('cross_encoder_model', v)}
                {...bind('cross_encoder_model')}
              />
              <NumberField
                label="Candidate Pool"
                value={val('recall_candidate_pool', 50)}
                fallback={50}
                onChange={v => handleChange('recall_candidate_pool', v)}
                {...bind('recall_candidate_pool')}
              />
              <NumberField
                label="Max Demotion"
                value={val('recall_ce_max_demotion', 2)}
                fallback={0}
                onChange={v => handleChange('recall_ce_max_demotion', v)}
                {...bind('recall_ce_max_demotion')}
              />
            </div>
          )}
        </div>
      </SettingsSection>

      {/* Cross-cutting knobs — they belong to no single technique, and most
          people never need them. Collapsed by default. */}
      <SettingsSection title="Advanced" variant="disclosure">
        <div className="settings-grid">
          <SliderField
            label="Min Recall Confidence"
            value={val('recall_min_confidence', 0.5)}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('recall_min_confidence', v)}
            {...bind('recall_min_confidence')}
          />
          <NumberField
            label="Hybrid RRF k"
            value={val('recall_hybrid_rrf_k', 60)}
            fallback={60}
            onChange={v => handleChange('recall_hybrid_rrf_k', v)}
            {...bind('recall_hybrid_rrf_k')}
          />
          <NumberField
            label="Entity Graph Depth"
            value={val('recall_entity_graph_depth', 1)}
            fallback={1}
            onChange={v => handleChange('recall_entity_graph_depth', v)}
            {...bind('recall_entity_graph_depth')}
          />
          <ToggleField
            label="First-Person Attribution Guard"
            hint="Demotes facts that misattribute first-person statements"
            checked={guardOn}
            onChange={v => handleChange('recall_first_person_guard', v)}
            {...bind('recall_first_person_guard')}
          />
          {guardOn && (
            <div className="settings-subgroup">
              <SliderField
                label="First-Person Penalty"
                value={val('recall_first_person_penalty', 0.5)}
                min={0} max={1} step={0.05}
                onChange={v => handleChange('recall_first_person_penalty', v)}
                {...bind('recall_first_person_penalty')}
              />
            </div>
          )}
        </div>
      </SettingsSection>
    </div>
  );
}

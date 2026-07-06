import { RefreshCw, Search } from 'lucide-react';
import { useSettingsAutosave } from '../../lib/hooks';
import { RecallSettings, api } from '../../lib/api';
import { ModelPickerField } from '../common/ModelPickerField';
import { useNotify } from '../../contexts/NotificationContext';
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
  const { settings, loading, error, status, update } = useSettingsAutosave<RecallDraft>({
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
          Failed to load recall settings{error ? `: ${error.message}` : ''}
        </div>
      </div>
    );
  }

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
            checked={settings.recall_enable_hybrid ?? true}
            onChange={v => handleChange('recall_enable_hybrid', v)}
          />
          <ToggleField
            label="Entity-Centric Retrieval"
            title="Traverse entity relationships to find linked facts"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="Finds facts via entity graph traversal"
            checked={settings.recall_enable_entity_centric ?? true}
            onChange={v => handleChange('recall_enable_entity_centric', v)}
          />
          <ToggleField
            label="Query Expansion"
            title="Transform questions to statement form for better matching"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint={'Transforms "When is my birthday?" → "birthday is"'}
            checked={settings.recall_enable_query_expansion ?? true}
            onChange={v => handleChange('recall_enable_query_expansion', v)}
          />
          <ToggleField
            label="HyDE (Hypothetical Document Embedding)"
            title="Generate hypothetical answer and search with that embedding (requires LLM)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="LLM generates hypothetical answer for better embedding match"
            checked={settings.recall_enable_hyde ?? false}
            onChange={v => handleChange('recall_enable_hyde', v)}
          />
          <ToggleField
            label="Self-Query (Filter Extraction)"
            title="LLM extracts structured filters from natural language (requires LLM)"
            badge={{ text: 'LLM Required', variant: 'warning' }}
            hint="Extracts time filters, keywords from queries"
            checked={settings.recall_enable_self_query ?? false}
            onChange={v => handleChange('recall_enable_self_query', v)}
          />
        </div>
      </SettingsSection>

      <SettingsSection title="Two-Stage Rerank">
        <div className="settings-grid">
          <ToggleField
            label="Cross-Encoder Rerank"
            title="Retrieve a wide candidate pool, then rerank it with a cross-encoder before returning results"
            badge={{ text: 'Recommended', variant: 'success' }}
            hint="reranks a wider candidate pool with a cross-encoder — +20pp retrieval accuracy in evals"
            checked={settings.cross_encoder_enabled ?? true}
            onChange={v => handleChange('cross_encoder_enabled', v)}
          />
          {(settings.cross_encoder_enabled ?? true) && (
            <>
              <TextField
                label="Cross-Encoder Model"
                value={settings.cross_encoder_model ?? ''}
                placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2"
                hint="Hugging Face cross-encoder model id"
                onChange={v => handleChange('cross_encoder_model', v)}
              />
              <NumberField
                label="Candidate Pool"
                value={settings.recall_candidate_pool ?? 50}
                min={10} max={200} fallback={50}
                hint="How many first-stage candidates the reranker scores"
                onChange={v => handleChange('recall_candidate_pool', v)}
              />
              <NumberField
                label="Max Demotion"
                value={settings.recall_ce_max_demotion ?? 2}
                min={0} max={20}
                hint="0 = pure cross-encoder order"
                onChange={v => handleChange('recall_ce_max_demotion', v)}
              />
            </>
          )}
        </div>
      </SettingsSection>

      {settings.recall_enable_hybrid && (
        <SettingsSection title="Hybrid Search Settings">
          <div className="settings-grid">
            <SliderField
              label="BM25 Weight"
              value={settings.recall_hybrid_bm25_weight ?? 0.3}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hybrid_bm25_weight', v)}
            />
            <SliderField
              label="Vector Weight"
              value={settings.recall_hybrid_vector_weight ?? 0.7}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hybrid_vector_weight', v)}
            />
          </div>
        </SettingsSection>
      )}

      {settings.recall_enable_entity_centric && (
        <SettingsSection title="Entity-Centric Settings">
          <div className="settings-grid">
            <SliderField
              label="Similarity Threshold"
              value={settings.recall_entity_similarity_threshold ?? 0.65}
              min={0.3} max={0.95} step={0.05}
              onChange={v => handleChange('recall_entity_similarity_threshold', v)}
            />
            <NumberField
              label="Max Entities"
              value={settings.recall_entity_max_entities ?? 5}
              min={1} max={20}
              onChange={v => handleChange('recall_entity_max_entities', v)}
            />
          </div>
        </SettingsSection>
      )}

      {settings.recall_enable_query_expansion && (
        <SettingsSection title="Query Expansion Settings">
          <div className="settings-grid">
            <NumberField
              label="Max Variants"
              value={settings.recall_expansion_max_variants ?? 3}
              min={1} max={10}
              onChange={v => handleChange('recall_expansion_max_variants', v)}
            />
          </div>
        </SettingsSection>
      )}

      {settings.recall_enable_hyde && (
        <SettingsSection title="HyDE Settings">
          <div className="settings-grid">
            <div className="setting-row">
              <ModelPickerField
                label="Model"
                value={settings.recall_hyde_model ?? ''}
                onChange={v => handleChange('recall_hyde_model', v)}
                showDefault={false}
              />
            </div>
            <SliderField
              label="Temperature"
              value={settings.recall_hyde_temperature ?? 0.7}
              min={0} max={1} step={0.1} format={oneDp}
              onChange={v => handleChange('recall_hyde_temperature', v)}
            />
          </div>
        </SettingsSection>
      )}

      {settings.recall_enable_self_query && (
        <SettingsSection title="Self-Query Settings">
          <div className="settings-grid">
            <div className="setting-row">
              <ModelPickerField
                label="Model"
                value={settings.recall_self_query_model ?? ''}
                onChange={v => handleChange('recall_self_query_model', v)}
                showDefault={false}
              />
            </div>
          </div>
        </SettingsSection>
      )}

      <SettingsSection title="Advanced">
        <div className="settings-grid">
          <SliderField
            label="Min Recall Confidence"
            value={settings.recall_min_confidence ?? 0.5}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('recall_min_confidence', v)}
          />
          <NumberField
            label="Hybrid RRF k"
            value={settings.recall_hybrid_rrf_k ?? 60}
            min={1} max={200} fallback={60}
            hint="Reciprocal Rank Fusion constant (standard: 60)"
            onChange={v => handleChange('recall_hybrid_rrf_k', v)}
          />
          <NumberField
            label="Entity Graph Depth"
            value={settings.recall_entity_graph_depth ?? 1}
            min={1} max={5} fallback={1}
            hint="Relationship hops for entity-centric traversal"
            onChange={v => handleChange('recall_entity_graph_depth', v)}
          />
          <NumberField
            label="HyDE Max Tokens"
            value={settings.recall_hyde_max_tokens ?? 150}
            min={50} max={2000} fallback={150}
            onChange={v => handleChange('recall_hyde_max_tokens', v)}
          />
          <SliderField
            label="Self-Query Temperature"
            value={settings.recall_self_query_temperature ?? 0.2}
            min={0} max={1} step={0.05}
            onChange={v => handleChange('recall_self_query_temperature', v)}
          />
          <NumberField
            label="Self-Query Max Tokens"
            value={settings.recall_self_query_max_tokens ?? 200}
            min={50} max={2000} fallback={200}
            onChange={v => handleChange('recall_self_query_max_tokens', v)}
          />
          <ToggleField
            label="First-Person Attribution Guard"
            title="Penalize recall of facts whose first-person attribution doesn't match the querying speaker"
            badge={{ text: 'Experimental' }}
            hint="Demotes facts that misattribute first-person statements"
            checked={settings.recall_first_person_guard ?? false}
            onChange={v => handleChange('recall_first_person_guard', v)}
          />
          {settings.recall_first_person_guard && (
            <SliderField
              label="First-Person Penalty"
              value={settings.recall_first_person_penalty ?? 0.5}
              min={0} max={1} step={0.05}
              onChange={v => handleChange('recall_first_person_penalty', v)}
            />
          )}
        </div>
      </SettingsSection>
    </div>
  );
}

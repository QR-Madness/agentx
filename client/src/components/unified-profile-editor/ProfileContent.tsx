import { useEffect, useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import * as Tabs from '@radix-ui/react-tabs';
import * as Accordion from '@radix-ui/react-accordion';
import {
  Brain,
  ChevronRight,
  ChevronDown,
  Cpu,
  Eye,
  Library,
  MessageSquare,
  Save,
  Settings2,
  Trash2,
  Wrench,
  X,
  Zap,
  Share2,
  AlertCircle,
} from 'lucide-react';
import { type AgentProfile, type ModelInfo } from '../../lib/api';
import { AVATAR_OPTIONS } from '../../lib/avatars';
import { fetchModelsOnce } from '../common/ModelSelector';
import { ModelPickerModal } from '../common/ModelPickerModal';
import { PromptLibraryPanel } from './PromptLibraryPanel';
import { ToolAccessSection } from './ToolAccessSection';
import { useProfileEditorState, REASONING_OPTIONS } from './hooks/useProfileEditorState';

const panelVariants = {
  enter: (dir: number) => ({
    x: dir > 0 ? '100%' : '-100%',
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
    transition: { type: 'spring' as const, damping: 28, stiffness: 320 },
  },
  exit: (dir: number) => ({
    x: dir > 0 ? '-100%' : '100%',
    opacity: 0,
    transition: { duration: 0.2 },
  }),
};

/** A collapsible section card (Radix Accordion item) styled like the editor cards. */
function AccordionCard({
  value,
  icon,
  title,
  hint,
  children,
}: {
  value: string;
  icon: ReactNode;
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <Accordion.Item value={value} className="profile-section-card profile-accordion-item">
      <Accordion.Header>
        <Accordion.Trigger className="profile-accordion-trigger">
          <span className="profile-section-header profile-accordion-title">
            {icon}
            <span>{title}</span>
            {hint && <span className="profile-accordion-hint">{hint}</span>}
          </span>
          <ChevronDown size={15} className="profile-accordion-chevron" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="profile-accordion-content">
        <div className="profile-accordion-content-inner">{children}</div>
      </Accordion.Content>
    </Accordion.Item>
  );
}

interface ProfileContentProps {
  profile: AgentProfile | null;
  onSaved: (profileId: string) => void;
  onDeleted: () => void;
  onCancel: () => void;
}

export function ProfileContent({
  profile,
  onSaved,
  onDeleted,
  onCancel,
}: ProfileContentProps) {
  const isEditing = profile !== null;

  const {
    name, setName,
    avatar, setAvatar,
    description, setDescription,
    defaultModel, setDefaultModel,
    temperature, setTemperature,
    reasoningStrategy, setReasoningStrategy,
    systemPrompt, setSystemPrompt,
    enableMemory, setEnableMemory,
    memoryChannel, setMemoryChannel,
    enableTools, setEnableTools,
    allowedTools, setAllowedTools,
    blockedTools, setBlockedTools,
    availableForDelegation, setAvailableForDelegation,
    setBaseTemplateId,
    baseTemplate,
    systemPromptRef,
    insertAtCursor,
    saving,
    deleting,
    error,
    handleSubmit,
    handleDelete,
  } = useProfileEditorState(profile);

  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryMode, setLibraryMode] = useState<'insert' | 'select'>('insert');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [modelCatalog, setModelCatalog] = useState<ModelInfo[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetchModelsOnce().then(m => { if (!cancelled) setModelCatalog(m); });
    return () => { cancelled = true; };
  }, []);

  const selectedModel = defaultModel ? modelCatalog.find(m => m.id === defaultModel) : undefined;
  const selectedCtx = selectedModel?.context_length ?? selectedModel?.context_window;
  const selectedLabel = (() => {
    if (!defaultModel) return 'System default';
    if (selectedModel) return selectedModel.name;
    const parts = defaultModel.split(':');
    return parts.length > 1 ? parts.slice(1).join(':') : defaultModel;
  })();
  const selectedProviderLabel = selectedModel?.provider ?? (defaultModel.includes(':') ? defaultModel.split(':')[0] : '');
  // dir: 1 = library slides in from right, -1 = form slides back in from left
  const [panelDir, setPanelDir] = useState(1);

  const promptPreview = systemPrompt.trim()
    ? `${systemPrompt.trim().slice(0, 80)}${systemPrompt.trim().length > 80 ? '…' : ''}`
    : 'Default instructions';

  const openLibrary = (mode: 'insert' | 'select') => {
    setLibraryMode(mode);
    setPanelDir(1);
    setLibraryOpen(true);
  };

  const closeLibrary = () => {
    setPanelDir(-1);
    setLibraryOpen(false);
  };

  const handleSave = () => {
    handleSubmit(isEditing, profile?.id, (saved) => {
      onSaved(saved.id);
    });
  };

  const handleDeleteClick = () => {
    if (!profile) return;
    if (!window.confirm(`Delete "${profile.name}"? This cannot be undone.`)) return;
    handleDelete(profile.id, onDeleted);
  };

  return (
    <div className="profile-content-outer">
      <AnimatePresence mode="wait" custom={panelDir}>
        {libraryOpen ? (
          <motion.div
            key="library"
            className="profile-panel-fill"
            custom={panelDir}
            variants={panelVariants}
            initial="enter"
            animate="center"
            exit="exit"
          >
            <PromptLibraryPanel
              mode={libraryMode}
              onBack={closeLibrary}
              onInsert={insertAtCursor}
              onSelectTemplate={(templateId, content) => {
                setBaseTemplateId(templateId);
                if (!systemPrompt.trim()) setSystemPrompt(content);
              }}
            />
          </motion.div>
        ) : (
          <motion.div
            key="form"
            className="profile-panel-fill profile-form-scroll"
            custom={panelDir}
            variants={panelVariants}
            initial="enter"
            animate="center"
            exit="exit"
          >
            {error && (
              <div className="profile-error-banner">
                <AlertCircle size={15} />
                <span>{error}</span>
              </div>
            )}

            <Tabs.Root defaultValue="core" className="profile-tabs-root">
              <Tabs.List className="profile-tabs-list" aria-label="Profile settings">
                <Tabs.Trigger value="core" className="profile-tab-trigger">Core</Tabs.Trigger>
                <Tabs.Trigger value="tools" className="profile-tab-trigger">Tools</Tabs.Trigger>
                <Tabs.Trigger value="advanced" className="profile-tab-trigger">Advanced</Tabs.Trigger>
              </Tabs.List>

              {/* ── Core ── */}
              <Tabs.Content value="core" className="profile-tab-content">
                <Accordion.Root
                  type="multiple"
                  defaultValue={['identity', 'model', 'delegation']}
                  className="profile-accordion-root"
                >
                  <AccordionCard value="identity" icon={<Settings2 size={14} />} title="Identity">
                    <div className="profile-form-group">
                      <label>Name</label>
                      <input
                        type="text"
                        value={name}
                        onChange={e => setName(e.target.value)}
                        placeholder="Enter profile name..."
                      />
                    </div>

                    <div className="profile-form-group">
                      <label>Avatar</label>
                      <div className="profile-avatar-picker">
                        {AVATAR_OPTIONS.map(opt => {
                          const Icon = opt.icon;
                          return (
                            <button
                              key={opt.id}
                              type="button"
                              className={`profile-avatar-option ${avatar === opt.id ? 'selected' : ''}`}
                              onClick={() => setAvatar(opt.id)}
                              title={opt.label}
                            >
                              <Icon size={18} />
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="profile-form-group">
                      <label>Description</label>
                      <textarea
                        value={description}
                        onChange={e => setDescription(e.target.value)}
                        placeholder="Describe this profile's purpose..."
                        rows={2}
                      />
                      <span className="profile-form-hint">
                        Shown to other agents when deciding whether to delegate here.
                      </span>
                    </div>
                  </AccordionCard>

                  <AccordionCard value="model" icon={<Cpu size={14} />} title="Model &amp; Generation">
                    <div className="profile-model-selector-wrap">
                      <button
                        type="button"
                        className="profile-model-trigger"
                        onClick={() => setPickerOpen(true)}
                      >
                        <div className="profile-model-trigger-main">
                          <span className="profile-model-trigger-label">Model</span>
                          <span className="profile-model-trigger-name">{selectedLabel}</span>
                          {selectedProviderLabel && (
                            <span className="profile-model-trigger-provider">{selectedProviderLabel}</span>
                          )}
                        </div>
                        <div className="profile-model-trigger-meta">
                          {selectedCtx && (
                            <span className="profile-model-trigger-badge">
                              {selectedCtx >= 1000 ? `${Math.round(selectedCtx / 1000)}k ctx` : `${selectedCtx} ctx`}
                            </span>
                          )}
                          {selectedModel?.supports_tools && (
                            <span className="profile-model-trigger-cap" title="Tools"><Wrench size={12} /></span>
                          )}
                          {selectedModel?.supports_vision && (
                            <span className="profile-model-trigger-cap" title="Vision"><Eye size={12} /></span>
                          )}
                          <ChevronRight size={14} className="profile-model-trigger-chev" />
                        </div>
                      </button>
                    </div>
                    <ModelPickerModal
                      isOpen={pickerOpen}
                      onClose={() => setPickerOpen(false)}
                      value={defaultModel}
                      onChange={setDefaultModel}
                      showDefault
                    />

                    <div className="profile-form-group">
                      <label>Temperature: {temperature.toFixed(1)}</label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={e => setTemperature(parseFloat(e.target.value))}
                        className="profile-slider"
                      />
                      <div className="profile-slider-labels">
                        <span>Focused</span>
                        <span>Creative</span>
                      </div>
                    </div>

                    <div className="profile-form-group">
                      <label>Reasoning Strategy</label>
                      <select
                        value={reasoningStrategy}
                        onChange={e => setReasoningStrategy(e.target.value as typeof reasoningStrategy)}
                      >
                        {REASONING_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label} — {opt.description}
                          </option>
                        ))}
                      </select>
                    </div>
                  </AccordionCard>

                  {/* System Prompt — collapsed by default so the long text doesn't
                      dominate the scroll; preview shown on the closed header. */}
                  <AccordionCard
                    value="prompt"
                    icon={<MessageSquare size={14} />}
                    title="System Prompt"
                    hint={promptPreview}
                  >
                    <div className="profile-form-group">
                      <label>Base Template</label>
                      <div className="profile-base-template-row">
                        {baseTemplate ? (
                          <>
                            <span className="profile-template-name">{baseTemplate.name}</span>
                            {baseTemplate.hasModifications && (
                              <span className="profile-modified-badge">Modified</span>
                            )}
                            <button
                              type="button"
                              className="profile-btn-secondary profile-btn-sm"
                              onClick={() => openLibrary('select')}
                            >
                              Change
                            </button>
                            <button
                              type="button"
                              className="profile-btn-ghost profile-btn-sm"
                              onClick={() => setBaseTemplateId('')}
                              title="Remove base template"
                            >
                              <X size={13} />
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className="profile-btn-secondary"
                            onClick={() => openLibrary('select')}
                          >
                            <Library size={14} />
                            Select Base Template
                          </button>
                        )}
                      </div>
                      <span className="profile-form-hint">Optional template for base agent instructions</span>
                    </div>

                    <div className="profile-form-group">
                      <div className="profile-prompt-toolbar">
                        <label>Agent Instructions</label>
                        <button
                          type="button"
                          className="profile-library-btn"
                          onClick={() => openLibrary('insert')}
                        >
                          <Library size={13} />
                          Insert from Library
                        </button>
                      </div>
                      <textarea
                        ref={systemPromptRef}
                        value={systemPrompt}
                        onChange={e => setSystemPrompt(e.target.value)}
                        placeholder="Custom instructions for this agent… (leave empty to use defaults)"
                        rows={6}
                        className="profile-system-prompt"
                      />
                      <span className="profile-form-hint">Agent-specific instructions prepended to conversations</span>
                    </div>
                  </AccordionCard>

                  {/* Delegation — Track D */}
                  <AccordionCard value="delegation" icon={<Share2 size={14} />} title="Delegation">
                    <label className="profile-toggle-row">
                      <span className="profile-toggle-label">
                        <Share2 size={15} />
                        Available for delegation
                      </span>
                      <div className="profile-toggle-right">
                        <input
                          type="checkbox"
                          checked={availableForDelegation}
                          onChange={e => setAvailableForDelegation(e.target.checked)}
                          className="profile-toggle-input"
                        />
                        <span className={`profile-toggle-switch ${availableForDelegation ? 'active' : ''}`} />
                      </div>
                    </label>
                    <span className="profile-form-hint">
                      When on, other agents can hand subtasks to this profile via the
                      <code> delegate_to </code> tool (requires ad-hoc delegation enabled
                      in Settings → Multi-Agent).
                    </span>
                  </AccordionCard>
                </Accordion.Root>
              </Tabs.Content>

              {/* ── Tools ── */}
              <Tabs.Content value="tools" className="profile-tab-content">
                <div className="profile-section-card">
                  <label className="profile-toggle-row">
                    <span className="profile-toggle-label">
                      <Zap size={15} />
                      Enable Tools
                    </span>
                    <div className="profile-toggle-right">
                      <input
                        type="checkbox"
                        checked={enableTools}
                        onChange={e => setEnableTools(e.target.checked)}
                        className="profile-toggle-input"
                      />
                      <span className={`profile-toggle-switch ${enableTools ? 'active' : ''}`} />
                    </div>
                  </label>

                  {enableTools ? (
                    <ToolAccessSection
                      allowedTools={allowedTools}
                      setAllowedTools={setAllowedTools}
                      blockedTools={blockedTools}
                      setBlockedTools={setBlockedTools}
                    />
                  ) : (
                    <span className="profile-form-hint">Tools are disabled for this agent.</span>
                  )}
                </div>
              </Tabs.Content>

              {/* ── Advanced ── */}
              <Tabs.Content value="advanced" className="profile-tab-content">
                <Accordion.Root type="multiple" defaultValue={['memory']} className="profile-accordion-root">
                  <AccordionCard value="memory" icon={<Brain size={14} />} title="Memory">
                    <label className="profile-toggle-row">
                      <span className="profile-toggle-label">
                        <Brain size={15} />
                        Enable Memory
                      </span>
                      <div className="profile-toggle-right">
                        <input
                          type="checkbox"
                          checked={enableMemory}
                          onChange={e => setEnableMemory(e.target.checked)}
                          className="profile-toggle-input"
                        />
                        <span className={`profile-toggle-switch ${enableMemory ? 'active' : ''}`} />
                      </div>
                    </label>

                    {enableMemory && (
                      <div className="profile-form-group profile-nested">
                        <label>Memory Channel</label>
                        <input
                          type="text"
                          value={memoryChannel}
                          onChange={e => setMemoryChannel(e.target.value)}
                          placeholder="_global"
                        />
                        <span className="profile-form-hint">Isolate memories to a specific channel</span>
                      </div>
                    )}
                  </AccordionCard>
                </Accordion.Root>
              </Tabs.Content>
            </Tabs.Root>

            {/* Sticky footer */}
            <div className="profile-footer">
              {isEditing && !profile?.isDefault && (
                <button
                  type="button"
                  className="profile-btn-danger"
                  onClick={handleDeleteClick}
                  disabled={deleting || saving}
                >
                  <Trash2 size={15} />
                  {deleting ? 'Deleting…' : 'Delete'}
                </button>
              )}
              <div className="profile-footer-right">
                <button
                  type="button"
                  className="profile-btn-secondary"
                  onClick={onCancel}
                  disabled={saving || deleting}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="profile-btn-primary"
                  onClick={handleSave}
                  disabled={saving || deleting}
                >
                  <Save size={15} />
                  {saving ? 'Saving…' : isEditing ? 'Save Changes' : 'Create Profile'}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

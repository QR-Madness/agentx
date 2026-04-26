import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Brain,
  Cpu,
  Library,
  MessageSquare,
  Save,
  Settings2,
  Trash2,
  X,
  Zap,
  AlertCircle,
} from 'lucide-react';
import { type AgentProfile } from '../../lib/api';
import { AVATAR_OPTIONS } from '../../lib/avatars';
import { ModelSelector } from '../common/ModelSelector';
import { PromptLibraryPanel } from './PromptLibraryPanel';
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
  // dir: 1 = library slides in from right, -1 = form slides back in from left
  const [panelDir, setPanelDir] = useState(1);

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

            {/* Identity */}
            <div className="profile-section-card">
              <div className="profile-section-header">
                <Settings2 size={14} />
                <span>Identity</span>
              </div>

              <div className="profile-form-group">
                <label>Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Enter profile name..."
                  autoFocus
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
              </div>
            </div>

            {/* Model & Generation */}
            <div className="profile-section-card">
              <div className="profile-section-header">
                <Cpu size={14} />
                <span>Model & Generation</span>
              </div>

              <div className="profile-model-selector-wrap">
                <ModelSelector
                  value={defaultModel}
                  onChange={setDefaultModel}
                  showDefault
                />
              </div>

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
            </div>

            {/* System Prompt */}
            <div className="profile-section-card">
              <div className="profile-section-header">
                <MessageSquare size={14} />
                <span>System Prompt</span>
              </div>

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
            </div>

            {/* Capabilities */}
            <div className="profile-section-card">
              <div className="profile-section-header">
                <Zap size={14} />
                <span>Capabilities</span>
              </div>

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
            </div>

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

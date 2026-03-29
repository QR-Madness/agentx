/**
 * ProfileEditorModal — Create or edit agent profiles
 * Enhanced with model selection, inline system prompt, and organized sections
 */

import { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Zap,
  Save,
  X,
  Cpu,
  MessageSquare,
  Settings2,
  RefreshCw,
  Trash2,
  Library,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { AVATAR_OPTIONS, getAvatarIcon } from '../../lib/avatars';
import { api, type AgentProfile, type AgentProfileCreate, type ReasoningStrategy, type PromptTemplate } from '../../lib/api';
import { PromptLibraryModal } from './PromptLibraryModal';
import { ModelSelector } from '../common/ModelSelector';
import './ProfileEditorModal.css';

const REASONING_OPTIONS: { value: ReasoningStrategy; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'Automatically select based on task' },
  { value: 'cot', label: 'Chain of Thought', description: 'Step-by-step reasoning' },
  { value: 'tot', label: 'Tree of Thought', description: 'Explore multiple paths' },
  { value: 'react', label: 'ReAct', description: 'Reason + Act iteratively' },
  { value: 'reflection', label: 'Reflection', description: 'Self-critique and improve' },
];

interface ProfileEditorModalProps {
  onClose: () => void;
  // Support both patterns: passing full profile or just ID
  editProfile?: AgentProfile;
  profileId?: string;
  isNew?: boolean;
}

export function ProfileEditorModal({ onClose, editProfile: editProfileProp, profileId, isNew }: ProfileEditorModalProps) {
  const { profiles, createProfile, updateProfile, deleteProfile } = useAgentProfile();

  // Resolve the profile to edit - either from prop or by finding it via ID
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [resolvedProfile, setResolvedProfile] = useState<AgentProfile | undefined>(editProfileProp);

  // Fetch profile if profileId was passed instead of the full object
  useEffect(() => {
    if (profileId && !editProfileProp && !isNew) {
      // First try to find in local profiles list
      const localProfile = profiles.find(p => p.id === profileId);
      if (localProfile) {
        setResolvedProfile(localProfile);
      } else {
        // Fetch from API if not in local cache
        setLoadingProfile(true);
        api.getAgentProfile(profileId)
          .then(({ profile }) => setResolvedProfile(profile))
          .catch(err => console.error('Failed to fetch profile:', err))
          .finally(() => setLoadingProfile(false));
      }
    }
  }, [profileId, editProfileProp, isNew, profiles]);

  const editProfile = resolvedProfile;
  const isEditing = !!editProfile && !isNew;

  // Form state
  const [name, setName] = useState('');
  const [avatar, setAvatar] = useState('sparkles');
  const [description, setDescription] = useState('');
  const [defaultModel, setDefaultModel] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [reasoningStrategy, setReasoningStrategy] = useState<ReasoningStrategy>('auto');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [enableMemory, setEnableMemory] = useState(true);
  const [memoryChannel, setMemoryChannel] = useState('_global');
  const [enableTools, setEnableTools] = useState(true);

  const [baseTemplateId, setBaseTemplateId] = useState('');
  const [baseTemplate, setBaseTemplate] = useState<PromptTemplate | null>(null);

  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Library modal state
  const [showLibrary, setShowLibrary] = useState(false);
  const [libraryMode, setLibraryMode] = useState<'insert' | 'select'>('insert');

  // Ref for textarea to handle cursor position
  const systemPromptRef = useRef<HTMLTextAreaElement>(null);

  // Fetch base template when baseTemplateId changes
  useEffect(() => {
    if (baseTemplateId) {
      api.getPromptTemplate(baseTemplateId)
        .then(({ template }) => setBaseTemplate(template))
        .catch(err => {
          console.error('Failed to fetch base template:', err);
          setBaseTemplate(null);
        });
    } else {
      setBaseTemplate(null);
    }
  }, [baseTemplateId]);

  // Initialize form when editProfile is resolved
  useEffect(() => {
    if (editProfile) {
      setName(editProfile.name);
      setAvatar(editProfile.avatar ?? 'sparkles');
      setDescription(editProfile.description ?? '');
      setDefaultModel(editProfile.defaultModel ?? '');
      setTemperature(editProfile.temperature);
      setReasoningStrategy(editProfile.reasoningStrategy);
      setSystemPrompt(editProfile.systemPrompt ?? '');
      // Use promptProfileId as baseTemplateId for backward compatibility
      setBaseTemplateId(editProfile.promptProfileId ?? '');
      setEnableMemory(editProfile.enableMemory);
      setMemoryChannel(editProfile.memoryChannel);
      setEnableTools(editProfile.enableTools);
    }
  }, [editProfile]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const profileData: AgentProfileCreate = {
        name: name.trim(),
        avatar,
        description: description.trim() || undefined,
        default_model: defaultModel || undefined,
        temperature,
        reasoning_strategy: reasoningStrategy,
        prompt_profile_id: baseTemplateId || undefined,
        system_prompt: systemPrompt.trim() || undefined,
        enable_memory: enableMemory,
        memory_channel: memoryChannel,
        enable_tools: enableTools,
      };

      if (isEditing && editProfile) {
        await updateProfile(editProfile.id, profileData);
      } else {
        await createProfile(profileData);
      }

      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!editProfile) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete "${editProfile.name}"? This action cannot be undone.`
    );
    if (!confirmed) return;

    setDeleting(true);
    setError(null);

    try {
      const success = await deleteProfile(editProfile.id);
      if (success) {
        onClose();
      } else {
        setError('Failed to delete profile');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete profile');
    } finally {
      setDeleting(false);
    }
  };

  const AvatarIcon = getAvatarIcon(avatar);

  // Insert text at cursor position in the system prompt textarea
  const insertAtCursor = (text: string) => {
    const textarea = systemPromptRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const before = systemPrompt.substring(0, start);
    const after = systemPrompt.substring(end);

    setSystemPrompt(before + text + after);

    // Restore focus and set cursor position after the inserted text
    requestAnimationFrame(() => {
      textarea.focus();
      textarea.selectionStart = textarea.selectionEnd = start + text.length;
    });
  };

  // Open library modal to insert a template snippet
  const handleInsertFromLibrary = () => {
    setLibraryMode('insert');
    setShowLibrary(true);
  };

  // Open library modal to select base template
  const handleSelectBaseTemplate = () => {
    setLibraryMode('select');
    setShowLibrary(true);
  };

  // Handle template insertion from library
  const handleLibraryInsert = (content: string) => {
    insertAtCursor(content);
  };

  // Handle base template selection from library
  const handleLibrarySelect = (templateId: string, content: string) => {
    setBaseTemplateId(templateId);
    // Optionally prepend template content to system prompt if empty
    if (!systemPrompt.trim()) {
      setSystemPrompt(content);
    }
  };

  // Show loading state while fetching profile
  if (loadingProfile) {
    return (
      <div className="profile-editor-modal">
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="avatar-preview">
              <RefreshCw size={24} className="spin" />
            </div>
            <h2>Loading Profile...</h2>
          </div>
          <button className="button-ghost close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <RefreshCw size={32} className="spin" style={{ marginBottom: '1rem' }} />
          <p>Loading profile data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-editor-modal">
      <div className="modal-header">
        <div className="modal-title-group">
          <div className="avatar-preview">
            <AvatarIcon size={24} />
          </div>
          <h2>{isEditing ? 'Edit Profile' : 'Create Profile'}</h2>
        </div>
        <button className="button-ghost close-btn" onClick={onClose}>
          <X size={20} />
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        {/* Identity Section */}
        <div className="form-section">
          <div className="section-header">
            <Settings2 size={14} />
            <span>Identity</span>
          </div>

          <div className="form-group">
            <label htmlFor="profile-name">Name</label>
            <input
              id="profile-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter profile name..."
              autoFocus
            />
          </div>

          <div className="form-group">
            <label>Avatar</label>
            <div className="avatar-picker">
              {AVATAR_OPTIONS.map(option => {
                const Icon = option.icon;
                return (
                  <button
                    key={option.id}
                    type="button"
                    className={`avatar-option ${avatar === option.id ? 'selected' : ''}`}
                    onClick={() => setAvatar(option.id)}
                    title={option.label}
                  >
                    <Icon size={20} />
                  </button>
                );
              })}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="profile-description">Description</label>
            <textarea
              id="profile-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe this profile's purpose..."
              rows={2}
            />
            <span className="form-hint">Optional description for this agent profile</span>
          </div>
        </div>

        {/* Model & Generation Section */}
        <div className="form-section">
          <div className="section-header">
            <Cpu size={14} />
            <span>Model & Generation</span>
          </div>

          {/* Model Selection */}
          <ModelSelector
            value={defaultModel}
            onChange={setDefaultModel}
            showDefault
          />

          <div className="form-group">
            <label htmlFor="temperature">
              Temperature: {temperature.toFixed(1)}
            </label>
            <input
              id="temperature"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="slider"
            />
            <div className="slider-labels">
              <span>Focused</span>
              <span>Creative</span>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="reasoning-strategy">Reasoning Strategy</label>
            <select
              id="reasoning-strategy"
              value={reasoningStrategy}
              onChange={(e) => setReasoningStrategy(e.target.value as ReasoningStrategy)}
            >
              {REASONING_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label} — {option.description}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* System Prompt Section */}
        <div className="form-section">
          <div className="section-header">
            <MessageSquare size={14} />
            <span>System Prompt</span>
          </div>

          <div className="form-group">
            <label>Base Template</label>
            <div className="base-template-picker">
              {baseTemplate ? (
                <div className="selected-template">
                  <span className="template-name">{baseTemplate.name}</span>
                  {baseTemplate.hasModifications && (
                    <span className="modified-badge">Modified</span>
                  )}
                  <button
                    type="button"
                    className="button-secondary button-small"
                    onClick={handleSelectBaseTemplate}
                  >
                    Change
                  </button>
                  <button
                    type="button"
                    className="button-ghost button-small"
                    onClick={() => setBaseTemplateId('')}
                    title="Remove base template"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="button-secondary"
                  onClick={handleSelectBaseTemplate}
                >
                  <Library size={14} />
                  Select Base Template
                </button>
              )}
            </div>
            <span className="form-hint">Optional template for base agent instructions</span>
          </div>

          <div className="form-group">
            <label htmlFor="system-prompt">Agent Instructions</label>
            <div className="prompt-toolbar">
              <button
                type="button"
                className="prompt-toolbar-btn"
                onClick={handleInsertFromLibrary}
                title="Insert snippet from prompt library"
              >
                <Library size={14} />
                <span>Insert from Library</span>
              </button>
            </div>
            <textarea
              ref={systemPromptRef}
              id="system-prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Custom instructions for this agent... (leave empty to use defaults)"
              rows={5}
              className="system-prompt-textarea"
            />
            <span className="form-hint">Agent-specific instructions prepended to conversations</span>
          </div>
        </div>

        {/* Capabilities Section */}
        <div className="form-section">
          <div className="section-header">
            <Zap size={14} />
            <span>Capabilities</span>
          </div>

          <div className="toggle-group">
            <label className="toggle-row">
              <span className="toggle-label">
                <Brain size={16} />
                Enable Memory
              </span>
              <input
                type="checkbox"
                checked={enableMemory}
                onChange={(e) => setEnableMemory(e.target.checked)}
                className="toggle-input"
              />
              <span className={`toggle-switch ${enableMemory ? 'active' : ''}`} />
            </label>

            {enableMemory && (
              <div className="form-group nested">
                <label htmlFor="memory-channel">Memory Channel</label>
                <input
                  id="memory-channel"
                  type="text"
                  value={memoryChannel}
                  onChange={(e) => setMemoryChannel(e.target.value)}
                  placeholder="_global"
                />
                <span className="form-hint">Isolate memories to a specific channel</span>
              </div>
            )}

            <label className="toggle-row">
              <span className="toggle-label">
                <Zap size={16} />
                Enable Tools
              </span>
              <input
                type="checkbox"
                checked={enableTools}
                onChange={(e) => setEnableTools(e.target.checked)}
                className="toggle-input"
              />
              <span className={`toggle-switch ${enableTools ? 'active' : ''}`} />
            </label>
          </div>
        </div>

        {/* Actions */}
        <div className="form-actions">
          {isEditing && !editProfile?.isDefault && (
            <button
              type="button"
              className="button-danger"
              onClick={handleDelete}
              disabled={deleting || saving}
            >
              <Trash2 size={16} />
              {deleting ? 'Deleting...' : 'Delete'}
            </button>
          )}
          <div className="form-actions-right">
            <button type="button" className="button-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="button-primary" disabled={saving || deleting}>
              <Save size={16} />
              {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Profile'}
            </button>
          </div>
        </div>
      </form>

      {/* Prompt Library Modal */}
      {showLibrary && (
        <div className="library-modal-overlay">
          <PromptLibraryModal
            onClose={() => setShowLibrary(false)}
            onInsert={libraryMode === 'insert' ? handleLibraryInsert : undefined}
            onSelectTemplate={libraryMode === 'select' ? handleLibrarySelect : undefined}
            mode={libraryMode}
          />
        </div>
      )}
    </div>
  );
}

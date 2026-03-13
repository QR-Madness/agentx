/**
 * ProfileEditorModal — Create or edit agent profiles
 */

import { useState, useEffect } from 'react';
import {
  User,
  Sparkles,
  Brain,
  Zap,
  Heart,
  Star,
  Moon,
  Sun,
  Cloud,
  Flame,
  Save,
  X,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import type { AgentProfile, AgentProfileCreate, ReasoningStrategy } from '../../lib/api';
import './ProfileEditorModal.css';

// Avatar icon options
const AVATAR_OPTIONS = [
  { id: 'sparkles', icon: Sparkles, label: 'Sparkles' },
  { id: 'brain', icon: Brain, label: 'Brain' },
  { id: 'zap', icon: Zap, label: 'Zap' },
  { id: 'heart', icon: Heart, label: 'Heart' },
  { id: 'star', icon: Star, label: 'Star' },
  { id: 'moon', icon: Moon, label: 'Moon' },
  { id: 'sun', icon: Sun, label: 'Sun' },
  { id: 'cloud', icon: Cloud, label: 'Cloud' },
  { id: 'flame', icon: Flame, label: 'Flame' },
  { id: 'user', icon: User, label: 'User' },
];

const REASONING_OPTIONS: { value: ReasoningStrategy; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'Automatically select based on task' },
  { value: 'cot', label: 'Chain of Thought', description: 'Step-by-step reasoning' },
  { value: 'tot', label: 'Tree of Thought', description: 'Explore multiple paths' },
  { value: 'react', label: 'ReAct', description: 'Reason + Act iteratively' },
  { value: 'reflection', label: 'Reflection', description: 'Self-critique and improve' },
];

interface ProfileEditorModalProps {
  onClose: () => void;
  editProfile?: AgentProfile;
}

export function ProfileEditorModal({ onClose, editProfile }: ProfileEditorModalProps) {
  const { createProfile, updateProfile } = useAgentProfile();
  const isEditing = !!editProfile;

  // Form state
  const [name, setName] = useState(editProfile?.name ?? '');
  const [avatar, setAvatar] = useState(editProfile?.avatar ?? 'sparkles');
  const [description, setDescription] = useState(editProfile?.description ?? '');
  const [temperature, setTemperature] = useState(editProfile?.temperature ?? 0.7);
  const [reasoningStrategy, setReasoningStrategy] = useState<ReasoningStrategy>(
    editProfile?.reasoningStrategy ?? 'auto'
  );
  const [enableMemory, setEnableMemory] = useState(editProfile?.enableMemory ?? true);
  const [memoryChannel, setMemoryChannel] = useState(editProfile?.memoryChannel ?? '_global');
  const [enableTools, setEnableTools] = useState(editProfile?.enableTools ?? true);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when editProfile changes
  useEffect(() => {
    if (editProfile) {
      setName(editProfile.name);
      setAvatar(editProfile.avatar ?? 'sparkles');
      setDescription(editProfile.description ?? '');
      setTemperature(editProfile.temperature);
      setReasoningStrategy(editProfile.reasoningStrategy);
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
        temperature,
        reasoning_strategy: reasoningStrategy,
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

  const getAvatarIcon = (avatarId: string) => {
    const option = AVATAR_OPTIONS.find(o => o.id === avatarId);
    return option?.icon ?? Sparkles;
  };

  const AvatarIcon = getAvatarIcon(avatar);

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

        {/* Name & Avatar */}
        <div className="form-section">
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
          </div>
        </div>

        {/* Model Settings */}
        <div className="form-section">
          <h3 className="section-label">Model Settings</h3>

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

        {/* Behavior Toggles */}
        <div className="form-section">
          <h3 className="section-label">Behavior</h3>

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
              <span className="toggle-switch" />
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
              <span className="toggle-switch" />
            </label>
          </div>
        </div>

        {/* Actions */}
        <div className="form-actions">
          <button type="button" className="button-secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="button-primary" disabled={saving}>
            <Save size={16} />
            {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Profile'}
          </button>
        </div>
      </form>
    </div>
  );
}

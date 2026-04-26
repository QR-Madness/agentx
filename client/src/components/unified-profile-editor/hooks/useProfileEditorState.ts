import { useState, useEffect, useRef } from 'react';
import {
  api,
  type AgentProfile,
  type AgentProfileCreate,
  type ReasoningStrategy,
  type PromptTemplate,
} from '../../../lib/api';
import { useAgentProfile } from '../../../contexts/AgentProfileContext';

export const REASONING_OPTIONS: { value: ReasoningStrategy; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'Automatically select based on task' },
  { value: 'cot', label: 'Chain of Thought', description: 'Step-by-step reasoning' },
  { value: 'tot', label: 'Tree of Thought', description: 'Explore multiple paths' },
  { value: 'react', label: 'ReAct', description: 'Reason + Act iteratively' },
  { value: 'reflection', label: 'Reflection', description: 'Self-critique and improve' },
];

export function useProfileEditorState(profile: AgentProfile | null) {
  const { createProfile, updateProfile, deleteProfile } = useAgentProfile();

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
  const systemPromptRef = useRef<HTMLTextAreaElement>(null);

  // Initialize (or reset) form whenever the profile changes
  useEffect(() => {
    if (profile) {
      setName(profile.name);
      setAvatar(profile.avatar ?? 'sparkles');
      setDescription(profile.description ?? '');
      setDefaultModel(profile.defaultModel ?? '');
      setTemperature(profile.temperature);
      setReasoningStrategy(profile.reasoningStrategy);
      setSystemPrompt(profile.systemPrompt ?? '');
      setBaseTemplateId(profile.promptProfileId ?? '');
      setEnableMemory(profile.enableMemory);
      setMemoryChannel(profile.memoryChannel);
      setEnableTools(profile.enableTools);
    } else {
      setName('');
      setAvatar('sparkles');
      setDescription('');
      setDefaultModel('');
      setTemperature(0.7);
      setReasoningStrategy('auto');
      setSystemPrompt('');
      setBaseTemplateId('');
      setEnableMemory(true);
      setMemoryChannel('_global');
      setEnableTools(true);
    }
    setError(null);
  }, [profile]);

  // Resolve base template name when id changes
  useEffect(() => {
    if (baseTemplateId) {
      api
        .getPromptTemplate(baseTemplateId)
        .then(({ template }) => setBaseTemplate(template))
        .catch(() => setBaseTemplate(null));
    } else {
      setBaseTemplate(null);
    }
  }, [baseTemplateId]);

  const insertAtCursor = (text: string) => {
    const textarea = systemPromptRef.current;
    if (!textarea) return;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const before = systemPrompt.substring(0, start);
    const after = systemPrompt.substring(end);
    setSystemPrompt(before + text + after);
    requestAnimationFrame(() => {
      textarea.focus();
      textarea.selectionStart = textarea.selectionEnd = start + text.length;
    });
  };

  const handleSubmit = async (
    isEditing: boolean,
    profileId: string | undefined,
    onSuccess: (saved: AgentProfile) => void
  ) => {
    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const data: AgentProfileCreate = {
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
      let saved: AgentProfile;
      if (isEditing && profileId) {
        const result = await updateProfile(profileId, data);
        if (!result) throw new Error('Failed to save profile');
        saved = result;
      } else {
        saved = await createProfile(data);
      }
      onSuccess(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (profileId: string, onSuccess: () => void) => {
    setDeleting(true);
    setError(null);
    try {
      const ok = await deleteProfile(profileId);
      if (ok) {
        onSuccess();
      } else {
        setError('Failed to delete profile');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete profile');
    } finally {
      setDeleting(false);
    }
  };

  return {
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
    baseTemplateId, setBaseTemplateId,
    baseTemplate,
    systemPromptRef,
    insertAtCursor,
    saving,
    deleting,
    error,
    handleSubmit,
    handleDelete,
  };
}

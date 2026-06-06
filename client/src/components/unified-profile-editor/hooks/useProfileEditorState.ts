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
  const [tags, setTags] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [reasoningStrategy, setReasoningStrategy] = useState<ReasoningStrategy>('auto');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [enableMemory, setEnableMemory] = useState(true);
  const [memoryChannel, setMemoryChannel] = useState('_global');
  const [enableTools, setEnableTools] = useState(true);
  // Phase 18.9.x — per-tool gating. `allowedTools === null` ⇔ "allow all"; a
  // non-null array switches to whitelist mode. `blockedTools` always wins.
  const [allowedTools, setAllowedTools] = useState<string[] | null>(null);
  const [blockedTools, setBlockedTools] = useState<string[]>([]);
  const [availableForDelegation, setAvailableForDelegation] = useState(true);
  // Phase 16.6 — ambassador section: when enabled, this profile can act as a
  // parallel conversation interpreter (briefs turns without entering the chat).
  const [ambassadorEnabled, setAmbassadorEnabled] = useState(false);
  const [ambassadorBriefingPrompt, setAmbassadorBriefingPrompt] = useState('');
  const [ambassadorVerbosity, setAmbassadorVerbosity] = useState<'brief' | 'normal' | 'deep'>('normal');
  // Functional-persona overrides (null = ride the shipped default).
  const [briefingPersona, setBriefingPersona] = useState<string | null>(null);
  const [qaPersona, setQaPersona] = useState<string | null>(null);
  const [draftPersona, setDraftPersona] = useState<string | null>(null);
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
      setTags(profile.tags ?? []);
      setDefaultModel(profile.defaultModel ?? '');
      setTemperature(profile.temperature);
      setReasoningStrategy(profile.reasoningStrategy);
      setSystemPrompt(profile.systemPrompt ?? '');
      setBaseTemplateId(profile.promptProfileId ?? '');
      setEnableMemory(profile.enableMemory);
      setMemoryChannel(profile.memoryChannel);
      setEnableTools(profile.enableTools);
      setAllowedTools(profile.allowedTools ?? null);
      setBlockedTools(profile.blockedTools ?? []);
      setAvailableForDelegation(profile.availableForDelegation ?? true);
      setAmbassadorEnabled(profile.ambassador?.enabled ?? false);
      setAmbassadorBriefingPrompt(profile.ambassador?.briefingPrompt ?? '');
      setAmbassadorVerbosity(profile.ambassador?.verbosity ?? 'normal');
      setBriefingPersona(profile.ambassador?.briefingPersona ?? null);
      setQaPersona(profile.ambassador?.qaPersona ?? null);
      setDraftPersona(profile.ambassador?.draftPersona ?? null);
    } else {
      setName('');
      setAvatar('sparkles');
      setDescription('');
      setTags([]);
      setDefaultModel('');
      setTemperature(0.7);
      setReasoningStrategy('auto');
      setSystemPrompt('');
      setBaseTemplateId('');
      setEnableMemory(true);
      setMemoryChannel('_global');
      setEnableTools(true);
      setAllowedTools(null);
      setBlockedTools([]);
      setAvailableForDelegation(true);
      setAmbassadorEnabled(false);
      setAmbassadorBriefingPrompt('');
      setAmbassadorVerbosity('normal');
      setBriefingPersona(null);
      setQaPersona(null);
      setDraftPersona(null);
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

  // Profile kind is set at creation and not flipped in normal editing.
  const kind: 'agent' | 'ambassador' = profile?.kind ?? 'agent';

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
        tags,
        default_model: defaultModel || undefined,
        temperature,
        reasoning_strategy: reasoningStrategy,
        prompt_profile_id: baseTemplateId || undefined,
        system_prompt: systemPrompt.trim() || undefined,
        enable_memory: enableMemory,
        memory_channel: memoryChannel,
        enable_tools: enableTools,
        // Send `null` explicitly when allow-all so the server clears any prior
        // whitelist; otherwise pass the array as-is. Always send blockedTools
        // (server defaults to []).
        allowed_tools: allowedTools,
        blocked_tools: blockedTools,
        available_for_delegation: availableForDelegation,
        kind,
        // Ambassador config travels only on ambassador-kind profiles; null clears
        // any legacy section on a normal agent. Persona overrides: null rides the
        // shipped default; a string overrides it.
        ambassador: kind === 'ambassador'
          ? {
              enabled: true,
              briefing_prompt: ambassadorBriefingPrompt.trim(),
              verbosity: ambassadorVerbosity,
              briefing_persona: briefingPersona,
              qa_persona: qaPersona,
              draft_persona: draftPersona,
            }
          : null,
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
    tags, setTags,
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
    kind,
    ambassadorEnabled, setAmbassadorEnabled,
    ambassadorBriefingPrompt, setAmbassadorBriefingPrompt,
    ambassadorVerbosity, setAmbassadorVerbosity,
    briefingPersona, setBriefingPersona,
    qaPersona, setQaPersona,
    draftPersona, setDraftPersona,
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

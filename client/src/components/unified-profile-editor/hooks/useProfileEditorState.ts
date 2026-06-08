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
  // Voice (TTS) block — spoken briefings + the immersive voice-mode opt-in.
  const [voiceMode, setVoiceMode] = useState(false);
  const [speechModel, setSpeechModel] = useState('');
  const [voice, setVoice] = useState('');
  const [transcriptionModel, setTranscriptionModel] = useState('');
  const [baseTemplateId, setBaseTemplateId] = useState('');
  const [baseTemplate, setBaseTemplate] = useState<PromptTemplate | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const systemPromptRef = useRef<HTMLTextAreaElement>(null);

  // Autosave (existing profiles): debounced, silent, location-preserving.
  const [autosaveState, setAutosaveState] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [hydratedId, setHydratedId] = useState<string | undefined>(undefined);
  const baselineRef = useRef<string | null>(null);
  const autosaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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
      setVoiceMode(profile.ambassador?.voiceMode ?? false);
      setSpeechModel(profile.ambassador?.speechModel ?? '');
      setVoice(profile.ambassador?.voice ?? '');
      setTranscriptionModel(profile.ambassador?.transcriptionModel ?? '');
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
    // Mark this profile id as hydrated → the baseline effect snapshots the freshly
    // loaded form (so autosave only fires on genuine user edits). Keyed on id only,
    // so list updates from a save don't reset the form (preserves cursor/scroll).
    setHydratedId(profile?.id ?? '__new__');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profile?.id]);

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

  // Single source of truth for the save payload — used by manual save AND autosave
  // (and by the autosave change-detector, so the two never drift).
  const buildPayload = (): AgentProfileCreate => ({
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
          voice_mode: voiceMode,
          speech_model: speechModel.trim() ? speechModel.trim() : null,
          voice: voice.trim() ? voice.trim() : null,
          transcription_model: transcriptionModel.trim() ? transcriptionModel.trim() : null,
        }
      : null,
  });

  // After a (re)hydration settles, snapshot the loaded form as the autosave
  // baseline. Defined BEFORE the autosave effect so on the hydration commit the
  // baseline updates first and the autosave effect then sees "no change".
  useEffect(() => {
    baselineRef.current = JSON.stringify(buildPayload());
    setAutosaveState('idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydratedId]);

  // Debounced autosave for existing profiles. New profiles are created via the
  // explicit "Create" button; once created they autosave like any other.
  useEffect(() => {
    if (!profile?.id || baselineRef.current === null) return;
    const pid = profile.id;
    const snap = JSON.stringify(buildPayload());
    if (snap === baselineRef.current) return; // no genuine change
    if (!name.trim()) return;                  // never autosave an invalid form
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    autosaveTimer.current = setTimeout(async () => {
      setAutosaveState('saving');
      try {
        await updateProfile(pid, buildPayload());
        baselineRef.current = snap;
        setAutosaveState('saved');
      } catch {
        setAutosaveState('idle'); // surfaced via manual save / next edit
      }
    }, 800);
    return () => {
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    name, avatar, description, tags, defaultModel, temperature, reasoningStrategy,
    baseTemplateId, systemPrompt, enableMemory, memoryChannel, enableTools,
    allowedTools, blockedTools, availableForDelegation, kind,
    ambassadorBriefingPrompt, ambassadorVerbosity, briefingPersona, qaPersona, draftPersona,
    voiceMode, speechModel, voice, transcriptionModel,
  ]);

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
      const data: AgentProfileCreate = buildPayload();
      let saved: AgentProfile;
      if (isEditing && profileId) {
        const result = await updateProfile(profileId, data);
        if (!result) throw new Error('Failed to save profile');
        saved = result;
      } else {
        saved = await createProfile(data);
      }
      // Keep the autosave baseline in sync so a manual save doesn't trigger a
      // redundant autosave right after.
      baselineRef.current = JSON.stringify(data);
      setAutosaveState('saved');
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
    voiceMode, setVoiceMode,
    speechModel, setSpeechModel,
    voice, setVoice,
    transcriptionModel, setTranscriptionModel,
    baseTemplateId, setBaseTemplateId,
    baseTemplate,
    systemPromptRef,
    insertAtCursor,
    saving,
    deleting,
    error,
    autosaveState,
    handleSubmit,
    handleDelete,
  };
}

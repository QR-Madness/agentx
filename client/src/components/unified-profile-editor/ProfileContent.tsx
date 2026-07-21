import { useEffect, useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import * as Tabs from '@radix-ui/react-tabs';
import {
  Brain,
  Cpu,
  Library,
  MessageSquare,
  Save,
  Trash2,
  X,
  Zap,
  Share2,
  Radio,
  AlertCircle,
  Sparkles,
  ListOrdered,
  GitBranch,
  Repeat2,
  RefreshCw,
  Orbit,
  Sliders,
  Volume2,
  Mic,
  ChevronLeft,
} from 'lucide-react';
import { type AgentProfile, type ModelInfo, type ReasoningStrategy } from '../../lib/api';
import { agentAccent } from '../../lib/agentAccent';
import { fetchModelsOnce } from '../common/modelCatalog';
import { ModelPickerField } from '../common/ModelPickerField';
import { VoicePicker } from '../common/VoicePicker';
import { AvatarPicker } from '../common/AvatarPicker';
import { ControlCard } from '../common/ControlCard';
import { CopyChip, SegmentedControl, Badge } from '../ui';
import { useConfirm } from '../ui/ConfirmDialog';
import { PromptEditor } from '../common/PromptEditor';
import { EffectivePromptPreview } from '../common/EffectivePromptPreview';
import { OverridablePromptField } from '../common/OverridablePromptField';
import { api } from '../../lib/api';
import { ChainStrip } from './ChainStrip';
import { NameDeck } from './NameDeck';
import { PromptLibraryPanel } from './PromptLibraryPanel';
import { ToolAccessSection } from './ToolAccessSection';
import { useProfileEditorState } from './hooks/useProfileEditorState';
import { useAgentProfile } from '../../contexts/AgentProfileContext';

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

const MAX_TAGS = 4;

/** Thinking patterns as segmented options (short label + glyph). Legacy
 * `tot`/`react` are no longer offered for new selection (chat degrades them:
 * tot→cot, react→native) but a profile still set to one keeps its segment so
 * the control never shows "nothing selected". */
const REASONING_SEGMENTS: { value: ReasoningStrategy; label: string; icon: ReactNode; title: string }[] = [
  { value: 'auto', label: 'Auto', icon: <Sparkles size={13} />, title: 'Pick the best pattern per message' },
  { value: 'native', label: 'Native', icon: <Orbit size={13} />, title: 'The model thinks freely — no scaffold' },
  { value: 'cot', label: 'Steps', icon: <ListOrdered size={13} />, title: 'Step-by-step — explicit numbered reasoning' },
  { value: 'step_back', label: 'StepBack', icon: <GitBranch size={13} />, title: 'Distill governing principles first' },
  { value: 'reflection', label: 'Reflect', icon: <RefreshCw size={13} />, title: 'Draft, self-critique, improve' },
  { value: 'deep_reflection', label: 'Reflect+', icon: <RefreshCw size={13} />, title: 'Live draft → critique → final (extra calls)' },
  { value: 'self_consistency', label: 'Consensus', icon: <Repeat2 size={13} />, title: 'Sample several solutions, keep the agreement' },
];

const LEGACY_SEGMENTS: { value: ReasoningStrategy; label: string; icon: ReactNode; title: string }[] = [
  { value: 'tot', label: 'ToT', icon: <GitBranch size={13} />, title: 'Tree of Thought (offline tasks; chat runs step-by-step)' },
  { value: 'react', label: 'ReAct', icon: <Repeat2 size={13} />, title: 'ReAct (offline tasks; chat tools already reason+act)' },
];

/** Temperature → a human label that tracks the value. */
function tempLabel(t: number): string {
  if (t < 0.4) return 'Precise';
  if (t < 0.9) return 'Balanced';
  if (t < 1.4) return 'Creative';
  return 'Wild';
}
const MAX_TAG_LEN = 24;

/** Chip-style editor for an agent's trait/role tags (max 4). */
function TagsField({ tags, setTags }: { tags: string[]; setTags: (t: string[]) => void }) {
  const [draft, setDraft] = useState('');

  const commit = () => {
    const tag = draft.trim().slice(0, MAX_TAG_LEN);
    setDraft('');
    if (!tag) return;
    if (tags.length >= MAX_TAGS) return;
    if (tags.some(t => t.toLowerCase() === tag.toLowerCase())) return;
    setTags([...tags, tag]);
  };

  const removeAt = (i: number) => setTags(tags.filter((_, idx) => idx !== i));

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      commit();
    } else if (e.key === 'Backspace' && !draft && tags.length) {
      removeAt(tags.length - 1);
    }
  };

  const full = tags.length >= MAX_TAGS;

  return (
    <div className="profile-form-group">
      <label>Tags</label>
      <div className="profile-tags-editor">
        {tags.map((tag, i) => (
          <span key={`${tag}-${i}`} className="profile-tag-chip">
            {tag}
            <button
              type="button"
              className="profile-tag-remove"
              onClick={() => removeAt(i)}
              aria-label={`Remove tag ${tag}`}
            >
              <X size={11} />
            </button>
          </span>
        ))}
        {!full && (
          <input
            type="text"
            className="profile-tag-input"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            onBlur={commit}
            maxLength={MAX_TAG_LEN}
            placeholder={tags.length ? 'Add tag…' : 'e.g. research, fast, writer'}
          />
        )}
      </div>
      <span className="profile-form-hint">
        Up to {MAX_TAGS} short labels (traits, roles) shown in the agent selector. Press Enter to add.
      </span>
    </div>
  );
}

interface ProfileContentProps {
  profile: AgentProfile | null;
  onSaved: (profileId: string) => void;
  onDeleted: () => void;
  onCancel: () => void;
  /** Mobile master-detail: when set, render a "‹ Profiles" back button that
   *  returns to the list. Undefined on desktop (list is always visible). */
  onBack?: () => void;
  /** Notifies the shell when the Prompt Library takes over the content area, so
   *  it can collapse the profile nav and give the library full width. */
  onLibraryOpenChange?: (open: boolean) => void;
  /** Chain-strip hops: select another profile in the editor (ProfileNav's select). */
  onSelectProfile?: (profileId: string) => void;
}

export function ProfileContent({
  profile,
  onSaved,
  onDeleted,
  onCancel,
  onBack,
  onLibraryOpenChange,
  onSelectProfile,
}: ProfileContentProps) {
  const isEditing = profile !== null;

  const {
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
    directMode, setDirectMode,
    allowedTools, setAllowedTools,
    blockedTools, setBlockedTools,
    availableForDelegation, setAvailableForDelegation,
    delegationHint, setDelegationHint,
    orgLevel, setRole,
    kind,
    ambassadorBriefingPrompt, setAmbassadorBriefingPrompt,
    ambassadorVerbosity, setAmbassadorVerbosity,
    briefingPersona, setBriefingPersona,
    qaPersona, setQaPersona,
    draftPersona, setDraftPersona,
    voicePersona, setVoicePersona,
    voiceMode, setVoiceMode,
    speechModel, setSpeechModel,
    voice, setVoice,
    transcriptionModel, setTranscriptionModel,
    setBaseTemplateId,
    baseTemplate,
    systemPromptRef,
    saving,
    deleting,
    error,
    handleSubmit,
    handleDelete,
  } = useProfileEditorState(profile);

  const confirm = useConfirm();
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryMode, setLibraryMode] = useState<'insert' | 'select'>('insert');
  const [modelCatalog, setModelCatalog] = useState<ModelInfo[]>([]);
  const isAmbassador = kind === 'ambassador';
  const accent = agentAccent(profile?.agentId ?? 'new');

  // Team membership card: who else is already on the ad-hoc delegation roster
  // (agent-kind, opted in, not this profile), plus the global gate so we can
  // warn when roster membership would be inert.
  const { profiles: allProfiles } = useAgentProfile();
  const rosterPeers = allProfiles
    .filter(p => p.kind === 'agent' && p.availableForDelegation && p.id !== profile?.id)
    .map(p => p.name);
  const [adhocDelegationEnabled, setAdhocDelegationEnabled] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.getConfig()
      .then(cfg => {
        if (cancelled) return;
        const alloy = (cfg.alloy || {}) as { allow_adhoc_delegation?: boolean };
        setAdhocDelegationEnabled(alloy.allow_adhoc_delegation ?? true);
      })
      .catch(() => { /* gate notice is best-effort — stay silent on failure */ });
    return () => { cancelled = true; };
  }, []);
  const [personaDefaults, setPersonaDefaults] = useState<{ briefing: string; qa: string; draft: string; voice: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchModelsOnce().then(m => { if (!cancelled) setModelCatalog(m); });
    return () => { cancelled = true; };
  }, []);

  // Persona defaults (for the ambassador voice override/diff) — fetched once, lazily.
  useEffect(() => {
    if (!isAmbassador || personaDefaults) return;
    let cancelled = false;
    api.ambassadorPersonaDefaults()
      .then(d => { if (!cancelled) setPersonaDefaults(d); })
      .catch(() => { if (!cancelled) setPersonaDefaults({ briefing: '', qa: '', draft: '', voice: '' }); });
    return () => { cancelled = true; };
  }, [isAmbassador, personaDefaults]);

  const selectedModel = defaultModel ? modelCatalog.find(m => m.id === defaultModel) : undefined;
  const selectedCtx = selectedModel?.context_length ?? selectedModel?.context_window;
  const selectedLabel = (() => {
    if (!defaultModel) return 'System default';
    if (selectedModel) return selectedModel.name;
    const parts = defaultModel.split(':');
    return parts.length > 1 ? parts.slice(1).join(':') : defaultModel;
  })();
  // Image-only model: outputs images but not text (e.g. flux). It can't act on a
  // system prompt or call tools, so Direct mode is auto-forced server-side; reflect
  // that here (toggle locked on + an info note recommending it).
  const _outMods = selectedModel?.output_modalities;
  const modelIsImageOnly = !!selectedModel
    && (!!selectedModel.supports_image || !!_outMods?.includes('image'))
    && Array.isArray(_outMods) && !_outMods.includes('text');
  const directModeEffective = directMode || modelIsImageOnly;
  const ctxLabel = selectedCtx
    ? (selectedCtx >= 1000 ? `${Math.round(selectedCtx / 1000)}k ctx` : `${selectedCtx} ctx`)
    : '';
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

  // Let the shell collapse the profile nav while the library owns the content.
  useEffect(() => {
    onLibraryOpenChange?.(libraryOpen);
  }, [libraryOpen, onLibraryOpenChange]);

  const handleSave = () => {
    handleSubmit(isEditing, profile?.id, (saved) => {
      onSaved(saved.id);
    });
  };

  const handleDeleteClick = async () => {
    if (!profile) return;
    const ok = await confirm({
      title: `Delete "${profile.name}"?`,
      body: 'This cannot be undone.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
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
              // Insert REPLACES the agent instructions with the snippet (per design:
              // the profile prompt is a single field, not an append-stack).
              onInsert={(content) => setSystemPrompt(content)}
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
            {onBack && (
              <button type="button" className="profile-mobile-back" onClick={onBack}>
                <ChevronLeft size={16} />
                Profiles
              </button>
            )}

            {error && (
              <div className="profile-error-banner">
                <AlertCircle size={15} />
                <span>{error}</span>
              </div>
            )}

            {/* Hero identity header */}
            <header
              className="profile-hero"
              style={{ ['--agent-accent' as string]: accent.accent, ['--agent-soft' as string]: accent.soft }}
            >
              <span className="profile-hero__aura" />
              <AvatarPicker
                value={avatar}
                onChange={setAvatar}
                size="lg"
                accent={accent}
                ariaLabel="Choose agent avatar"
                subjectSeed={(() => {
                  const n = name.trim();
                  if (!n) return undefined;
                  const detail = (tags[0] || description.split(/[.!?]/)[0] || '').trim();
                  return detail ? `${n} — ${detail}`.slice(0, 80) : n;
                })()}
              />
              <div className="profile-hero__main">
                <div className="profile-hero__name-row">
                  <input
                    className="profile-hero__name"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    placeholder="Name your agent…"
                    aria-label="Agent name"
                  />
                  <NameDeck onPick={setName} />
                </div>
                <div className="profile-hero__meta">
                  <Badge variant={isAmbassador ? 'accent' : 'neutral'} size="sm">
                    {isAmbassador ? 'Ambassador' : 'Agent'}
                  </Badge>
                  {profile?.isDefault && <Badge variant="success" size="sm">default</Badge>}
                  {profile?.agentId && <CopyChip value={profile.agentId} />}
                </div>
                <TagsField tags={tags} setTags={setTags} />
                <textarea
                  className="profile-hero__desc"
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Describe this profile's purpose… (shown to other agents when delegating)"
                  rows={2}
                />
              </div>
            </header>

            <Tabs.Root defaultValue="core" className="profile-tabs-root">
              <Tabs.List className="profile-tabs-list" aria-label="Profile settings">
                <Tabs.Trigger value="core" className="profile-tab-trigger">Core</Tabs.Trigger>
                <Tabs.Trigger value="tools" className="profile-tab-trigger">Tools</Tabs.Trigger>
                <Tabs.Trigger value="advanced" className="profile-tab-trigger">Advanced</Tabs.Trigger>
              </Tabs.List>

              {/* ── Core ── */}
              <Tabs.Content value="core" className="profile-tab-content">
                <motion.div
                  className="profile-card-grid"
                  initial="hidden"
                  animate="show"
                  variants={{ show: { transition: { staggerChildren: 0.05 } } }}
                >
                  <ControlCard full icon={<MessageSquare size={14} />} title={isAmbassador ? 'Communications' : 'Instructions'}>
                    {!isAmbassador && (
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
                      <span className="profile-form-hint">Optionally start from a saved template</span>
                    </div>
                    )}

                    <div className="profile-form-group">
                      <PromptEditor
                        ref={systemPromptRef}
                        label={isAmbassador ? 'Communications (personality)' : undefined}
                        hint={isAmbassador
                          ? 'Modifies the flavour of the ambassador — how it speaks to you across all its briefings.'
                          : 'Layered on top of the global System Prompt — leave empty to use it as-is.'}
                        placeholder={isAmbassador
                          ? 'Give your ambassador a personality… (leave empty for a neutral voice)'
                          : 'Custom instructions for this agent… (leave empty to use defaults)'}
                        value={systemPrompt}
                        onChange={setSystemPrompt}
                        onInsertFromLibrary={() => openLibrary('insert')}
                      />
                      {!isAmbassador && <EffectivePromptPreview name={name} agentPrompt={systemPrompt} />}
                    </div>
                  </ControlCard>

                  <ControlCard icon={<Cpu size={14} />} title="Model" summary={`${selectedLabel}${ctxLabel ? ` · ${ctxLabel}` : ''}`}>
                    <div className="profile-model-selector-wrap">
                      <ModelPickerField
                        label="Model"
                        value={defaultModel}
                        onChange={setDefaultModel}
                        showDefault
                      />
                    </div>
                  </ControlCard>

                  <ControlCard icon={<Sliders size={14} />} title="Generation" summary={`${temperature.toFixed(1)} · ${tempLabel(temperature)}`}>
                    <div className="profile-form-group">
                      <label>Temperature: {temperature.toFixed(1)} · <span className="profile-temp-label">{tempLabel(temperature)}</span></label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={e => setTemperature(parseFloat(e.target.value))}
                        className="profile-slider profile-temp"
                      />
                      <div className="profile-slider-labels">
                        <span>Focused</span>
                        <span>Creative</span>
                      </div>
                    </div>

                    <div className="profile-form-group">
                      <label>Thinking pattern</label>
                      <SegmentedControl
                        size="sm"
                        ariaLabel="Thinking pattern"
                        value={reasoningStrategy}
                        onChange={(v) => setReasoningStrategy(v)}
                        options={[
                          ...REASONING_SEGMENTS,
                          ...LEGACY_SEGMENTS.filter(s => s.value === reasoningStrategy),
                        ]}
                      />
                    </div>
                  </ControlCard>

                  {/* Team membership — agents only (ambassadors are never delegation targets) */}
                  {!isAmbassador && (
                  <ControlCard
                    icon={<Share2 size={14} />}
                    title="Team membership"
                    summary={orgLevel !== 'agent' ? orgLevel : (availableForDelegation ? 'in roster' : 'not in roster')}
                  >
                    <div className="profile-form-group">
                      <label>Role</label>
                      <SegmentedControl
                        size="sm"
                        ariaLabel="Organization role"
                        value={orgLevel === 'executive' ? 'agent' : orgLevel}
                        onChange={(v) => setRole(v)}
                        options={[
                          { value: 'agent', label: 'Agent', title: 'A team member — works tasks handed down by its lead' },
                          { value: 'lead', label: 'Lead', title: 'Runs a team hands-on: works its members, may do manual work' },
                          { value: 'manager', label: 'Manager', title: 'Staffing director — reports and directives only, delegates to the leads of its teams' },
                        ]}
                      />
                      <span className="profile-form-hint">
                        Once this agent is part of an organization (a team with a manager), the
                        chain of command decides who it may delegate to — the roster toggle below
                        then only governs org-free agents. Choosing <strong>Manager</strong> applies
                        the report-only tool template (manual-work tools blocked — editable in the
                        Tools tab).
                      </span>
                      {profile && <ChainStrip profile={profile} onHop={onSelectProfile} />}
                    </div>
                    <label className="profile-toggle-row">
                      <span className="profile-toggle-label">
                        <Share2 size={15} />
                        Join the team roster
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
                      When on, other agents see this profile as a teammate and can hand it
                      subtasks via the <code>delegate_to</code> tool.
                    </span>
                    {/* Deliberately NOT gated on the roster toggle: team + chain rosters
                        read the hint too, and org members are often flat-opted-out. */}
                    <div className="profile-form-field" style={{ marginTop: 12 }}>
                      <label className="profile-form-label">Specialty</label>
                      <textarea
                        value={delegationHint}
                        onChange={e => setDelegationHint(e.target.value)}
                        placeholder="What this agent is best at — teammates read this when deciding whom to delegate to"
                        rows={2}
                        maxLength={200}
                        className="profile-system-prompt"
                      />
                      <span className="profile-form-hint">
                        {availableForDelegation
                          ? 'One line, shown in teammates’ rosters. Leave empty to fall back to the profile description.'
                          : 'One line, shown wherever teammates pick delegates — team and chain-of-command rosters read it even while this profile is off the flat roster.'}
                      </span>
                    </div>
                    <span className="profile-form-hint" style={{ marginTop: 10 }}>
                      {rosterPeers.length > 0
                        ? <>Current roster: {rosterPeers.join(', ')}{availableForDelegation ? ' — and this profile' : ''}.</>
                        : availableForDelegation
                          ? 'This profile is the only one on the roster so far.'
                          : 'No other agents are on the roster yet.'}
                    </span>
                    {adhocDelegationEnabled === false && (
                      <span className="profile-form-hint" style={{ marginTop: 6 }}>
                        <AlertCircle size={12} style={{ verticalAlign: 'text-top', marginRight: 4 }} />
                        Ad-hoc delegation is disabled globally (Settings → Agent Teams) — roster
                        membership has no effect until it's re-enabled.
                      </span>
                    )}
                  </ControlCard>
                  )}

                  {/* Ambassador voices — ambassador-kind profiles only */}
                  {isAmbassador && (
                  <ControlCard full icon={<Radio size={14} />} title="Ambassador voices">
                    <span className="profile-form-hint">
                      This profile <strong>is</strong> an ambassador — it briefs you in parallel
                      and never enters the chat. Its personality is the Communications prompt above;
                      the load-bearing voices below ship with a default you can override and reset.
                    </span>
                    <div className="profile-form-field" style={{ marginTop: 12 }}>
                      <label className="profile-form-label">Briefing verbosity</label>
                      <select
                        className="profile-select"
                        value={ambassadorVerbosity}
                        onChange={e =>
                          setAmbassadorVerbosity(e.target.value as 'brief' | 'normal' | 'deep')
                        }
                      >
                        <option value="brief">Brief — one or two sentences</option>
                        <option value="normal">Normal — a short paragraph</option>
                        <option value="deep">Deep — reasoning, tensions, open questions</option>
                      </select>
                    </div>
                    <div className="profile-form-field" style={{ marginTop: 12 }}>
                      <label className="profile-form-label">Briefing instructions</label>
                      <textarea
                        value={ambassadorBriefingPrompt}
                        onChange={e => setAmbassadorBriefingPrompt(e.target.value)}
                        placeholder="What should the ambassador surface or emphasize when briefing a turn? (optional)"
                        rows={3}
                        className="profile-system-prompt"
                      />
                    </div>
                    {personaDefaults && (
                      <div className="profile-form-field" style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                        <OverridablePromptField
                          title="Briefing voice"
                          description="How it narrates a single turn to you."
                          defaultText={personaDefaults.briefing}
                          override={briefingPersona}
                          onChange={setBriefingPersona}
                          onReset={() => setBriefingPersona(null)}
                        />
                        <OverridablePromptField
                          title="Q&A voice"
                          description="How it answers free-form questions about the conversation."
                          defaultText={personaDefaults.qa}
                          override={qaPersona}
                          onChange={setQaPersona}
                          onReset={() => setQaPersona(null)}
                        />
                        <OverridablePromptField
                          title="Draft voice"
                          description="How it ghostwrites a message from you to the agent."
                          defaultText={personaDefaults.draft}
                          override={draftPersona}
                          onChange={setDraftPersona}
                          onReset={() => setDraftPersona(null)}
                        />
                        <OverridablePromptField
                          title="Voice-command routing"
                          description="How it interprets spoken commands — answer, relay, or a confirm-first proposal."
                          defaultText={personaDefaults.voice}
                          override={voicePersona}
                          onChange={setVoicePersona}
                          onReset={() => setVoicePersona(null)}
                        />
                      </div>
                    )}
                  </ControlCard>
                  )}

                  {/* Voice (TTS) — ambassador-kind profiles only */}
                  {isAmbassador && (
                  <ControlCard full icon={<Volume2 size={14} />} title="Voice">
                    <span className="profile-form-hint">
                      Speak briefings + answers aloud via OpenRouter text-to-speech. Turn on
                      <strong> voice mode</strong> for an immersive, two-way voice surface in the
                      Ambassador panel. Leave the model/voice empty to use the shipped default
                      (<code>microsoft/mai-voice-2</code>).
                    </span>

                    <label className="profile-toggle-row" style={{ marginTop: 12 }}>
                      <span className="profile-toggle-label">
                        <Mic size={15} />
                        Voice mode
                      </span>
                      <div className="profile-toggle-right">
                        <input
                          type="checkbox"
                          checked={voiceMode}
                          onChange={e => setVoiceMode(e.target.checked)}
                          className="profile-toggle-input"
                        />
                        <span className={`profile-toggle-switch ${voiceMode ? 'active' : ''}`} />
                      </div>
                    </label>
                    <span className="profile-form-hint">
                      Offer the immersive, two-way voice surface for this ambassador (hold-to-talk
                      voice input + spoken briefings). With it off, you still get the per-message
                      speaker buttons.
                    </span>

                    <div className="profile-form-field" style={{ marginTop: 12 }}>
                      <label className="profile-form-label">Speech model</label>
                      <ModelPickerField
                        value={speechModel}
                        onChange={setSpeechModel}
                        showDefault
                        requireCapability="speech"
                        placeholder="microsoft/mai-voice-2 (default)"
                      />
                    </div>

                    <div className="profile-form-group" style={{ marginTop: 12 }}>
                      <label>Voice</label>
                      <VoicePicker
                        model={speechModel || 'openrouter:microsoft/mai-voice-2'}
                        value={voice}
                        onChange={setVoice}
                        placeholder="Custom voice id (model-specific)"
                      />
                    </div>

                    <div className="profile-form-field" style={{ marginTop: 12 }}>
                      <label className="profile-form-label">
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <Mic size={13} /> Transcription model
                        </span>
                        <span className="profile-form-hint" style={{ display: 'block', marginTop: 2 }}>
                          Speech-to-text for push-to-talk voice input. The transcript fills the input
                          for review before you send.
                        </span>
                      </label>
                      <ModelPickerField
                        value={transcriptionModel}
                        onChange={setTranscriptionModel}
                        showDefault
                        requireCapability="transcription"
                        placeholder="openai/whisper-1 (default)"
                      />
                    </div>
                  </ControlCard>
                  )}
                </motion.div>
              </Tabs.Content>

              {/* ── Tools ── */}
              <Tabs.Content value="tools" className="profile-tab-content">
                <motion.div
                  className="profile-card-grid"
                  initial="hidden"
                  animate="show"
                  variants={{ show: { transition: { staggerChildren: 0.05 } } }}
                >
                <ControlCard full icon={<Zap size={14} />} title="Tools" summary={enableTools ? 'enabled' : 'off'}>
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
                </ControlCard>
                </motion.div>
              </Tabs.Content>

              {/* ── Advanced ── */}
              <Tabs.Content value="advanced" className="profile-tab-content">
                <motion.div
                  className="profile-card-grid"
                  initial="hidden"
                  animate="show"
                  variants={{ show: { transition: { staggerChildren: 0.05 } } }}
                >
                  <ControlCard icon={<Brain size={14} />} title="Memory" summary={enableMemory ? memoryChannel : 'off'}>
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
                  </ControlCard>

                  <ControlCard
                    icon={<Zap size={14} />}
                    title="Direct mode"
                    summary={directModeEffective ? 'on' : 'off'}
                  >
                    <label className="profile-toggle-row">
                      <span className="profile-toggle-label">
                        <Zap size={15} />
                        Prompt only
                      </span>
                      <div className="profile-toggle-right">
                        <input
                          type="checkbox"
                          checked={directModeEffective}
                          disabled={modelIsImageOnly}
                          onChange={e => setDirectMode(e.target.checked)}
                          className="profile-toggle-input"
                        />
                        <span className={`profile-toggle-switch ${directModeEffective ? 'active' : ''}`} />
                      </div>
                    </label>
                    <span className="profile-form-hint">
                      Sends the model only your message — no system prompt, memory, or tools.
                      Best for a transform-only model (a fast classifier/rewriter) or an image
                      generator.
                    </span>
                    {modelIsImageOnly && (
                      <div className="profile-nested" style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <AlertCircle size={14} style={{ marginTop: 2, flexShrink: 0 }} className="text-info" />
                        <span className="profile-form-hint" style={{ margin: 0 }}>
                          This is an image-only model, so Direct mode is always on — it can’t use a
                          prompt or tools, only the text you send becomes the image prompt.
                        </span>
                      </div>
                    )}
                  </ControlCard>
                </motion.div>
              </Tabs.Content>
            </Tabs.Root>

            {/* Footer — new profiles only (Cancel/Create). An existing profile
                autosaves silently and the header ✕ closes it, so edit mode has
                no footer bar; Delete for a non-default profile floats over the
                panel instead (see the floating Delete below). */}
            {!isEditing && (
              <div className="profile-footer">
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
                    {saving ? 'Saving…' : 'Create Profile'}
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Delete — non-default profiles in edit mode. Anchored absolute
          within .profile-content-outer (position:relative), NOT fixed/portal,
          so it's Tauri-safe and stays put while the form scrolls. The default
          ambassador is system-owned (org apex) — no delete affordance; the
          server refuses the DELETE anyway. */}
      {!libraryOpen && isEditing && !profile?.isDefault
        && !(profile?.kind === 'ambassador' && profile?.isDefaultAmbassador) && (
        <button
          type="button"
          className="profile-btn-danger profile-delete-float"
          onClick={handleDeleteClick}
          disabled={deleting || saving}
          title="Delete this profile"
        >
          <Trash2 size={15} />
          {deleting ? 'Deleting…' : 'Delete'}
        </button>
      )}
    </div>
  );
}

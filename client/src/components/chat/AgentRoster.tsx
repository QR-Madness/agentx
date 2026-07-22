/**
 * AgentRoster — the full two-pane agent browser: an org-chart roster on the left
 * (chain of command via `lib/orgPlacement.groupRoster`) and a live **dossier** on
 * the right (everything about the highlighted agent — role, chain of command,
 * model, reasoning, capabilities, tags). The browse-and-compare sibling of the
 * composer's compact AgentSelectorDropdown; opened from the picker footer, the
 * command palette, and anywhere "see all agents" makes sense.
 *
 * Full-screen surface (registered in ModalPortal's FULLSCREEN_SURFACES): renders
 * its own backdrop + container + Escape/close, mounted only while open.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowUpRight, Bot, BrainCircuit, Check, ChevronLeft, Crown, PenLine, Plus,
  Search, Sparkles, Users, Wrench, X, Zap,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useModal } from '../../contexts/ModalContext';
import { useIsMobile } from '../../lib/hooks';
import { groupRoster, orgPlacement, type RosterGroup } from '../../lib/orgPlacement';
import { modelShortLabel } from '../../lib/modelLabel';
import type { AgentProfile } from '../../lib/api/types';
import { AgentAvatar } from '../common/AgentAvatar';
import './AgentRoster.css';

type Tier = 'manager' | 'lead';

const REASONING_LABEL: Record<string, string> = {
  auto: 'Auto', native: 'Native', cot: 'Chain of thought', tot: 'Tree of thought',
  react: 'ReAct', reflection: 'Reflection', deep_reflection: 'Deep reflection',
  step_back: 'Step back', self_consistency: 'Self-consistency',
};

function roleLine(p: AgentProfile): string {
  return (p.delegationHint || p.description || '').trim();
}

function groupHeader(group: RosterGroup): { icon: React.ReactNode; label: string } {
  if (group.kind === 'manager') return { icon: <Crown size={12} />, label: 'Manager' };
  if (group.kind === 'org-free') return { icon: <Bot size={12} />, label: 'Independent' };
  return { icon: <Users size={12} />, label: group.label };
}

function TierBadge({ tier }: { tier: Tier }) {
  return (
    <span className={`roster-tier roster-tier--${tier}`}>{tier === 'manager' ? 'MGR' : 'LEAD'}</span>
  );
}

/** A capability chip — success-tinted when on, muted when off. */
function Pip({ on, icon, label }: { on: boolean; icon: React.ReactNode; label: string }) {
  return (
    <span className={`roster-pip ${on ? 'roster-pip--on' : 'roster-pip--off'}`}>
      {icon}
      {label}
    </span>
  );
}

export function AgentRoster({
  onClose,
  initialAgentId,
}: {
  onClose: () => void;
  initialAgentId?: string;
}) {
  const { profiles, activeProfile, setActiveProfile } = useAgentProfile();
  const { workflows } = useAlloyWorkflow();
  const { openModal } = useModal();
  const isMobile = useIsMobile();

  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(
    initialAgentId ?? activeProfile?.id ?? null,
  );

  // Escape + body scroll lock (this surface renders bare, so it owns both).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => { window.removeEventListener('keydown', handler); document.body.style.overflow = ''; };
  }, [onClose]);

  const tierByAgentId = useMemo(() => {
    const m = new Map<string, Tier>();
    for (const wf of workflows) if (wf.managerAgentId) m.set(wf.managerAgentId, 'manager');
    for (const wf of workflows) if (!m.has(wf.supervisorAgentId)) m.set(wf.supervisorAgentId, 'lead');
    return m;
  }, [workflows]);

  const groups = useMemo(() => {
    const all = groupRoster(profiles, workflows).filter(g => g.kind !== 'ambassador');
    const q = query.trim().toLowerCase();
    if (!q) return all;
    return all
      .map(g => ({
        ...g,
        profiles: g.profiles.filter(p =>
          p.name.toLowerCase().includes(q) ||
          (p.defaultModel ?? '').toLowerCase().includes(q) ||
          roleLine(p).toLowerCase().includes(q) ||
          (p.tags ?? []).some(t => t.toLowerCase().includes(q)),
        ),
      }))
      .filter(g => g.profiles.length > 0);
  }, [profiles, workflows, query]);

  const flat = useMemo(() => groups.flatMap(g => g.profiles), [groups]);
  const selected = useMemo(
    () => flat.find(p => p.id === selectedId) ?? flat[0] ?? null,
    [flat, selectedId],
  );

  const placement = useMemo(
    () => (selected ? orgPlacement(selected.agentId, profiles, workflows) : null),
    [selected, profiles, workflows],
  );

  const useAgent = useCallback((id: string) => { setActiveProfile(id); onClose(); }, [setActiveProfile, onClose]);
  const editAgent = useCallback((id: string) => {
    openModal({ id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full', props: { initialProfileId: id } });
    onClose();
  }, [openModal, onClose]);
  const createAgent = useCallback(() => {
    openModal({ id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full', props: { isNew: true } });
    onClose();
  }, [openModal, onClose]);

  // Mobile master-detail: list until a card is tapped, then the dossier.
  const [mobileDetail, setMobileDetail] = useState(false);
  const showList = !isMobile || !mobileDetail;
  const showDossier = !isMobile || mobileDetail;
  const pick = (id: string) => { setSelectedId(id); if (isMobile) setMobileDetail(true); };

  return (
    <>
      <div className="roster-backdrop" onClick={onClose} aria-hidden />
      <div className="roster-container" role="dialog" aria-label="Agent roster" aria-modal="true">
        <header className="roster-header">
          <h1 className="roster-title">
            <Users size={18} className="text-accent" />
            Roster
            <span className="roster-title-count">{flat.length} agent{flat.length === 1 ? '' : 's'}</span>
          </h1>
          <div className="roster-header-actions">
            <button className="roster-new" onClick={createAgent}>
              <Plus size={15} /> New agent
            </button>
            <button className="roster-close" onClick={onClose} aria-label="Close" title="Close">
              <X size={20} />
            </button>
          </div>
        </header>

        <div className="roster-body">
          {/* Left — the org chart */}
          {showList && (
            <aside className="roster-rail">
              <div className="roster-search">
                <Search size={14} />
                <input
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Search agents, roles, models…"
                  aria-label="Search agents"
                />
              </div>
              <div className="roster-list">
                {flat.length === 0 ? (
                  <p className="roster-empty">No agents match “{query}”.</p>
                ) : (
                  groups.map(group => {
                    const head = groupHeader(group);
                    return (
                      <div key={group.label} className="roster-group">
                        <div className="roster-group-label">{head.icon} {head.label}</div>
                        {group.profiles.map(p => {
                          const tier = tierByAgentId.get(p.agentId);
                          const model = modelShortLabel(p.defaultModel);
                          const role = roleLine(p);
                          const isSel = selected?.id === p.id;
                          return (
                            <button
                              key={p.id}
                              className={`roster-card ${isSel ? 'active' : ''}`}
                              onClick={() => pick(p.id)}
                            >
                              <div className="roster-card-avatar"><AgentAvatar avatar={p.avatar} size={20} fill /></div>
                              <div className="roster-card-info">
                                <span className="roster-card-name">
                                  <span className="roster-card-name-text">{p.name}</span>
                                  {tier && <TierBadge tier={tier} />}
                                  {p.isDefault && <span className="roster-card-badge">default</span>}
                                </span>
                                {model && <span className="roster-card-model">{model}</span>}
                                {role && <span className="roster-card-role">{role}</span>}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    );
                  })
                )}
              </div>
            </aside>
          )}

          {/* Right — the dossier */}
          {showDossier && (
            <section className="roster-dossier">
              {!selected ? (
                <div className="roster-dossier-empty">Select an agent to see its dossier.</div>
              ) : (
                <>
                  {isMobile && (
                    <button className="roster-back" onClick={() => setMobileDetail(false)}>
                      <ChevronLeft size={18} /> Roster
                    </button>
                  )}
                  <div className="roster-dossier-scroll">
                    <div className="roster-dossier-head">
                      <div className="roster-dossier-avatar"><AgentAvatar avatar={selected.avatar} size={56} fill /></div>
                      <div className="roster-dossier-headinfo">
                        <div className="roster-dossier-name">
                          {selected.name}
                          {tierByAgentId.get(selected.agentId) && <TierBadge tier={tierByAgentId.get(selected.agentId)!} />}
                          {selected.isDefault && <span className="roster-card-badge">default</span>}
                        </div>
                        {roleLine(selected) && <p className="roster-dossier-role">{roleLine(selected)}</p>}
                      </div>
                    </div>

                    {/* Chain of command */}
                    <div className="roster-section">
                      <div className="roster-section-label">Chain of command</div>
                      {placement?.inOrg ? (
                        <div className="roster-chain">
                          {placement.teams.map(t => (
                            <div key={t.teamId} className="roster-chain-row">
                              <span className={`roster-chain-role roster-chain-role--${t.role}`}>{t.role}</span>
                              <span className="roster-chain-of">{t.teamName}</span>
                            </div>
                          ))}
                          {placement.superiors.length > 0 && (
                            <div className="roster-chain-line">
                              <span className="roster-chain-dir">Reports to</span>
                              {placement.superiors.map(s => (
                                <button key={s.id} className="roster-chip" onClick={() => setSelectedId(s.id)}>
                                  <AgentAvatar avatar={s.avatar} size={14} /> {s.name}
                                </button>
                              ))}
                            </div>
                          )}
                          {placement.subordinates.length > 0 && (
                            <div className="roster-chain-line">
                              <span className="roster-chain-dir">Directs</span>
                              {placement.subordinates.map(s => (
                                <button key={s.id} className="roster-chip" onClick={() => setSelectedId(s.id)}>
                                  <AgentAvatar avatar={s.avatar} size={14} /> {s.name}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="roster-independent">
                          Independent agent{selected.availableForDelegation ? ' · available for ad-hoc delegation' : ''}.
                        </p>
                      )}
                    </div>

                    {/* Configuration */}
                    <div className="roster-section">
                      <div className="roster-section-label">Configuration</div>
                      <div className="roster-kv">
                        <span className="roster-kv-key">Model</span>
                        <span className="roster-kv-val">
                          {modelShortLabel(selected.defaultModel) ?? 'Inherits default'}
                          {selected.defaultModel && <code className="roster-kv-code">{selected.defaultModel}</code>}
                        </span>
                      </div>
                      <div className="roster-pips">
                        <Pip on icon={<Sparkles size={12} />} label={REASONING_LABEL[selected.reasoningStrategy] ?? selected.reasoningStrategy} />
                        <Pip on={selected.enableMemory} icon={<BrainCircuit size={12} />} label="Memory" />
                        <Pip on={selected.enableTools} icon={<Wrench size={12} />} label="Tools" />
                        <Pip on={!!selected.availableForDelegation} icon={<Zap size={12} />} label="Delegation" />
                      </div>
                    </div>

                    {/* Tags */}
                    {selected.tags && selected.tags.length > 0 && (
                      <div className="roster-section">
                        <div className="roster-section-label">Tags</div>
                        <div className="roster-tags">
                          {selected.tags.map(t => <span key={t} className="roster-tag">{t}</span>)}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Sticky actions */}
                  <div className="roster-actions">
                    <button className="roster-action-primary" onClick={() => useAgent(selected.id)}>
                      {selected.id === activeProfile?.id ? <><Check size={15} /> Current agent</> : <><ArrowUpRight size={15} /> Use this agent</>}
                    </button>
                    <button className="roster-action-secondary" onClick={() => editAgent(selected.id)}>
                      <PenLine size={15} /> Edit profile
                    </button>
                  </div>
                </>
              )}
            </section>
          )}
        </div>
      </div>
    </>
  );
}

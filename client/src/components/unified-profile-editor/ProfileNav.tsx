/**
 * ProfileNav — the roster rail of the profile editor.
 *
 * Two views over the same rows:
 * - **Grouped** (default): sections derived live from the org chart via
 *   `lib/orgPlacement.groupRoster` — Manager, one per team (lead first),
 *   Org-free, Ambassador — with two-line annotated rows (tier chip, model
 *   short-id, tags) and a search filter across name/tags/model.
 * - **Manual order**: the flat dnd-kit drag list (the persisted profile order
 *   also drives within-group ordering in the grouped view). Reordering is
 *   disabled while a search filter is active.
 */

import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { GripVertical, ListOrdered, Network, Plus, Search, User } from 'lucide-react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { groupRoster } from '../../lib/orgPlacement';
import { modelShortLabel } from '../../lib/modelLabel';
import type { AgentProfile } from '../../lib/api';
import { AgentAvatar } from '../common/AgentAvatar';
import { IconButton, Input } from '../ui';
import { navVariants } from '../unified-settings/animations/transitions';

const NAV_MODE_KEY = 'agentx:profile-nav-mode';

type Tier = 'manager' | 'lead';

interface ProfileNavProps {
  selectedProfileId: string | null;
  isCreatingNew: boolean;
  onSelectProfile: (id: string) => void;
  onCreateNew: () => void;
}

/** The two-line annotated row body, shared by both views. */
function RowBody({ profile, tier }: { profile: AgentProfile; tier?: Tier }) {
  const model = modelShortLabel(profile.defaultModel);
  const tags = (profile.tags ?? []).slice(0, 2);
  return (
    <>
      <div className="profile-nav-avatar">
        <AgentAvatar avatar={profile.avatar} size={15} fill />
      </div>
      <div className="profile-nav-info">
        <span className="profile-nav-line1">
          <span className="profile-nav-name">{profile.name}</span>
          {tier && (
            <span className={`profile-nav-tier profile-nav-tier--${tier}`}>
              {tier === 'manager' ? 'MGR' : 'LEAD'}
            </span>
          )}
          {profile.isDefault && <span className="profile-nav-badge">default</span>}
        </span>
        {(model || tags.length > 0) && (
          <span className="profile-nav-line2">
            {model && <span className="profile-nav-model">{model}</span>}
            {tags.map(t => (
              <span key={t} className="profile-nav-tag">{t}</span>
            ))}
          </span>
        )}
      </div>
    </>
  );
}

function rowInteraction(onSelect: () => void) {
  return {
    role: 'button' as const,
    tabIndex: 0,
    onClick: onSelect,
    onKeyDown: (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onSelect();
      }
    },
  };
}

/** One draggable profile row (manual-order view) — handle reorders, the rest selects. */
function SortableProfileItem({
  profile,
  tier,
  isActive,
  onSelect,
}: {
  profile: AgentProfile;
  tier?: Tier;
  isActive: boolean;
  onSelect: () => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: profile.id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`profile-nav-item ${isActive ? 'active' : ''}`}
      data-dragging={isDragging || undefined}
      {...rowInteraction(onSelect)}
    >
      <button
        type="button"
        className="profile-nav-handle"
        aria-label={`Reorder ${profile.name}`}
        onClick={(e) => e.stopPropagation()}
        {...attributes}
        {...listeners}
      >
        <GripVertical size={15} />
      </button>
      <RowBody profile={profile} tier={tier} />
    </div>
  );
}

export function ProfileNav({
  selectedProfileId,
  isCreatingNew,
  onSelectProfile,
  onCreateNew,
}: ProfileNavProps) {
  const { profiles, reorderProfiles } = useAgentProfile();
  const { workflows } = useAlloyWorkflow();

  const [query, setQuery] = useState('');
  const [manual, setManual] = useState(() => {
    try { return localStorage.getItem(NAV_MODE_KEY) === 'manual'; } catch { return false; }
  });
  const toggleManual = () => {
    setManual(m => {
      const next = !m;
      try { localStorage.setItem(NAV_MODE_KEY, next ? 'manual' : 'grouped'); } catch { /* ignore */ }
      return next;
    });
  };

  // Tier chips come from the org chart's edges (a declared-but-teamless tier
  // shows no chip — structure lives on the workflows, same as the Chain strip).
  const tierOf = useMemo(() => {
    const map = new Map<string, Tier>();
    for (const p of profiles) {
      if (workflows.some(w => w.managerAgentId === p.agentId)) map.set(p.id, 'manager');
      else if (workflows.some(w => w.supervisorAgentId === p.agentId)) map.set(p.id, 'lead');
    }
    return map;
  }, [profiles, workflows]);

  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return profiles;
    return profiles.filter(p =>
      p.name.toLowerCase().includes(q) ||
      (p.tags ?? []).some(t => t.toLowerCase().includes(q)) ||
      (p.defaultModel ?? '').toLowerCase().includes(q),
    );
  }, [profiles, q]);

  const groups = useMemo(() => groupRoster(filtered, workflows), [filtered, workflows]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const handleReorder = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const ids = profiles.map((p) => p.id);
    const from = ids.indexOf(String(active.id));
    const to = ids.indexOf(String(over.id));
    if (from < 0 || to < 0) return;
    void reorderProfiles(arrayMove(ids, from, to));
  };

  const isActive = (id: string) => !isCreatingNew && selectedProfileId === id;
  // Dragging under a filter would reorder against a partial list — fall back
  // to plain rows until the query clears.
  const dragEnabled = manual && !q;

  return (
    <motion.nav
      className="profile-nav"
      variants={navVariants}
      initial="initial"
      animate="animate"
    >
      <div className="profile-nav-header">
        <User size={14} />
        <span>Agent Profiles</span>
        <IconButton
          size="xs"
          className="profile-nav-mode"
          aria-label={manual ? 'Group by organization' : 'Switch to manual order'}
          title={manual ? 'Group by organization' : 'Manual order (drag to reorder)'}
          active={manual}
          onClick={toggleManual}
        >
          {manual ? <Network size={13} /> : <ListOrdered size={13} />}
        </IconButton>
      </div>

      <Input
        icon={<Search size={13} />}
        className="profile-nav-search"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search agents…"
        aria-label="Search agent profiles"
      />

      <div className="profile-nav-list">
        {manual ? (
          dragEnabled ? (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              modifiers={[restrictToVerticalAxis, restrictToParentElement]}
              onDragEnd={handleReorder}
            >
              <SortableContext items={profiles.map((p) => p.id)} strategy={verticalListSortingStrategy}>
                {profiles.map((profile) => (
                  <SortableProfileItem
                    key={profile.id}
                    profile={profile}
                    tier={tierOf.get(profile.id)}
                    isActive={isActive(profile.id)}
                    onSelect={() => onSelectProfile(profile.id)}
                  />
                ))}
              </SortableContext>
            </DndContext>
          ) : (
            filtered.map((profile) => (
              <div
                key={profile.id}
                className={`profile-nav-item ${isActive(profile.id) ? 'active' : ''}`}
                {...rowInteraction(() => onSelectProfile(profile.id))}
              >
                <RowBody profile={profile} tier={tierOf.get(profile.id)} />
              </div>
            ))
          )
        ) : (
          groups.map((group) => (
            <div className="profile-nav-group" key={`${group.kind}-${group.teamId ?? group.label}`}>
              <div className="profile-nav-group-label">{group.label}</div>
              {group.profiles.map((profile) => (
                <div
                  key={profile.id}
                  className={`profile-nav-item ${isActive(profile.id) ? 'active' : ''}`}
                  {...rowInteraction(() => onSelectProfile(profile.id))}
                >
                  <RowBody profile={profile} tier={tierOf.get(profile.id)} />
                </div>
              ))}
            </div>
          ))
        )}

        {q && filtered.length === 0 && (
          <div className="profile-nav-empty">No agent matches “{query.trim()}”.</div>
        )}

        {isCreatingNew && (
          <div className="profile-nav-item active profile-nav-new-placeholder">
            <div className="profile-nav-avatar">
              <Plus size={15} />
            </div>
            <div className="profile-nav-info">
              <span className="profile-nav-name">New Profile</span>
            </div>
          </div>
        )}
      </div>

      <div className="profile-nav-footer">
        <button
          className={`profile-nav-create-btn ${isCreatingNew ? 'creating' : ''}`}
          onClick={onCreateNew}
          disabled={isCreatingNew}
        >
          <Plus size={16} />
          <span>New Profile</span>
        </button>
      </div>
    </motion.nav>
  );
}

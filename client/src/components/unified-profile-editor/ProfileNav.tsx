import { motion } from 'framer-motion';
import { GripVertical, Plus, User } from 'lucide-react';
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
import type { AgentProfile } from '../../lib/api';
import { AgentAvatar } from '../common/AgentAvatar';
import { navVariants } from '../unified-settings/animations/transitions';

interface ProfileNavProps {
  selectedProfileId: string | null;
  isCreatingNew: boolean;
  onSelectProfile: (id: string) => void;
  onCreateNew: () => void;
}

/** One draggable profile row — handle reorders, the rest selects. */
function SortableProfileItem({
  profile,
  isActive,
  onSelect,
}: {
  profile: AgentProfile;
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
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
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
      <div className="profile-nav-avatar">
        <AgentAvatar avatar={profile.avatar} size={15} fill />
      </div>
      <div className="profile-nav-info">
        <span className="profile-nav-name">{profile.name}</span>
        {profile.isDefault && <span className="profile-nav-badge">default</span>}
      </div>
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
      </div>

      <div className="profile-nav-list">
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
                isActive={!isCreatingNew && selectedProfileId === profile.id}
                onSelect={() => onSelectProfile(profile.id)}
              />
            ))}
          </SortableContext>
        </DndContext>

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

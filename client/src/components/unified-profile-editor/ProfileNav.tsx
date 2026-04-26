import { motion } from 'framer-motion';
import { Plus, User } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import { navVariants } from '../unified-settings/animations/transitions';

interface ProfileNavProps {
  selectedProfileId: string | null;
  isCreatingNew: boolean;
  onSelectProfile: (id: string) => void;
  onCreateNew: () => void;
}

export function ProfileNav({
  selectedProfileId,
  isCreatingNew,
  onSelectProfile,
  onCreateNew,
}: ProfileNavProps) {
  const { profiles } = useAgentProfile();

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
        {profiles.map(profile => {
          const AvatarIcon = getAvatarIcon(profile.avatar);
          const isActive = !isCreatingNew && selectedProfileId === profile.id;
          return (
            <button
              key={profile.id}
              className={`profile-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectProfile(profile.id)}
            >
              <div className="profile-nav-avatar">
                <AvatarIcon size={15} />
              </div>
              <div className="profile-nav-info">
                <span className="profile-nav-name">{profile.name}</span>
                {profile.isDefault && (
                  <span className="profile-nav-badge">default</span>
                )}
              </div>
            </button>
          );
        })}

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

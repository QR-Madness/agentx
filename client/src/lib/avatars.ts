/**
 * Avatar utilities — Shared icon mapping for agent profiles
 */

import {
  Sparkles,
  Brain,
  Zap,
  Heart,
  Star,
  Moon,
  Sun,
  Cloud,
  Flame,
  Bot,
  User,
  type LucideIcon,
} from 'lucide-react';

export interface AvatarOption {
  id: string;
  icon: LucideIcon;
  label: string;
}

export const AVATAR_OPTIONS: AvatarOption[] = [
  { id: 'sparkles', icon: Sparkles, label: 'Sparkles' },
  { id: 'brain', icon: Brain, label: 'Brain' },
  { id: 'zap', icon: Zap, label: 'Zap' },
  { id: 'heart', icon: Heart, label: 'Heart' },
  { id: 'star', icon: Star, label: 'Star' },
  { id: 'moon', icon: Moon, label: 'Moon' },
  { id: 'sun', icon: Sun, label: 'Sun' },
  { id: 'cloud', icon: Cloud, label: 'Cloud' },
  { id: 'flame', icon: Flame, label: 'Flame' },
  { id: 'bot', icon: Bot, label: 'Bot' },
  { id: 'user', icon: User, label: 'User' },
];

/**
 * Get the Lucide icon component for an avatar ID
 * Returns Sparkles as default if ID not found
 */
export function getAvatarIcon(avatarId: string | undefined): LucideIcon {
  const option = AVATAR_OPTIONS.find(o => o.id === avatarId);
  return option?.icon ?? Sparkles;
}

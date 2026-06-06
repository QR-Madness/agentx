/**
 * Avatar utilities — shared icon mapping for agent profiles.
 *
 * A curated set of Lucide icons. They're statically imported (and tree-shaken),
 * so even this larger set adds negligible bundle weight — no lazy loading needed
 * at this scale, and `getAvatarIcon` stays a synchronous component lookup that
 * every render site can use directly.
 */

import {
  // originals
  Sparkles, Brain, Zap, Heart, Star, Moon, Sun, Cloud, Flame, Bot, User,
  // tech / cosmos
  Rocket, Atom, Cpu, Orbit, Satellite, Telescope, Compass, Globe, Lightbulb, Infinity,
  // nature
  Leaf, Sprout, TreePine, Mountain, Waves, Droplet, Snowflake, Wind, Feather,
  // creatures
  Bird, Cat, Dog, Fish, Bug, Rabbit, Ghost,
  // craft / play
  Wand2, Palette, Music, Camera, Coffee, Gamepad2, Puzzle, BookOpen, Anchor,
  // emblems
  Crown, Gem, Shield, Swords, Target, Trophy, Key, Bell, Megaphone, Radio, Hexagon, Eye, Smile,
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
  // tech / cosmos
  { id: 'rocket', icon: Rocket, label: 'Rocket' },
  { id: 'atom', icon: Atom, label: 'Atom' },
  { id: 'cpu', icon: Cpu, label: 'Chip' },
  { id: 'orbit', icon: Orbit, label: 'Orbit' },
  { id: 'satellite', icon: Satellite, label: 'Satellite' },
  { id: 'telescope', icon: Telescope, label: 'Telescope' },
  { id: 'compass', icon: Compass, label: 'Compass' },
  { id: 'globe', icon: Globe, label: 'Globe' },
  { id: 'lightbulb', icon: Lightbulb, label: 'Idea' },
  { id: 'infinity', icon: Infinity, label: 'Infinity' },
  // nature
  { id: 'leaf', icon: Leaf, label: 'Leaf' },
  { id: 'sprout', icon: Sprout, label: 'Sprout' },
  { id: 'tree', icon: TreePine, label: 'Tree' },
  { id: 'mountain', icon: Mountain, label: 'Mountain' },
  { id: 'waves', icon: Waves, label: 'Waves' },
  { id: 'droplet', icon: Droplet, label: 'Droplet' },
  { id: 'snowflake', icon: Snowflake, label: 'Snowflake' },
  { id: 'wind', icon: Wind, label: 'Wind' },
  { id: 'feather', icon: Feather, label: 'Feather' },
  // creatures
  { id: 'bird', icon: Bird, label: 'Bird' },
  { id: 'cat', icon: Cat, label: 'Cat' },
  { id: 'dog', icon: Dog, label: 'Dog' },
  { id: 'fish', icon: Fish, label: 'Fish' },
  { id: 'bug', icon: Bug, label: 'Bug' },
  { id: 'rabbit', icon: Rabbit, label: 'Rabbit' },
  { id: 'ghost', icon: Ghost, label: 'Ghost' },
  // craft / play
  { id: 'wand', icon: Wand2, label: 'Wand' },
  { id: 'palette', icon: Palette, label: 'Palette' },
  { id: 'music', icon: Music, label: 'Music' },
  { id: 'camera', icon: Camera, label: 'Camera' },
  { id: 'coffee', icon: Coffee, label: 'Coffee' },
  { id: 'gamepad', icon: Gamepad2, label: 'Gamepad' },
  { id: 'puzzle', icon: Puzzle, label: 'Puzzle' },
  { id: 'book', icon: BookOpen, label: 'Book' },
  { id: 'anchor', icon: Anchor, label: 'Anchor' },
  // emblems
  { id: 'crown', icon: Crown, label: 'Crown' },
  { id: 'gem', icon: Gem, label: 'Gem' },
  { id: 'shield', icon: Shield, label: 'Shield' },
  { id: 'swords', icon: Swords, label: 'Swords' },
  { id: 'target', icon: Target, label: 'Target' },
  { id: 'trophy', icon: Trophy, label: 'Trophy' },
  { id: 'key', icon: Key, label: 'Key' },
  { id: 'bell', icon: Bell, label: 'Bell' },
  { id: 'megaphone', icon: Megaphone, label: 'Megaphone' },
  { id: 'radio', icon: Radio, label: 'Radio' },
  { id: 'hexagon', icon: Hexagon, label: 'Hexagon' },
  { id: 'eye', icon: Eye, label: 'Eye' },
  { id: 'smile', icon: Smile, label: 'Smile' },
];

/**
 * Get the Lucide icon component for an avatar ID
 * Returns Sparkles as default if ID not found
 */
export function getAvatarIcon(avatarId: string | undefined): LucideIcon {
  const option = AVATAR_OPTIONS.find(o => o.id === avatarId);
  return option?.icon ?? Sparkles;
}

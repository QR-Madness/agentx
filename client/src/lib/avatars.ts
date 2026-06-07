/**
 * Avatar utilities — shared icon catalog for agent profiles.
 *
 * A curated, categorized set of Lucide icons (statically imported + tree-shaken, so
 * even this larger set is light). `getAvatarIcon` stays a synchronous lookup that every
 * render site uses directly; the `category`/`keywords` feed the searchable AvatarPicker.
 */

import {
  // tech & cosmos
  Sparkles, Bot, Cpu, Atom, Rocket, Orbit, Satellite, Telescope, Compass, Globe,
  Lightbulb, Infinity, Brain, Code2, Terminal, Database, Server, Network, CircuitBoard, Workflow, Wifi,
  Sunrise, Sunset, Moon, Sun, Star, Cloud, Zap,
  // nature
  Leaf, Sprout, TreePine, Trees, Flower2, Clover, Cherry, Wheat, Mountain, Waves, Droplet,
  Snowflake, Wind, Feather, Flame,
  // creatures
  Bird, Cat, Dog, Fish, Bug, Rabbit, Squirrel, Turtle, Snail, PawPrint, Egg, Shell, Ghost,
  // craft & play
  Wand2, Palette, Paintbrush, Pencil, PenTool, Scissors, Hammer, Music, Mic, Headphones,
  Camera, Film, Gamepad2, Dices, Puzzle, BookOpen, Coffee, Cookie, Anchor,
  // emblems
  Crown, Gem, Shield, Swords, Target, Trophy, Award, Medal, Flag, Castle, Tent, Key,
  // symbols
  Heart, Hexagon, Sigma, Hash, AtSign, Asterisk, Triangle, Circle, Square, Bell, Megaphone, Radio,
  // people & faces
  User, Users, Smile, Laugh, Glasses, Drama, Skull, PersonStanding, Eye,
  type LucideIcon,
} from 'lucide-react';

export type AvatarCategory = 'tech' | 'nature' | 'creatures' | 'craft' | 'emblems' | 'symbols' | 'people';

export interface AvatarOption {
  id: string;
  icon: LucideIcon;
  label: string;
  category: AvatarCategory;
  keywords?: string[];
}

export const AVATAR_CATEGORIES: { id: AvatarCategory; label: string }[] = [
  { id: 'tech', label: 'Tech & Cosmos' },
  { id: 'nature', label: 'Nature' },
  { id: 'creatures', label: 'Creatures' },
  { id: 'craft', label: 'Craft & Play' },
  { id: 'emblems', label: 'Emblems' },
  { id: 'symbols', label: 'Symbols' },
  { id: 'people', label: 'People & Faces' },
];

const A = (
  id: string,
  icon: LucideIcon,
  label: string,
  category: AvatarCategory,
  keywords?: string[],
): AvatarOption => ({ id, icon, label, category, keywords });

export const AVATAR_OPTIONS: AvatarOption[] = [
  // ── Tech & Cosmos ──
  A('sparkles', Sparkles, 'Sparkles', 'tech', ['magic', 'ai', 'shine']),
  A('bot', Bot, 'Bot', 'tech', ['robot', 'agent']),
  A('cpu', Cpu, 'Chip', 'tech', ['processor', 'compute']),
  A('atom', Atom, 'Atom', 'tech', ['science', 'physics']),
  A('rocket', Rocket, 'Rocket', 'tech', ['launch', 'space', 'fast']),
  A('orbit', Orbit, 'Orbit', 'tech', ['space', 'planet']),
  A('satellite', Satellite, 'Satellite', 'tech', ['space', 'signal']),
  A('telescope', Telescope, 'Telescope', 'tech', ['space', 'observe', 'research']),
  A('compass', Compass, 'Compass', 'tech', ['navigate', 'direction']),
  A('globe', Globe, 'Globe', 'tech', ['world', 'earth', 'web']),
  A('lightbulb', Lightbulb, 'Idea', 'tech', ['idea', 'bright', 'think']),
  A('infinity', Infinity, 'Infinity', 'tech', ['loop', 'endless']),
  A('code', Code2, 'Code', 'tech', ['dev', 'program']),
  A('terminal', Terminal, 'Terminal', 'tech', ['shell', 'cli', 'dev']),
  A('database', Database, 'Database', 'tech', ['data', 'store', 'sql']),
  A('server', Server, 'Server', 'tech', ['host', 'backend']),
  A('network', Network, 'Network', 'tech', ['graph', 'nodes', 'connect']),
  A('circuit', CircuitBoard, 'Circuit', 'tech', ['board', 'electronics']),
  A('workflow', Workflow, 'Workflow', 'tech', ['pipeline', 'flow', 'automation']),
  A('wifi', Wifi, 'Wifi', 'tech', ['signal', 'wireless']),
  A('sunrise', Sunrise, 'Sunrise', 'tech', ['dawn', 'morning']),
  A('sunset', Sunset, 'Sunset', 'tech', ['dusk', 'evening']),
  A('moon', Moon, 'Moon', 'tech', ['night', 'lunar']),
  A('sun', Sun, 'Sun', 'tech', ['day', 'bright']),
  A('star', Star, 'Star', 'tech', ['favorite', 'shine']),
  A('cloud', Cloud, 'Cloud', 'tech', ['weather', 'sky']),
  A('zap', Zap, 'Bolt', 'tech', ['fast', 'energy', 'lightning']),

  // ── Nature ──
  A('leaf', Leaf, 'Leaf', 'nature', ['plant', 'green']),
  A('sprout', Sprout, 'Sprout', 'nature', ['grow', 'seedling']),
  A('tree', TreePine, 'Pine', 'nature', ['forest', 'wood']),
  A('trees', Trees, 'Trees', 'nature', ['forest', 'wood']),
  A('flower', Flower2, 'Flower', 'nature', ['bloom', 'petal']),
  A('clover', Clover, 'Clover', 'nature', ['luck', 'leaf']),
  A('cherry', Cherry, 'Cherry', 'nature', ['fruit', 'berry']),
  A('wheat', Wheat, 'Wheat', 'nature', ['grain', 'harvest']),
  A('mountain', Mountain, 'Mountain', 'nature', ['peak', 'climb']),
  A('waves', Waves, 'Waves', 'nature', ['water', 'sea', 'ocean']),
  A('droplet', Droplet, 'Droplet', 'nature', ['water', 'rain']),
  A('snowflake', Snowflake, 'Snowflake', 'nature', ['cold', 'winter']),
  A('wind', Wind, 'Wind', 'nature', ['air', 'breeze']),
  A('feather', Feather, 'Feather', 'nature', ['light', 'write']),
  A('flame', Flame, 'Flame', 'nature', ['fire', 'hot']),

  // ── Creatures ──
  A('bird', Bird, 'Bird', 'creatures', ['fly', 'animal']),
  A('cat', Cat, 'Cat', 'creatures', ['animal', 'pet']),
  A('dog', Dog, 'Dog', 'creatures', ['animal', 'pet']),
  A('fish', Fish, 'Fish', 'creatures', ['animal', 'sea']),
  A('bug', Bug, 'Bug', 'creatures', ['insect', 'debug']),
  A('rabbit', Rabbit, 'Rabbit', 'creatures', ['animal', 'fast']),
  A('squirrel', Squirrel, 'Squirrel', 'creatures', ['animal']),
  A('turtle', Turtle, 'Turtle', 'creatures', ['animal', 'slow']),
  A('snail', Snail, 'Snail', 'creatures', ['animal', 'slow']),
  A('paw', PawPrint, 'Paw', 'creatures', ['animal', 'track']),
  A('egg', Egg, 'Egg', 'creatures', ['hatch', 'new']),
  A('shell', Shell, 'Shell', 'creatures', ['sea', 'spiral']),
  A('ghost', Ghost, 'Ghost', 'creatures', ['spooky', 'spirit']),

  // ── Craft & Play ──
  A('wand', Wand2, 'Wand', 'craft', ['magic', 'enhance']),
  A('palette', Palette, 'Palette', 'craft', ['art', 'color', 'paint']),
  A('paintbrush', Paintbrush, 'Paintbrush', 'craft', ['art', 'paint']),
  A('pencil', Pencil, 'Pencil', 'craft', ['write', 'draw', 'edit']),
  A('pen', PenTool, 'Pen', 'craft', ['design', 'vector']),
  A('scissors', Scissors, 'Scissors', 'craft', ['cut', 'edit']),
  A('hammer', Hammer, 'Hammer', 'craft', ['build', 'tool']),
  A('music', Music, 'Music', 'craft', ['note', 'sound']),
  A('mic', Mic, 'Mic', 'craft', ['voice', 'record', 'speak']),
  A('headphones', Headphones, 'Headphones', 'craft', ['audio', 'listen']),
  A('camera', Camera, 'Camera', 'craft', ['photo', 'vision']),
  A('film', Film, 'Film', 'craft', ['movie', 'video']),
  A('gamepad', Gamepad2, 'Gamepad', 'craft', ['game', 'play']),
  A('dice', Dices, 'Dice', 'craft', ['random', 'game', 'chance']),
  A('puzzle', Puzzle, 'Puzzle', 'craft', ['solve', 'piece']),
  A('book', BookOpen, 'Book', 'craft', ['read', 'knowledge']),
  A('coffee', Coffee, 'Coffee', 'craft', ['drink', 'cafe']),
  A('cookie', Cookie, 'Cookie', 'craft', ['treat', 'snack']),
  A('anchor', Anchor, 'Anchor', 'craft', ['sea', 'stable']),

  // ── Emblems ──
  A('crown', Crown, 'Crown', 'emblems', ['king', 'royal', 'lead']),
  A('gem', Gem, 'Gem', 'emblems', ['diamond', 'jewel', 'value']),
  A('shield', Shield, 'Shield', 'emblems', ['guard', 'secure', 'safe']),
  A('swords', Swords, 'Swords', 'emblems', ['battle', 'fight']),
  A('target', Target, 'Target', 'emblems', ['goal', 'aim', 'focus']),
  A('trophy', Trophy, 'Trophy', 'emblems', ['win', 'award']),
  A('award', Award, 'Award', 'emblems', ['medal', 'prize']),
  A('medal', Medal, 'Medal', 'emblems', ['win', 'rank']),
  A('flag', Flag, 'Flag', 'emblems', ['mark', 'milestone']),
  A('castle', Castle, 'Castle', 'emblems', ['fortress', 'kingdom']),
  A('tent', Tent, 'Tent', 'emblems', ['camp', 'explore']),
  A('key', Key, 'Key', 'emblems', ['unlock', 'access', 'secret']),

  // ── Symbols ──
  A('heart', Heart, 'Heart', 'symbols', ['love', 'like']),
  A('hexagon', Hexagon, 'Hexagon', 'symbols', ['shape']),
  A('sigma', Sigma, 'Sigma', 'symbols', ['sum', 'math']),
  A('hash', Hash, 'Hash', 'symbols', ['tag', 'number']),
  A('at', AtSign, 'At', 'symbols', ['mention', 'email']),
  A('asterisk', Asterisk, 'Asterisk', 'symbols', ['star', 'wildcard']),
  A('triangle', Triangle, 'Triangle', 'symbols', ['shape']),
  A('circle', Circle, 'Circle', 'symbols', ['shape', 'dot']),
  A('square', Square, 'Square', 'symbols', ['shape', 'box']),
  A('bell', Bell, 'Bell', 'symbols', ['notify', 'alert']),
  A('megaphone', Megaphone, 'Megaphone', 'symbols', ['announce', 'loud']),
  A('radio', Radio, 'Radio', 'symbols', ['broadcast', 'signal', 'ambassador']),

  // ── People & Faces ──
  A('user', User, 'User', 'people', ['person', 'profile']),
  A('users', Users, 'Users', 'people', ['team', 'group']),
  A('smile', Smile, 'Smile', 'people', ['happy', 'face']),
  A('laugh', Laugh, 'Laugh', 'people', ['happy', 'joy', 'face']),
  A('glasses', Glasses, 'Glasses', 'people', ['smart', 'read']),
  A('drama', Drama, 'Masks', 'people', ['theatre', 'persona', 'act']),
  A('skull', Skull, 'Skull', 'people', ['danger', 'edgy']),
  A('person', PersonStanding, 'Person', 'people', ['human', 'stand']),
  A('eye', Eye, 'Eye', 'people', ['watch', 'vision', 'see']),
  A('brain', Brain, 'Brain', 'people', ['mind', 'think', 'smart']),
];

/**
 * Get the Lucide icon component for an avatar ID. Returns Sparkles as default if
 * the ID isn't found.
 */
export function getAvatarIcon(avatarId: string | undefined): LucideIcon {
  const option = AVATAR_OPTIONS.find(o => o.id === avatarId);
  return option?.icon ?? Sparkles;
}

/** Find an avatar option by id (for label/category lookups). */
export function getAvatarOption(avatarId: string | undefined): AvatarOption | undefined {
  return AVATAR_OPTIONS.find(o => o.id === avatarId);
}

/** Filter the catalog by a free-text query over id / label / category / keywords. */
export function searchAvatars(query: string): AvatarOption[] {
  const q = query.trim().toLowerCase();
  if (!q) return AVATAR_OPTIONS;
  return AVATAR_OPTIONS.filter(
    o =>
      o.id.includes(q) ||
      o.label.toLowerCase().includes(q) ||
      o.category.includes(q) ||
      (o.keywords ?? []).some(k => k.includes(q)),
  );
}

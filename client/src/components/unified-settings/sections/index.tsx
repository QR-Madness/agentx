/**
 * Section registry and hierarchy for unified settings
 * Note: Using .tsx extension to support JSX syntax for icons
 */

import { lazy } from 'react';
import {
  Server,
  Key,
  Layers,
  Layers3,
  BrainCircuit,
  Sparkles,
  Database,
  Wrench,
  Languages,
  Palette,
  Brain,
  ListTree,
  Globe,
  Users,
  Radio,
  SquareStack,
  Image as ImageIcon,
  Boxes,
  FileText,
  Library,
  ScrollText,
  Search,
  GitMerge,
  Telescope,
} from 'lucide-react';
import { MemoryIcon } from '../../common/MemoryIcon';

export interface Section {
  id: string;
  label: string;
  icon: React.ReactNode;
  component: React.LazyExoticComponent<React.ComponentType<any>>;
  keywords?: string[];
}

export interface CategoryGroup {
  label: string;
  icon: React.ReactNode;
  sections: Section[];
}

export type SectionHierarchy = Record<string, CategoryGroup>;

export const SECTION_HIERARCHY: SectionHierarchy = {
  infrastructure: {
    label: 'Infrastructure',
    icon: <Server size={16} />,
    sections: [
      {
        id: 'providers',
        label: 'Model Providers',
        icon: <Key size={18} />,
        component: lazy(() => import('./ProvidersSection')),
        keywords: ['api key', 'anthropic', 'openai', 'lm studio', 'authentication']
      },
      {
        id: 'models',
        label: 'Model Limits',
        icon: <Layers size={18} />,
        component: lazy(() => import('./ModelsSection')),
        keywords: ['context', 'tokens', 'window', 'output', 'limits']
      },
      {
        id: 'model-roles',
        label: 'Model Roles',
        icon: <Boxes size={18} />,
        component: lazy(() => import('./ModelRolesSection')),
        keywords: ['roles', 'fast utility', 'deep reasoning', 'summarizer', 'cluster', 'utility models', 'effective model']
      },
      {
        id: 'search',
        label: 'Web Search',
        icon: <Globe size={18} />,
        component: lazy(() => import('./SearchSection')),
        keywords: ['search', 'web search', 'tavily', 'brave', 'backend']
      },
      {
        id: 'images',
        label: 'Images',
        icon: <ImageIcon size={18} />,
        component: lazy(() => import('./ImagesSection')),
        keywords: ['image', 'images', 'avatar', 'avatars', 'generate', 'flux', 'openrouter', 'multimodal']
      },
    ]
  },
  intelligence: {
    label: 'Intelligence',
    icon: <Brain size={16} />,
    sections: [
      {
        id: 'planner',
        label: 'Task Planner',
        icon: <ListTree size={18} />,
        component: lazy(() => import('./PlannerSection')),
        keywords: ['plan', 'planner', 'decompose', 'subtask', 'complexity']
      },
      {
        id: 'thinking',
        label: 'Thinking Patterns',
        icon: <BrainCircuit size={18} />,
        component: lazy(() => import('./ThinkingSection')),
        keywords: ['thinking', 'reasoning', 'cot', 'chain of thought', 'reflection', 'step-back', 'self-consistency', 'consensus', 'classifier', 'pattern', 'scaffold']
      },
      {
        id: 'alloy',
        label: 'Agent Teams',
        icon: <Users size={18} />,
        component: lazy(() => import('./AlloySection')),
        keywords: ['delegation', 'delegate', 'alloy', 'team', 'teams', 'multi-agent', 'ad-hoc', 'parallel', 'fan-out']
      },
      {
        id: 'ambassador',
        label: 'Ambassador',
        icon: <Radio size={18} />,
        component: lazy(() => import('./AmbassadorSection')),
        keywords: ['ambassador', 'briefing', 'summarize', 'parallel', 'interpreter', 'cc']
      },
      {
        id: 'research',
        label: 'Research Mode',
        icon: <Telescope size={18} />,
        component: lazy(() => import('./ResearchSection')),
        keywords: ['research', 'deep research', 'report', 'search budget', 'citations', 'sources', 'web_research', 'cost']
      },
    ]
  },
  prompts: {
    label: 'Prompts',
    icon: <ScrollText size={16} />,
    sections: [
      {
        id: 'prompt-stack',
        label: 'System Prompt',
        icon: <SquareStack size={18} />,
        component: lazy(() => import('./SystemPromptSection')),
        keywords: ['system prompt', 'layers', 'stack', 'persona', 'global prompt', 'composer', 'behavior']
      },
      {
        id: 'prompts',
        label: 'Prompt Enhancement',
        icon: <Sparkles size={18} />,
        component: lazy(() => import('./PromptsSection')),
        keywords: ['enhance', 'improve', 'temperature', 'rewrite']
      },
      {
        id: 'prompt-templates',
        label: 'Template Library',
        icon: <Library size={18} />,
        component: lazy(() => import('./PromptTemplatesSection')),
        keywords: ['template', 'templates', 'library', 'snippet', 'reusable', 'prompt library']
      },
      {
        id: 'feature-prompts',
        label: 'Feature Prompts',
        icon: <FileText size={18} />,
        component: lazy(() => import('./FeaturePromptsSection')),
        keywords: ['extraction prompt', 'relevance prompt', 'planner prompt', 'enhancement prompt', 'override', 'defaults', 'diff']
      },
    ]
  },
  memory: {
    label: 'Memory',
    icon: <MemoryIcon size={16} />,
    sections: [
      {
        id: 'memory-overview',
        label: 'Overview',
        icon: <Database size={18} />,
        component: lazy(() => import('./MemoryOverviewSection')),
        keywords: ['storage', 'data', 'retention', 'postgresql', 'neo4j', 'redis']
      },
      {
        id: 'context',
        label: 'Conversation Context',
        icon: <Layers3 size={18} />,
        component: lazy(() => import('./ContextSection')),
        keywords: ['context', 'compaction', 'digest', 'conversation state', 'rolling summary', 'verbatim', 'trajectory', 'compression', 'tool output', 'rehydrate', 'episodic leads', 'window']
      },
      {
        id: 'memory-recall',
        label: 'Recall',
        icon: <Search size={18} />,
        component: lazy(() => import('./MemoryRecallSection')),
        keywords: ['recall', 'retrieval', 'hybrid', 'hyde', 'self-query', 'cross-encoder', 'rerank', 'candidate pool', 'rrf']
      },
      {
        id: 'memory-consolidation',
        label: 'Consolidation',
        icon: <GitMerge size={18} />,
        component: lazy(() => import('./MemoryConsolidationSection')),
        keywords: ['consolidation', 'extraction', 'facts', 'entities', 'contradiction', 'correction', 'jobs', 'intervals', 'procedural']
      },
    ]
  },
  tools: {
    label: 'Tools',
    icon: <Wrench size={16} />,
    sections: [
      // 'tools-browser' removed — now lives in the immersive Toolkit (Phase 18.2).
      {
        id: 'translation',
        label: 'Translation',
        icon: <Languages size={18} />,
        component: lazy(() => import('./TranslationSection')),
        keywords: ['language', 'translate', 'nllb', 'multilingual']
      },
    ]
  },
  interface: {
    label: 'Interface',
    icon: <Palette size={16} />,
    sections: [
      {
        id: 'appearance',
        label: 'Appearance',
        icon: <Palette size={18} />,
        component: lazy(() => import('./AppearanceSection')),
        keywords: ['theme', 'dark', 'light', 'cosmic', 'colors']
      },
    ]
  }
};

/** Helper to get all sections as flat list */
export function getAllSections(): Section[] {
  return Object.values(SECTION_HIERARCHY).flatMap(category => category.sections);
}

/** Helper to find section by ID */
export function findSectionById(id: string): Section | null {
  for (const category of Object.values(SECTION_HIERARCHY)) {
    const section = category.sections.find(s => s.id === id);
    if (section) return section;
  }
  return null;
}

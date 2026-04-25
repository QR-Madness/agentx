/**
 * Section registry and hierarchy for unified settings
 * Note: Using .tsx extension to support JSX syntax for icons
 */

import { lazy } from 'react';
import {
  Server,
  Key,
  Layers,
  Sparkles,
  Database,
  Users,
  FileText,
  Zap,
  Clock,
  Settings,
  Wrench,
  Languages,
  Palette,
  Brain
} from 'lucide-react';

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
        id: 'servers',
        label: 'Backend Servers',
        icon: <Server size={18} />,
        component: lazy(() => import('./ServersSection')),
        keywords: ['api', 'backend', 'connection', 'url']
      },
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
    ]
  },
  intelligence: {
    label: 'Intelligence',
    icon: <Brain size={16} />,
    sections: [
      {
        id: 'prompts',
        label: 'Prompt Enhancement',
        icon: <Sparkles size={18} />,
        component: lazy(() => import('./PromptsSection')),
        keywords: ['enhance', 'improve', 'temperature', 'system prompt']
      },
      {
        id: 'memory-overview',
        label: 'Memory Overview',
        icon: <Database size={18} />,
        component: lazy(() => import('./MemoryOverviewSection')),
        keywords: ['storage', 'data', 'retention', 'postgresql', 'neo4j', 'redis']
      },
      {
        id: 'entities',
        label: 'Entities',
        icon: <Users size={18} />,
        component: lazy(() => import('./EntitiesSection')),
        keywords: ['people', 'organizations', 'semantic', 'knowledge graph']
      },
      {
        id: 'facts',
        label: 'Facts',
        icon: <FileText size={18} />,
        component: lazy(() => import('./FactsSection')),
        keywords: ['knowledge', 'claims', 'confidence', 'semantic memory']
      },
      {
        id: 'strategies',
        label: 'Strategies',
        icon: <Zap size={18} />,
        component: lazy(() => import('./StrategiesSection')),
        keywords: ['procedural', 'patterns', 'tools', 'learning']
      },
      {
        id: 'jobs',
        label: 'Background Jobs',
        icon: <Clock size={18} />,
        component: lazy(() => import('./JobsSection')),
        keywords: ['consolidation', 'scheduled', 'tasks', 'background']
      },
      {
        id: 'memory-settings',
        label: 'Memory Settings',
        icon: <Settings size={18} />,
        component: lazy(() => import('./MemorySettingsSection')),
        keywords: ['consolidation', 'recall', 'extraction', 'hyde', 'self-query']
      },
    ]
  },
  tools: {
    label: 'Tools',
    icon: <Wrench size={16} />,
    sections: [
      {
        id: 'tools-browser',
        label: 'MCP Tools',
        icon: <Wrench size={18} />,
        component: lazy(() => import('./ToolsSection')),
        keywords: ['mcp', 'servers', 'capabilities', 'functions']
      },
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

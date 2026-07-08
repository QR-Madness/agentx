import { request as apiRequest } from './core';
import type { FeaturePromptDefaults, PromptTemplate, PromptTemplateCreate, PromptTemplateUpdate, TemplateTag, TemplateType } from './types';

export const promptTemplatesApi = {
  // === Prompt Templates ===

  async listPromptTemplates(params: {
    type?: TemplateType;
    tag?: string;
    search?: string;
  } = {}): Promise<{ templates: PromptTemplate[]; total: number }> {
    const query = new URLSearchParams();
    if (params.type) query.set('type', params.type);
    if (params.tag) query.set('tag', params.tag);
    if (params.search) query.set('search', params.search);
    const queryString = query.toString();

    const response = await apiRequest<{ templates: Array<{
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }>; total: number }>(`/api/prompts/templates${queryString ? `?${queryString}` : ''}`);

    // Transform snake_case to camelCase
    return {
      templates: response.templates.map(t => ({
        id: t.id,
        name: t.name,
        content: t.content,
        defaultContent: t.default_content,
        tags: t.tags,
        placeholders: t.placeholders,
        type: t.type as TemplateType,
        isBuiltin: t.is_builtin,
        description: t.description,
        hasModifications: t.has_modifications,
        createdAt: t.created_at || '',
        updatedAt: t.updated_at || '',
      })),
      total: response.total,
    };
  },

  async getPromptTemplate(id: string): Promise<{ template: PromptTemplate }> {
    const response = await apiRequest<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/prompts/templates/${encodeURIComponent(id)}`);

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
    };
  },

  async createPromptTemplate(template: PromptTemplateCreate): Promise<{ template: PromptTemplate; message: string }> {
    const response = await apiRequest<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>('/api/prompts/templates', {
      method: 'POST',
      body: JSON.stringify(template),
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  },

  async updatePromptTemplate(id: string, updates: PromptTemplateUpdate): Promise<{ template: PromptTemplate; message: string }> {
    const response = await apiRequest<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>(`/api/prompts/templates/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  },

  async deletePromptTemplate(id: string): Promise<{ deleted: boolean; message: string }> {
    return apiRequest(`/api/prompts/templates/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  },

  async resetPromptTemplate(id: string): Promise<{ template: PromptTemplate; message: string }> {
    const response = await apiRequest<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>(`/api/prompts/templates/${encodeURIComponent(id)}/reset`, {
      method: 'POST',
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  },

  async listPromptTemplateTags(): Promise<{ tags: TemplateTag[]; total: number }> {
    return apiRequest('/api/prompts/templates/tags');
  },

  async enhancePrompt(prompt: string, context?: Array<{ role: string; content: string }>): Promise<{
    enhanced_prompt: string;
    original_length: number;
    enhanced_length: number;
    model: string;
  }> {
    return apiRequest('/api/prompts/enhance', {
      method: 'POST',
      body: JSON.stringify({ prompt, context }),
    });
  },

  /** Shipped defaults for the overridable feature prompts (diff/reset UI). */
  async getFeaturePromptDefaults(): Promise<FeaturePromptDefaults> {
    return apiRequest('/api/prompts/feature-defaults');
  },

  /** Generate a concise conversation title from compact inputs (state + first/last
   *  message, all pre-truncated). Any subset may be provided. */
  async generateTitle(input: { state?: string; first?: string; last?: string }): Promise<{
    title: string;
    model: string;
  }> {
    return apiRequest('/api/prompts/title', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  },
};

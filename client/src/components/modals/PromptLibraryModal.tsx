/**
 * PromptLibraryModal — Browse, search, select, and edit prompt templates
 *
 * Features:
 * - Tag-based filtering sidebar
 * - Search across name, description, content
 * - Template preview with placeholder highlighting
 * - Insert/Select actions
 * - Create, edit, and delete templates
 * - Reset modified templates
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  X,
  Search,
  Tag,
  FileText,
  MessageSquare,
  Code2,
  RefreshCw,
  Plus,
  Check,
  AlertCircle,
  Sparkles,
  Edit3,
  Save,
  Trash2,
} from 'lucide-react';
import { api, type PromptTemplate, type TemplateTag, type TemplateType, type PromptTemplateCreate } from '../../lib/api';
import './PromptLibraryModal.css';

interface PromptLibraryModalProps {
  onClose: () => void;
  onInsert?: (content: string) => void;
  onSelectTemplate?: (templateId: string, content: string) => void;
  mode?: 'insert' | 'select';
  initialTag?: string;
}

const TYPE_ICONS: Record<TemplateType, typeof FileText> = {
  system: Sparkles,
  user: MessageSquare,
  snippet: Code2,
};

const TYPE_LABELS: Record<TemplateType, string> = {
  system: 'System Prompt',
  user: 'User Message',
  snippet: 'Snippet',
};

export function PromptLibraryModal({
  onClose,
  onInsert,
  onSelectTemplate,
  mode = 'insert',
  initialTag,
}: PromptLibraryModalProps) {
  // Data state
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [tags, setTags] = useState<TemplateTag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [selectedTag, setSelectedTag] = useState<string | null>(initialTag || null);
  const [selectedType, setSelectedType] = useState<TemplateType | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Selection state
  const [selectedTemplate, setSelectedTemplate] = useState<PromptTemplate | null>(null);
  const [resetting, setResetting] = useState(false);

  // Edit/Create state
  const [editingTemplate, setEditingTemplate] = useState<PromptTemplate | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Form fields for edit/create
  const [formName, setFormName] = useState('');
  const [formContent, setFormContent] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [formTags, setFormTags] = useState('');
  const [formType, setFormType] = useState<TemplateType>('snippet');

  // Fetch templates and tags
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [templatesRes, tagsRes] = await Promise.all([
        api.listPromptTemplates({
          tag: selectedTag || undefined,
          type: selectedType || undefined,
          search: searchQuery || undefined,
        }),
        api.listPromptTemplateTags(),
      ]);
      setTemplates(templatesRes.templates);
      setTags(tagsRes.tags);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load templates');
    } finally {
      setLoading(false);
    }
  }, [selectedTag, selectedType, searchQuery]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Filter templates by search (client-side for responsiveness)
  const filteredTemplates = useMemo(() => {
    if (!searchQuery.trim()) return templates;

    const query = searchQuery.toLowerCase();
    return templates.filter(
      t =>
        t.name.toLowerCase().includes(query) ||
        (t.description && t.description.toLowerCase().includes(query)) ||
        t.content.toLowerCase().includes(query)
    );
  }, [templates, searchQuery]);

  // Handle template selection
  const handleSelectTemplate = (template: PromptTemplate) => {
    setSelectedTemplate(template);
  };

  // Handle insert action
  const handleInsert = () => {
    if (selectedTemplate && onInsert) {
      onInsert(selectedTemplate.content);
      onClose();
    }
  };

  // Handle select as base action
  const handleSelectAsBase = () => {
    if (selectedTemplate && onSelectTemplate) {
      onSelectTemplate(selectedTemplate.id, selectedTemplate.content);
      onClose();
    }
  };

  // Handle reset to default
  const handleReset = async () => {
    if (!selectedTemplate) return;

    setResetting(true);
    try {
      const { template } = await api.resetPromptTemplate(selectedTemplate.id);
      // Update in list
      setTemplates(prev => prev.map(t => (t.id === template.id ? template : t)));
      setSelectedTemplate(template);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset template');
    } finally {
      setResetting(false);
    }
  };

  // Start editing an existing template
  const startEditing = (template: PromptTemplate) => {
    setEditingTemplate(template);
    setIsCreating(false);
    setFormName(template.name);
    setFormContent(template.content);
    setFormDescription(template.description || '');
    setFormTags(template.tags.join(', '));
    setFormType(template.type);
  };

  // Start creating a new template
  const startCreating = () => {
    setIsCreating(true);
    setEditingTemplate(null);
    setSelectedTemplate(null);
    setFormName('');
    setFormContent('');
    setFormDescription('');
    setFormTags('');
    setFormType('snippet');
  };

  // Cancel edit/create mode
  const cancelEdit = () => {
    setEditingTemplate(null);
    setIsCreating(false);
  };

  // Save template (create or update)
  const handleSave = async () => {
    if (!formName.trim() || !formContent.trim()) {
      setError('Name and content are required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const tagsArray = formTags
        .split(',')
        .map(t => t.trim())
        .filter(t => t.length > 0);

      if (isCreating) {
        const templateData: PromptTemplateCreate = {
          name: formName.trim(),
          content: formContent,
          description: formDescription.trim() || undefined,
          tags: tagsArray.length > 0 ? tagsArray : undefined,
          type: formType,
        };
        const { template } = await api.createPromptTemplate(templateData);
        setSelectedTemplate(template);
      } else if (editingTemplate) {
        const { template } = await api.updatePromptTemplate(editingTemplate.id, {
          name: formName.trim(),
          content: formContent,
          description: formDescription.trim() || undefined,
          tags: tagsArray,
        });
        setSelectedTemplate(template);
      }

      await fetchData();
      cancelEdit();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save template');
    } finally {
      setSaving(false);
    }
  };

  // Delete a template
  const handleDelete = async () => {
    if (!editingTemplate || editingTemplate.isBuiltin) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete "${editingTemplate.name}"? This action cannot be undone.`
    );
    if (!confirmed) return;

    setDeleting(true);
    setError(null);

    try {
      await api.deletePromptTemplate(editingTemplate.id);
      await fetchData();
      cancelEdit();
      setSelectedTemplate(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete template');
    } finally {
      setDeleting(false);
    }
  };

  // Check if we're in edit mode
  const isEditing = isCreating || editingTemplate !== null;

  // Highlight placeholders in content
  const highlightPlaceholders = (content: string) => {
    const parts = content.split(/(\{[^}]+\})/g);
    return parts.map((part, i) => {
      if (part.startsWith('{') && part.endsWith('}')) {
        return (
          <span key={i} className="placeholder-highlight">
            {part}
          </span>
        );
      }
      return part;
    });
  };

  const TypeIcon = selectedTemplate ? TYPE_ICONS[selectedTemplate.type] : FileText;

  return (
    <div className="prompt-library-modal">
      {/* Header */}
      <div className="modal-header">
        <div className="modal-title-group">
          <FileText size={20} />
          <h2>Prompt Library</h2>
        </div>
        <div className="modal-header-actions">
          <button
            className="button-primary"
            onClick={startCreating}
            disabled={isEditing}
          >
            <Plus size={16} />
            New Template
          </button>
          <button className="button-ghost close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Content */}
      <div className="library-content">
        {/* Tag Sidebar */}
        <div className="tag-sidebar">
          <div className="sidebar-header">
            <Tag size={14} />
            <span>Tags</span>
          </div>
          <div className="tag-list">
            <button
              className={`tag-item ${selectedTag === null ? 'active' : ''}`}
              onClick={() => setSelectedTag(null)}
            >
              <span className="tag-name">All</span>
              <span className="tag-count">{templates.length}</span>
            </button>
            {tags.map(tag => (
              <button
                key={tag.name}
                className={`tag-item ${selectedTag === tag.name ? 'active' : ''}`}
                onClick={() => setSelectedTag(tag.name)}
              >
                <span className="tag-name">{tag.name}</span>
                <span className="tag-count">{tag.count}</span>
              </button>
            ))}
          </div>

          {/* Type Filter */}
          <div className="sidebar-header" style={{ marginTop: '1rem' }}>
            <FileText size={14} />
            <span>Type</span>
          </div>
          <div className="tag-list">
            <button
              className={`tag-item ${selectedType === null ? 'active' : ''}`}
              onClick={() => setSelectedType(null)}
            >
              <span className="tag-name">All Types</span>
            </button>
            {(['system', 'user', 'snippet'] as TemplateType[]).map(type => {
              const Icon = TYPE_ICONS[type];
              return (
                <button
                  key={type}
                  className={`tag-item ${selectedType === type ? 'active' : ''}`}
                  onClick={() => setSelectedType(type)}
                >
                  <Icon size={14} />
                  <span className="tag-name">{TYPE_LABELS[type]}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Template List */}
        <div className="template-list-container">
          {/* Search Bar */}
          <div className="search-bar">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Template Grid */}
          <div className="template-list">
            {loading ? (
              <div className="loading-state">
                <RefreshCw size={24} className="spin" />
                <span>Loading templates...</span>
              </div>
            ) : filteredTemplates.length === 0 ? (
              <div className="empty-state">
                <FileText size={32} />
                <span>No templates found</span>
                {searchQuery && (
                  <button
                    className="button-secondary"
                    onClick={() => setSearchQuery('')}
                  >
                    Clear search
                  </button>
                )}
              </div>
            ) : (
              filteredTemplates.map(template => {
                const Icon = TYPE_ICONS[template.type];
                return (
                  <button
                    key={template.id}
                    className={`template-card ${selectedTemplate?.id === template.id ? 'selected' : ''}`}
                    onClick={() => handleSelectTemplate(template)}
                  >
                    <div className="template-card-header">
                      <Icon size={16} />
                      <span className="template-name">{template.name}</span>
                      {template.hasModifications && (
                        <span className="modified-badge" title="Modified from default">
                          Modified
                        </span>
                      )}
                      {template.isBuiltin && (
                        <span className="builtin-badge" title="Built-in template">
                          Built-in
                        </span>
                      )}
                    </div>
                    {template.description && (
                      <p className="template-description">{template.description}</p>
                    )}
                    <div className="template-tags">
                      {template.tags.slice(0, 3).map(tag => (
                        <span key={tag} className="tag-pill">
                          {tag}
                        </span>
                      ))}
                      {template.tags.length > 3 && (
                        <span className="tag-pill more">+{template.tags.length - 3}</span>
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Preview Panel / Edit Form */}
        <div className="preview-panel">
          {isEditing ? (
            /* Edit/Create Form */
            <div className="edit-form">
              <div className="edit-form-header">
                <h3>{isCreating ? 'Create New Template' : 'Edit Template'}</h3>
                {editingTemplate?.isBuiltin && (
                  <span className="builtin-badge">Built-in (edits create a modified copy)</span>
                )}
              </div>

              <div className="edit-form-field">
                <label htmlFor="template-name">Name</label>
                <input
                  id="template-name"
                  type="text"
                  value={formName}
                  onChange={e => setFormName(e.target.value)}
                  placeholder="Template name..."
                  autoFocus
                />
              </div>

              {isCreating && (
                <div className="edit-form-field">
                  <label htmlFor="template-type">Type</label>
                  <select
                    id="template-type"
                    value={formType}
                    onChange={e => setFormType(e.target.value as TemplateType)}
                  >
                    <option value="snippet">Snippet</option>
                    <option value="system">System Prompt</option>
                    <option value="user">User Message</option>
                  </select>
                </div>
              )}

              <div className="edit-form-field">
                <label htmlFor="template-description">Description</label>
                <input
                  id="template-description"
                  type="text"
                  value={formDescription}
                  onChange={e => setFormDescription(e.target.value)}
                  placeholder="Brief description (optional)..."
                />
              </div>

              <div className="edit-form-field">
                <label htmlFor="template-tags">Tags</label>
                <input
                  id="template-tags"
                  type="text"
                  value={formTags}
                  onChange={e => setFormTags(e.target.value)}
                  placeholder="Comma-separated tags..."
                />
                <span className="form-hint">e.g., reasoning, coding, creative</span>
              </div>

              <div className="edit-form-field edit-form-field-content">
                <label htmlFor="template-content">Content</label>
                <textarea
                  id="template-content"
                  value={formContent}
                  onChange={e => setFormContent(e.target.value)}
                  placeholder="Template content... Use {placeholder} for variables."
                  rows={10}
                />
                <span className="form-hint">
                  Use {'{'}placeholder{'}'} syntax for dynamic values
                </span>
              </div>

              <div className="edit-form-actions">
                {editingTemplate && !editingTemplate.isBuiltin && (
                  <button
                    className="button-danger"
                    onClick={handleDelete}
                    disabled={deleting || saving}
                  >
                    <Trash2 size={14} />
                    {deleting ? 'Deleting...' : 'Delete'}
                  </button>
                )}
                <div className="edit-form-actions-right">
                  <button
                    className="button-secondary"
                    onClick={cancelEdit}
                    disabled={saving || deleting}
                  >
                    Cancel
                  </button>
                  <button
                    className="button-primary"
                    onClick={handleSave}
                    disabled={saving || deleting || !formName.trim() || !formContent.trim()}
                  >
                    <Save size={14} />
                    {saving ? 'Saving...' : isCreating ? 'Create' : 'Save'}
                  </button>
                </div>
              </div>
            </div>
          ) : selectedTemplate ? (
            /* Preview Mode */
            <>
              <div className="preview-header">
                <TypeIcon size={16} />
                <span className="preview-title">{selectedTemplate.name}</span>
                <span className="type-badge">{TYPE_LABELS[selectedTemplate.type]}</span>
              </div>

              {selectedTemplate.description && (
                <p className="preview-description">{selectedTemplate.description}</p>
              )}

              <div className="preview-content">
                <pre>{highlightPlaceholders(selectedTemplate.content)}</pre>
              </div>

              {selectedTemplate.placeholders.length > 0 && (
                <div className="preview-placeholders">
                  <span className="placeholders-label">Placeholders:</span>
                  {selectedTemplate.placeholders.map(p => (
                    <code key={p}>{`{${p}}`}</code>
                  ))}
                </div>
              )}

              <div className="preview-actions">
                <button
                  className="button-secondary"
                  onClick={() => startEditing(selectedTemplate)}
                >
                  <Edit3 size={14} />
                  Edit
                </button>
                {selectedTemplate.hasModifications && (
                  <button
                    className="button-secondary"
                    onClick={handleReset}
                    disabled={resetting}
                  >
                    <RefreshCw size={14} className={resetting ? 'spin' : ''} />
                    Reset to Default
                  </button>
                )}
                {mode === 'insert' && onInsert && (
                  <button className="button-primary" onClick={handleInsert}>
                    <Plus size={14} />
                    Insert
                  </button>
                )}
                {mode === 'select' && onSelectTemplate && (
                  <button className="button-primary" onClick={handleSelectAsBase}>
                    <Check size={14} />
                    Use as Base
                  </button>
                )}
              </div>
            </>
          ) : (
            <div className="preview-empty">
              <FileText size={32} />
              <span>Select a template to preview</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

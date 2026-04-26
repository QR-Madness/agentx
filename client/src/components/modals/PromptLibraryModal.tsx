/**
 * PromptLibraryModal — Browse, search, select, and edit prompt templates
 *
 * Two responsive layouts:
 *   wide (≥680px): persistent list on left, preview/edit on right
 *   narrow (<680px): state-machine with slide transitions
 *
 * State machine: browse → preview → edit/create (back returns to preview, not browse)
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Search,
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
  ArrowLeft,
  RotateCcw,
} from 'lucide-react';
import {
  api,
  type PromptTemplate,
  type TemplateTag,
  type TemplateType,
  type PromptTemplateCreate,
} from '../../lib/api';
import './PromptLibraryModal.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface PromptLibraryModalProps {
  onClose: () => void;
  onInsert?: (content: string) => void;
  onSelectTemplate?: (templateId: string, content: string) => void;
  mode?: 'insert' | 'select';
  initialTag?: string;
  variant?: 'modal' | 'panel';
}

type ViewState = 'browse' | 'preview' | 'edit' | 'create';

// ─── Constants ────────────────────────────────────────────────────────────────

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

const WIDE_BREAKPOINT = 680;

const slideVariants = {
  enter: (dir: number) => ({
    x: dir >= 0 ? '60%' : '-60%',
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
    transition: { type: 'spring' as const, damping: 28, stiffness: 320 },
  },
  exit: (dir: number) => ({
    x: dir >= 0 ? '-60%' : '60%',
    opacity: 0,
    transition: { duration: 0.18 },
  }),
};

// ─── Data hook ────────────────────────────────────────────────────────────────

function usePromptLibraryData() {
  const [allTemplates, setAllTemplates] = useState<PromptTemplate[]>([]);
  const [allTags, setAllTags] = useState<TemplateTag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    setError(null);
    try {
      const [tr, tagr] = await Promise.all([
        api.listPromptTemplates({}),
        api.listPromptTemplateTags(),
      ]);
      setAllTemplates(tr.templates);
      setAllTags(tagr.tags);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load templates');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return { allTemplates, allTags, loading, error, refresh: loadAll };
}

// ─── Highlight placeholders ───────────────────────────────────────────────────

function HighlightedContent({ content }: { content: string }) {
  const parts = content.split(/(\{[^}]+\})/g);
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith('{') && part.endsWith('}') ? (
          <span key={i} className="placeholder-highlight">{part}</span>
        ) : (
          part
        )
      )}
    </>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function PromptLibraryModal({
  onClose,
  onInsert,
  onSelectTemplate,
  mode,
  initialTag,
  variant = 'modal',
}: PromptLibraryModalProps) {
  const { allTemplates, allTags, loading, error: dataError, refresh } = usePromptLibraryData();

  // Layout
  const rootRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(true);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      setIsWide(entries[0].contentRect.width >= WIDE_BREAKPOINT);
    });
    obs.observe(el);
    setIsWide(el.getBoundingClientRect().width >= WIDE_BREAKPOINT);
    return () => obs.disconnect();
  }, []);

  // Filter state
  const [activeType, setActiveType] = useState<TemplateType | null>(null);
  const [activeTag, setActiveTag] = useState<string | null>(initialTag ?? null);
  const [query, setQuery] = useState('');
  const [tagsExpanded, setTagsExpanded] = useState(false);

  // View state
  const [viewState, setViewState] = useState<ViewState>('browse');
  const [slideDir, setSlideDir] = useState(1);
  const [cameFromPreview, setCameFromPreview] = useState(false);

  // Selection
  const [selectedTemplate, setSelectedTemplate] = useState<PromptTemplate | null>(null);

  // Edit/create form
  const [formName, setFormName] = useState('');
  const [formContent, setFormContent] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [formTags, setFormTags] = useState('');
  const [formType, setFormType] = useState<TemplateType>('snippet');
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);

  // ── Filtered templates ──────────────────────────────────────────────────────

  const filteredTemplates = useMemo(() => {
    return allTemplates
      .filter(t => !activeType || t.type === activeType)
      .filter(t => !activeTag || t.tags.includes(activeTag))
      .filter(t => {
        if (!query.trim()) return true;
        const q = query.toLowerCase();
        return (
          t.name.toLowerCase().includes(q) ||
          (t.description?.toLowerCase().includes(q) ?? false) ||
          t.content.toLowerCase().includes(q)
        );
      });
  }, [allTemplates, activeType, activeTag, query]);

  // ── Navigation helpers ──────────────────────────────────────────────────────

  const goTo = (next: ViewState, dir: number) => {
    setSlideDir(dir);
    setViewState(next);
  };

  const selectCard = (template: PromptTemplate) => {
    setSelectedTemplate(template);
    if (!isWide) goTo('preview', 1);
  };

  const openEdit = (template: PromptTemplate) => {
    setFormName(template.name);
    setFormContent(template.content);
    setFormDescription(template.description ?? '');
    setFormTags(template.tags.join(', '));
    setFormType(template.type);
    setFormError(null);
    setCameFromPreview(true);
    goTo('edit', 1);
  };

  const openCreate = () => {
    setFormName('');
    setFormContent('');
    setFormDescription('');
    setFormTags('');
    setFormType('snippet');
    setFormError(null);
    setCameFromPreview(false);
    goTo('create', 1);
  };

  const goBack = () => {
    if (viewState === 'preview') {
      goTo('browse', -1);
    } else if (viewState === 'edit' || viewState === 'create') {
      if (cameFromPreview && selectedTemplate) {
        goTo('preview', -1);
      } else {
        goTo('browse', -1);
      }
    }
  };

  // ── Actions ─────────────────────────────────────────────────────────────────

  const handleInsert = () => {
    if (selectedTemplate && onInsert) {
      onInsert(selectedTemplate.content);
      onClose();
    }
  };

  const handleSelectAsBase = () => {
    if (selectedTemplate && onSelectTemplate) {
      onSelectTemplate(selectedTemplate.id, selectedTemplate.content);
      onClose();
    }
  };

  const handleReset = async () => {
    if (!selectedTemplate) return;
    setResetting(true);
    try {
      const { template } = await api.resetPromptTemplate(selectedTemplate.id);
      setSelectedTemplate(template);
      await refresh();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to reset');
    } finally {
      setResetting(false);
    }
  };

  const handleSave = async () => {
    if (!formName.trim() || !formContent.trim()) {
      setFormError('Name and content are required');
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      const tagsArray = formTags
        .split(',')
        .map(t => t.trim())
        .filter(Boolean);

      let saved: PromptTemplate;
      if (viewState === 'create') {
        const data: PromptTemplateCreate = {
          name: formName.trim(),
          content: formContent,
          description: formDescription.trim() || undefined,
          tags: tagsArray.length > 0 ? tagsArray : undefined,
          type: formType,
        };
        const res = await api.createPromptTemplate(data);
        saved = res.template;
      } else {
        // edit — selectedTemplate is the one being edited
        const res = await api.updatePromptTemplate(selectedTemplate!.id, {
          name: formName.trim(),
          content: formContent,
          description: formDescription.trim() || undefined,
          tags: tagsArray,
        });
        saved = res.template;
      }

      await refresh();
      setSelectedTemplate(saved);
      goTo('preview', -1);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedTemplate || selectedTemplate.isBuiltin) return;
    if (!window.confirm(`Delete "${selectedTemplate.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    setFormError(null);
    try {
      await api.deletePromptTemplate(selectedTemplate.id);
      await refresh();
      setSelectedTemplate(null);
      goTo('browse', -1);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to delete');
    } finally {
      setDeleting(false);
    }
  };

  // ── Tag chip overflow ────────────────────────────────────────────────────────

  const TAG_SHOW_LIMIT = 6;
  const visibleTags = tagsExpanded ? allTags : allTags.slice(0, TAG_SHOW_LIMIT);
  const hiddenCount = allTags.length - TAG_SHOW_LIMIT;

  // ── Render helpers ───────────────────────────────────────────────────────────

  const ModeChip = () => {
    if (!mode) return null;
    const label = mode === 'insert' ? 'Inserting into system prompt' : 'Selecting base template';
    return <span className="plm-mode-chip">✦ {label}</span>;
  };

  const FilterBar = () => (
    <div className="plm-filter-bar">
      <div className="plm-search">
        <Search size={14} className="plm-search-icon" />
        <input
          type="text"
          placeholder="Search templates…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          className="plm-search-input"
        />
        {query && (
          <button className="plm-search-clear" onClick={() => setQuery('')}>×</button>
        )}
      </div>
      <div className="plm-type-pills">
        {(['all', 'system', 'user', 'snippet'] as const).map(t => {
          const isAll = t === 'all';
          const active = isAll ? activeType === null : activeType === t;
          const Icon = isAll ? null : TYPE_ICONS[t];
          return (
            <button
              key={t}
              className={`plm-type-pill ${active ? 'active' : ''}`}
              onClick={() => setActiveType(isAll ? null : t)}
            >
              {Icon && <Icon size={12} />}
              {isAll ? 'All' : TYPE_LABELS[t]}
            </button>
          );
        })}
      </div>
      {allTags.length > 0 && (
        <div className="plm-tag-chips">
          <button
            className={`plm-tag-chip ${activeTag === null ? 'active' : ''}`}
            onClick={() => setActiveTag(null)}
          >
            All Tags
          </button>
          {visibleTags.map(tag => (
            <button
              key={tag.name}
              className={`plm-tag-chip ${activeTag === tag.name ? 'active' : ''}`}
              onClick={() => setActiveTag(tag.name)}
            >
              {tag.name}
              <span className="plm-tag-count">{tag.count}</span>
            </button>
          ))}
          {!tagsExpanded && hiddenCount > 0 && (
            <button className="plm-tag-chip plm-tag-more" onClick={() => setTagsExpanded(true)}>
              +{hiddenCount} more
            </button>
          )}
          {tagsExpanded && hiddenCount > 0 && (
            <button className="plm-tag-chip plm-tag-more" onClick={() => setTagsExpanded(false)}>
              show less
            </button>
          )}
        </div>
      )}
    </div>
  );

  const TemplateList = () => (
    <div className="plm-list-scroll">
      {loading ? (
        <div className="plm-state-center">
          <RefreshCw size={24} className="spin" />
          <span>Loading…</span>
        </div>
      ) : filteredTemplates.length === 0 ? (
        <div className="plm-state-center">
          <FileText size={28} />
          <span>No templates found</span>
          {query && (
            <button className="plm-btn-ghost" onClick={() => setQuery('')}>Clear search</button>
          )}
        </div>
      ) : (
        <div className="plm-card-grid">
          {filteredTemplates.map(template => {
            const Icon = TYPE_ICONS[template.type];
            const isSelected = selectedTemplate?.id === template.id;
            return (
              <button
                key={template.id}
                className={`plm-card ${isSelected ? 'selected' : ''}`}
                onClick={() => selectCard(template)}
              >
                <div className="plm-card-header">
                  <Icon size={14} />
                  <span className="plm-card-name">{template.name}</span>
                  <div className="plm-card-badges">
                    {template.isBuiltin && <span className="plm-badge plm-badge-builtin">Built-in</span>}
                    {template.hasModifications && <span className="plm-badge plm-badge-modified">Modified</span>}
                  </div>
                </div>
                {template.description && (
                  <p className="plm-card-desc">{template.description}</p>
                )}
                {template.tags.length > 0 && (
                  <div className="plm-card-tags">
                    {template.tags.slice(0, 3).map(tag => (
                      <span key={tag} className="plm-mini-tag">{tag}</span>
                    ))}
                    {template.tags.length > 3 && (
                      <span className="plm-mini-tag">+{template.tags.length - 3}</span>
                    )}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );

  const PreviewPanel = () => {
    if (!selectedTemplate) {
      return (
        <div className="plm-detail-empty">
          <FileText size={32} />
          <span>Select a template to preview</span>
        </div>
      );
    }
    const TypeIcon = TYPE_ICONS[selectedTemplate.type];
    return (
      <div className="plm-preview">
        <div className="plm-preview-header">
          <TypeIcon size={15} />
          <span className="plm-preview-name">{selectedTemplate.name}</span>
          <span className="plm-type-badge">{TYPE_LABELS[selectedTemplate.type]}</span>
          {selectedTemplate.isBuiltin && <span className="plm-badge plm-badge-builtin">Built-in</span>}
          {selectedTemplate.hasModifications && <span className="plm-badge plm-badge-modified">Modified</span>}
        </div>
        {selectedTemplate.description && (
          <p className="plm-preview-desc">{selectedTemplate.description}</p>
        )}
        <div className="plm-preview-content">
          <pre><HighlightedContent content={selectedTemplate.content} /></pre>
        </div>
        {selectedTemplate.placeholders.length > 0 && (
          <div className="plm-placeholders">
            <span className="plm-placeholders-label">Placeholders:</span>
            {selectedTemplate.placeholders.map(p => (
              <code key={p}>{`{${p}}`}</code>
            ))}
          </div>
        )}
        <div className="plm-preview-actions">
          <button className="plm-btn-secondary" onClick={() => openEdit(selectedTemplate)}>
            <Edit3 size={13} />
            Edit
          </button>
          {selectedTemplate.hasModifications && (
            <button
              className="plm-btn-secondary"
              onClick={handleReset}
              disabled={resetting}
            >
              <RotateCcw size={13} className={resetting ? 'spin' : ''} />
              Reset
            </button>
          )}
          <div className="plm-preview-actions-right">
            {mode === 'insert' && onInsert && (
              <button className="plm-btn-primary" onClick={handleInsert}>
                <Plus size={13} />
                Insert
              </button>
            )}
            {mode === 'select' && onSelectTemplate && (
              <button className="plm-btn-primary" onClick={handleSelectAsBase}>
                <Check size={13} />
                Use as Base
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  const EditForm = () => {
    const isCreate = viewState === 'create';
    const isBuiltinEdit = !isCreate && selectedTemplate?.isBuiltin;
    return (
      <div className="plm-edit-form">
        <div className="plm-edit-header">
          <h3>{isCreate ? 'New Template' : 'Edit Template'}</h3>
          {isBuiltinEdit && (
            <span className="plm-badge plm-badge-builtin">Edits create a modified copy</span>
          )}
        </div>
        {formError && (
          <div className="plm-form-error">
            <AlertCircle size={13} />
            {formError}
          </div>
        )}
        <div className="plm-edit-fields">
          <div className="plm-field">
            <label>Name</label>
            <input
              type="text"
              value={formName}
              onChange={e => setFormName(e.target.value)}
              placeholder="Template name…"
              autoFocus
            />
          </div>
          {isCreate && (
            <div className="plm-field">
              <label>Type</label>
              <select value={formType} onChange={e => setFormType(e.target.value as TemplateType)}>
                <option value="snippet">Snippet</option>
                <option value="system">System Prompt</option>
                <option value="user">User Message</option>
              </select>
            </div>
          )}
          <div className="plm-field">
            <label>Description</label>
            <input
              type="text"
              value={formDescription}
              onChange={e => setFormDescription(e.target.value)}
              placeholder="Brief description (optional)…"
            />
          </div>
          <div className="plm-field">
            <label>Tags</label>
            <input
              type="text"
              value={formTags}
              onChange={e => setFormTags(e.target.value)}
              placeholder="Comma-separated: reasoning, coding…"
            />
          </div>
          <div className="plm-field plm-field-content">
            <label>Content</label>
            <textarea
              value={formContent}
              onChange={e => setFormContent(e.target.value)}
              placeholder="Template content… Use {placeholder} for variables."
            />
            <span className="plm-hint">Use {'{'}placeholder{'}'} syntax for dynamic values</span>
          </div>
        </div>
        <div className="plm-edit-actions">
          {!isCreate && selectedTemplate && !selectedTemplate.isBuiltin && (
            <button
              className="plm-btn-danger"
              onClick={handleDelete}
              disabled={deleting || saving}
            >
              <Trash2 size={13} />
              {deleting ? 'Deleting…' : 'Delete'}
            </button>
          )}
          <div className="plm-edit-actions-right">
            <button
              className="plm-btn-secondary"
              onClick={goBack}
              disabled={saving || deleting}
            >
              Cancel
            </button>
            <button
              className="plm-btn-primary"
              onClick={handleSave}
              disabled={saving || deleting || !formName.trim() || !formContent.trim()}
            >
              <Save size={13} />
              {saving ? 'Saving…' : isCreate ? 'Create' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  // ── Layout ───────────────────────────────────────────────────────────────────

  const showingDetail = viewState === 'preview' || viewState === 'edit' || viewState === 'create';

  return (
    <div
      ref={rootRef}
      className={`plm-root plm-variant-${variant} ${isWide ? 'plm-wide' : 'plm-narrow'}`}
    >
      {/* Header */}
      <div className="plm-header">
        {!isWide && showingDetail ? (
          <button className="plm-back-btn" onClick={goBack}>
            <ArrowLeft size={15} />
            Back
          </button>
        ) : (
          <div className="plm-header-title">
            <FileText size={16} />
            <span>Prompt Library</span>
          </div>
        )}
        <ModeChip />
        <div className="plm-header-right">
          <button
            className="plm-btn-primary plm-btn-sm"
            onClick={openCreate}
            disabled={viewState === 'create'}
          >
            <Plus size={14} />
            New
          </button>
        </div>
      </div>

      {dataError && (
        <div className="plm-data-error">
          <AlertCircle size={13} />
          {dataError}
        </div>
      )}

      {/* Wide layout: persistent list + detail panel */}
      {isWide ? (
        <div className="plm-wide-body">
          <div className="plm-wide-list">
            <FilterBar />
            <TemplateList />
          </div>
          <div className="plm-wide-detail">
            <AnimatePresence mode="wait">
              {(viewState === 'edit' || viewState === 'create') ? (
                <motion.div
                  key="edit"
                  className="plm-detail-fill"
                  initial={{ opacity: 0, x: 24 }}
                  animate={{ opacity: 1, x: 0, transition: { type: 'spring', damping: 28, stiffness: 320 } }}
                  exit={{ opacity: 0, x: 24, transition: { duration: 0.15 } }}
                >
                  <EditForm />
                </motion.div>
              ) : (
                <motion.div
                  key={selectedTemplate?.id ?? 'empty'}
                  className="plm-detail-fill"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1, transition: { duration: 0.2 } }}
                  exit={{ opacity: 0, transition: { duration: 0.1 } }}
                >
                  <PreviewPanel />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      ) : (
        /* Narrow layout: slide state machine */
        <div className="plm-narrow-body">
          <AnimatePresence mode="wait" custom={slideDir}>
            {viewState === 'browse' ? (
              <motion.div
                key="browse"
                className="plm-narrow-view"
                custom={slideDir}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
              >
                <FilterBar />
                <TemplateList />
              </motion.div>
            ) : viewState === 'preview' ? (
              <motion.div
                key="preview"
                className="plm-narrow-view"
                custom={slideDir}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
              >
                <PreviewPanel />
              </motion.div>
            ) : (
              <motion.div
                key="edit"
                className="plm-narrow-view"
                custom={slideDir}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
              >
                <EditForm />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

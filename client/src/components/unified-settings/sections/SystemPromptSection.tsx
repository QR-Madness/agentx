/**
 * SystemPromptSection — the block-stack editor for the global system prompt.
 *
 * Two-pane composer: a draggable stack of editable layer cards (left) beside a
 * live composed preview (right). Built-in layers ship a default that can be
 * overridden per-layer; edits autosave (debounced) and persist durably via the
 * Phase-2 layer API. The agent's actual global prompt is `composed` here.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { restrictToVerticalAxis, restrictToParentElement } from '@dnd-kit/modifiers';
import { SquareStack, Plus, RefreshCw, Eye, Library } from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import type { PromptLayer } from '../../../lib/api/types';
import { composeStack, effectiveContent } from '../../../lib/promptStack';
import { Button, SectionHeader } from '../../ui';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../ui/Dialog';
import { PromptLibraryModal } from '../../modals/PromptLibraryModal';
import { LayerCard } from './prompt-stack/LayerCard';
import { ComposedPreview } from './prompt-stack/ComposedPreview';
import { LayerDiffModal } from './prompt-stack/LayerDiffModal';
import './prompt-stack/PromptStack.css';

const AUTOSAVE_MS = 600;

interface DiffTarget {
  layer: PromptLayer;
  mode: 'update' | 'edited';
}

function sortLayers(layers: PromptLayer[]): PromptLayer[] {
  return [...layers].sort((a, b) => a.order - b.order);
}

export default function SystemPromptSection() {
  const { notifyError } = useNotify();
  const [layers, setLayers] = useState<PromptLayer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [diffTarget, setDiffTarget] = useState<DiffTarget | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);

  const saveTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const composed = useMemo(() => composeStack(layers), [layers]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const { layers: fetched } = await api.listPromptLayers();
      setLayers(sortLayers(fetched));
    } catch (err) {
      setError(true);
      notifyError(err, 'Failed to load the prompt stack');
    } finally {
      setLoading(false);
    }
  }, [notifyError]);

  useEffect(() => {
    void load();
    const timers = saveTimers.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, [load]);

  const patchLocal = useCallback((id: string, patch: Partial<PromptLayer>) => {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));
  }, []);

  // --- content edit (instant local + debounced persist) ---
  const handleContentChange = useCallback(
    (id: string, content: string) => {
      patchLocal(id, { override: content });
      const timers = saveTimers.current;
      const existing = timers.get(id);
      if (existing) clearTimeout(existing);
      timers.set(
        id,
        setTimeout(async () => {
          timers.delete(id);
          try {
            const { layer } = await api.updatePromptLayer(id, { content });
            // Only reconcile if the user hasn't typed past what we just saved.
            setLayers((prev) =>
              prev.map((l) => (l.id === id && l.override === content ? { ...layer } : l))
            );
          } catch (err) {
            notifyError(err, 'Failed to save layer');
            void load();
          }
        }, AUTOSAVE_MS)
      );
    },
    [patchLocal, notifyError, load]
  );

  const handleToggleEnabled = useCallback(
    async (id: string, enabled: boolean) => {
      patchLocal(id, { enabled }); // optimistic
      try {
        await api.updatePromptLayer(id, { enabled });
      } catch (err) {
        notifyError(err, 'Failed to toggle layer');
        void load();
      }
    },
    [patchLocal, notifyError, load]
  );

  const handleReorder = useCallback(
    async (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over || active.id === over.id) return;
      const ids = layers.map((l) => l.id);
      const from = ids.indexOf(String(active.id));
      const to = ids.indexOf(String(over.id));
      if (from < 0 || to < 0) return;
      const reordered = arrayMove(layers, from, to).map((l, i) => ({ ...l, order: i * 10 }));
      setLayers(reordered); // optimistic, with refreshed local order
      try {
        await api.reorderPromptLayers(reordered.map((l) => l.id));
      } catch (err) {
        notifyError(err, 'Failed to reorder layers');
        void load();
      }
    },
    [layers, notifyError, load]
  );

  const handleAdd = useCallback(async () => {
    try {
      const { layer } = await api.createPromptLayer('New layer');
      setLayers((prev) => sortLayers([...prev, layer]));
      setExpandedId(layer.id);
    } catch (err) {
      notifyError(err, 'Failed to add layer');
    }
  }, [notifyError]);

  const handleInsertFromLibrary = useCallback(
    async (content: string, name?: string) => {
      try {
        const { layer } = await api.createPromptLayer(name?.trim() || 'Snippet', content);
        setLayers((prev) => sortLayers([...prev, layer]));
        setExpandedId(layer.id);
      } catch (err) {
        notifyError(err, 'Failed to insert snippet as a layer');
      }
    },
    [notifyError]
  );

  const handleEnhance = useCallback(
    async (content: string): Promise<string | null> => {
      try {
        const { enhanced_prompt } = await api.enhancePrompt(content);
        return enhanced_prompt;
      } catch (err) {
        notifyError(err, 'Failed to enhance layer');
        return null;
      }
    },
    [notifyError]
  );

  const handleDelete = useCallback(
    async (id: string) => {
      const prev = layers;
      setLayers((cur) => cur.filter((l) => l.id !== id)); // optimistic
      try {
        await api.deletePromptLayer(id);
      } catch (err) {
        notifyError(err, 'Failed to delete layer');
        setLayers(prev);
      }
    },
    [layers, notifyError]
  );

  const handleReset = useCallback(
    async (id: string) => {
      try {
        const { layer } = await api.resetPromptLayer(id);
        patchLocal(id, { ...layer });
        setDiffTarget(null);
      } catch (err) {
        notifyError(err, 'Failed to reset layer');
      }
    },
    [patchLocal, notifyError]
  );

  const handleAcknowledge = useCallback(
    async (id: string) => {
      try {
        const { layer } = await api.acknowledgePromptLayer(id);
        patchLocal(id, { ...layer });
        setDiffTarget(null);
      } catch (err) {
        notifyError(err, 'Failed to acknowledge update');
      }
    },
    [patchLocal, notifyError]
  );

  const handleLoadDefault = useCallback(
    (layer: PromptLayer) => {
      // Merge-assist: seed the editor with the new default, persist as the override.
      handleContentChange(layer.id, layer.default ?? '');
      setExpandedId(layer.id);
      setDiffTarget(null);
    },
    [handleContentChange]
  );

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  if (loading) {
    return (
      <div className="settings-section fade-in">
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading the prompt stack…</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="settings-section fade-in">
        <SectionHeader icon={<SquareStack size={20} />} title="System Prompt" />
        <Button variant="secondary" onClick={() => void load()}>
          <RefreshCw size={16} /> Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="settings-section fade-in prompt-stack">
      <SectionHeader
        icon={<SquareStack size={20} />}
        title="System Prompt"
        description="Compose your agent's global system prompt from editable layers. Built-in layers keep their shipped default until you override them; your edits persist and are never overwritten by updates."
        actions={
          <Button
            variant="secondary"
            size="sm"
            className="prompt-stack__preview-toggle"
            onClick={() => setPreviewOpen(true)}
          >
            <Eye size={14} /> Preview
          </Button>
        }
      />

      <div className="prompt-stack__panes">
        <div className="prompt-stack__list">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            modifiers={[restrictToVerticalAxis, restrictToParentElement]}
            onDragEnd={handleReorder}
          >
            <SortableContext items={layers.map((l) => l.id)} strategy={verticalListSortingStrategy}>
              {layers.map((layer) => (
                <LayerCard
                  key={layer.id}
                  layer={layer}
                  expanded={expandedId === layer.id}
                  onToggleExpand={() => setExpandedId((cur) => (cur === layer.id ? null : layer.id))}
                  onContentChange={handleContentChange}
                  onToggleEnabled={handleToggleEnabled}
                  onReset={handleReset}
                  onDelete={handleDelete}
                  onViewDiff={(l) =>
                    setDiffTarget({ layer: l, mode: l.update_available ? 'update' : 'edited' })
                  }
                  onEnhance={handleEnhance}
                />
              ))}
            </SortableContext>
          </DndContext>

          <div className="prompt-stack__footer">
            <Button variant="ghost" size="sm" onClick={() => void handleAdd()}>
              <Plus size={16} /> Add layer
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setLibraryOpen(true)}>
              <Library size={16} /> Insert from library
            </Button>
          </div>
        </div>

        <div className="prompt-stack__preview-col">
          <ComposedPreview composed={composed} />
        </div>
      </div>

      {/* Narrow-width / mobile preview */}
      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Composed preview</DialogTitle>
          </DialogHeader>
          <div className="px-6 pb-6 pt-2">
            <ComposedPreview composed={composed} />
          </div>
        </DialogContent>
      </Dialog>

      {/* Insert-from-library (snippet → custom layer) */}
      <Dialog open={libraryOpen} onOpenChange={setLibraryOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Insert from library</DialogTitle>
          </DialogHeader>
          <div className="px-4 pb-4 pt-2" style={{ height: '70vh' }}>
            <PromptLibraryModal
              variant="panel"
              mode="insert"
              onClose={() => setLibraryOpen(false)}
              onInsert={(content, name) => void handleInsertFromLibrary(content, name)}
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* Diff / update-resolution modal */}
      {diffTarget && (
        <LayerDiffModal
          open
          onOpenChange={(open) => !open && setDiffTarget(null)}
          title={
            diffTarget.mode === 'update'
              ? `Update available — ${diffTarget.layer.title}`
              : `Your edits — ${diffTarget.layer.title}`
          }
          {...(diffTarget.mode === 'update'
            ? {
                leftLabel: 'Your version',
                leftText: effectiveContent(diffTarget.layer),
                rightLabel: 'New default',
                rightText: diffTarget.layer.default ?? '',
                onKeep: () => void handleAcknowledge(diffTarget.layer.id),
                onAdopt: () => void handleReset(diffTarget.layer.id),
                onLoadDefault: () => handleLoadDefault(diffTarget.layer),
              }
            : {
                leftLabel: 'Default',
                leftText: diffTarget.layer.default ?? '',
                rightLabel: 'Your version',
                rightText: effectiveContent(diffTarget.layer),
                onAdopt: () => void handleReset(diffTarget.layer.id),
              })}
        />
      )}
    </div>
  );
}

/**
 * LayerCard — one draggable block in the prompt stack. Collapsed: handle, title,
 * kind badge, status dots, enable switch, chevron. Expanded: inline editor with
 * (debounced, by the parent) autosave, token count, and contextual actions.
 */

import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { motion, AnimatePresence } from 'framer-motion';
import { GripVertical, ChevronDown, RotateCcw, GitCompare, Trash2 } from 'lucide-react';
import type { PromptLayer } from '../../../../lib/api/types';
import { effectiveContent, estimateTokens, isModified } from '../../../../lib/promptStack';
import { Badge, Button, Switch } from '../../../ui';
import { Textarea } from '../../../ui/Field';

interface LayerCardProps {
  layer: PromptLayer;
  expanded: boolean;
  onToggleExpand: () => void;
  onContentChange: (id: string, content: string) => void;
  onToggleEnabled: (id: string, enabled: boolean) => void;
  onReset: (id: string) => void;
  onDelete: (id: string) => void;
  onViewDiff: (layer: PromptLayer) => void;
}

export function LayerCard({
  layer,
  expanded,
  onToggleExpand,
  onContentChange,
  onToggleEnabled,
  onReset,
  onDelete,
  onViewDiff,
}: LayerCardProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: layer.id,
  });

  const isBuiltin = layer.kind === 'builtin';
  const modified = isModified(layer);
  const content = effectiveContent(layer);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="layer-card"
      data-dragging={isDragging || undefined}
      data-disabled={!layer.enabled || undefined}
    >
      <div className="layer-card__head">
        <button
          type="button"
          className="layer-card__handle"
          aria-label={`Reorder ${layer.title}`}
          {...attributes}
          {...listeners}
        >
          <GripVertical size={16} />
        </button>

        <button
          type="button"
          className="layer-card__title-btn"
          onClick={onToggleExpand}
          aria-expanded={expanded}
        >
          <ChevronDown size={16} className="layer-card__chevron" data-open={expanded || undefined} />
          <span className="layer-card__title">{layer.title}</span>
          <Badge variant={isBuiltin ? 'neutral' : 'accent'} size="sm">
            {isBuiltin ? 'Built-in' : 'Custom'}
          </Badge>
          <span className="layer-card__dots">
            {modified && (
              <span className="layer-card__dot layer-card__dot--modified" title="Edited — differs from the default">
                ● edited
              </span>
            )}
            {layer.update_available && (
              <span className="layer-card__dot layer-card__dot--update" title="A new default is available">
                ▲ update
              </span>
            )}
          </span>
        </button>

        <div className="layer-card__controls">
          <Switch
            checked={layer.enabled}
            onCheckedChange={(v) => onToggleEnabled(layer.id, v)}
            aria-label={`${layer.enabled ? 'Disable' : 'Enable'} ${layer.title}`}
          />
        </div>
      </div>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            className="layer-card__body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: 'easeInOut' }}
          >
            <div className="layer-card__body-inner">
              <Textarea
                className="layer-card__editor"
                value={content}
                spellCheck={false}
                onChange={(e) => onContentChange(layer.id, e.target.value)}
                placeholder="Layer content…"
              />
              <div className="layer-card__editor-meta">
                <span className="layer-card__count" title="Approximate">
                  ~{estimateTokens(content)} tokens · {content.length} chars
                </span>
                <div className="layer-card__actions">
                  {layer.update_available && (
                    <Button variant="secondary" size="sm" onClick={() => onViewDiff(layer)}>
                      <GitCompare size={14} /> Review update
                    </Button>
                  )}
                  {isBuiltin && modified && !layer.update_available && (
                    <Button variant="ghost" size="sm" onClick={() => onViewDiff(layer)}>
                      <GitCompare size={14} /> View diff
                    </Button>
                  )}
                  {isBuiltin && modified && (
                    <Button variant="ghost" size="sm" onClick={() => onReset(layer.id)}>
                      <RotateCcw size={14} /> Reset
                    </Button>
                  )}
                  {!isBuiltin && (
                    <Button variant="danger" size="sm" onClick={() => onDelete(layer.id)}>
                      <Trash2 size={14} /> Delete
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

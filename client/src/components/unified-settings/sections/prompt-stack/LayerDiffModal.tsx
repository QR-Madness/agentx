/**
 * LayerDiffModal — shows the diff between two versions of a built-in layer and
 * offers resolution actions. Two cases:
 *   • update available: left = your version, right = the new shipped default
 *     → Keep mine (acknowledge) / Adopt default (reset) / Load default into editor.
 *   • edited (no update): left = default, right = your version → Reset / Close.
 */

import { useMemo } from 'react';
import { diffLines } from 'diff';
import { cn } from '../../../../lib/utils';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  Button,
} from '../../../ui';
import './LayerDiffModal.css';

interface LayerDiffModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  leftLabel: string;
  leftText: string;
  rightLabel: string;
  rightText: string;
  /** Keep the user's override, mark the new default as seen (acknowledge). */
  onKeep?: () => void;
  /** Drop the override, adopt the shipped default (reset). */
  onAdopt?: () => void;
  /** Seed the editor with the new default to hand-merge (merge-assist). */
  onLoadDefault?: () => void;
}

interface DiffRow {
  text: string;
  kind: 'add' | 'del' | 'same';
}

function buildRows(leftText: string, rightText: string): DiffRow[] {
  const parts = diffLines(leftText, rightText);
  const rows: DiffRow[] = [];
  for (const part of parts) {
    const kind: DiffRow['kind'] = part.added ? 'add' : part.removed ? 'del' : 'same';
    // diffLines keeps trailing newlines; split into displayable lines without empties at the tail.
    const lines = part.value.split('\n');
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    for (const line of lines) rows.push({ text: line, kind });
  }
  return rows;
}

export function LayerDiffModal({
  open,
  onOpenChange,
  title,
  leftLabel,
  leftText,
  rightLabel,
  rightText,
  onKeep,
  onAdopt,
  onLoadDefault,
}: LayerDiffModalProps) {
  const rows = useMemo(() => buildRows(leftText, rightText), [leftText, rightText]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            Removed (from <strong>{leftLabel}</strong>) in red, added (in <strong>{rightLabel}</strong>) in green.
          </DialogDescription>
        </DialogHeader>

        <div className="px-6 pt-3">
          <div className="layer-diff__legend">
            <span>− {leftLabel}</span>
            <span>+ {rightLabel}</span>
          </div>
          <div className="layer-diff">
            {rows.map((row, i) => (
              <div
                key={i}
                className={cn(
                  'layer-diff__row',
                  row.kind === 'add' && 'layer-diff__row--add',
                  row.kind === 'del' && 'layer-diff__row--del'
                )}
              >
                {row.kind === 'add' ? '+ ' : row.kind === 'del' ? '− ' : '  '}
                {row.text || ' '}
              </div>
            ))}
          </div>
        </div>

        <DialogFooter>
          {onLoadDefault && (
            <Button variant="ghost" onClick={onLoadDefault}>
              Load default into editor
            </Button>
          )}
          {onAdopt && (
            <Button variant="secondary" onClick={onAdopt}>
              Adopt default
            </Button>
          )}
          {onKeep && (
            <Button variant="primary" onClick={onKeep}>
              Keep mine
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

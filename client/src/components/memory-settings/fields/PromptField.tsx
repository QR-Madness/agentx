/**
 * PromptField — labelled prompt editor (Textarea) with a reset-to-default
 * button and a "leave empty to use default" hint. Replaces the
 * `.setting-textarea` + `.textarea-header` markup.
 */

import { RotateCcw } from 'lucide-react';
import { Label, Textarea, Button } from '../../ui';

interface PromptFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onReset: () => void;
  placeholder?: string;
  rows?: number;
}

export function PromptField({ label, value, onChange, onReset, placeholder, rows = 6 }: PromptFieldProps) {
  return (
    <div className="setting-textarea">
      <div className="textarea-header">
        <Label>{label}</Label>
        <Button variant="ghost" size="icon" onClick={onReset} title="Reset to default">
          <RotateCcw size={14} />
        </Button>
      </div>
      <Textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
      />
      {!value && <p className="prompt-hint">Leave empty to use default prompt</p>}
    </div>
  );
}

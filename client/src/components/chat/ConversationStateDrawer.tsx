/**
 * ConversationStateDrawer — the first *editable* conversation-state surface.
 *
 * The agent maintains a structured, slot-based working record (goals / decisions
 * / open threads / artifacts / narrative) via the `update_conversation_state`
 * tool; it rides every turn and survives context compression. This drawer lets
 * the user read and curate it — add, edit, or remove entries per slot. A user
 * edit is authoritative ("user wins"); the agent re-reads the state fresh each
 * turn. Existing entries round-trip their provenance so editing one entry doesn't
 * relabel the rest.
 */

import { useEffect, useState } from 'react';
import { Plus, Trash2, Target, CheckCircle2, ListTodo, FileText, NotebookPen } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { api } from '../../lib/api';
import { useConversationState } from '../../lib/hooks';
import { useNotify } from '../../contexts/NotificationContext';
import { Button, IconButton, Textarea } from '../ui';
import type { ConversationState, ConversationStateSlot, StateEntry } from '../../lib/api/types';
import './ConversationStateDrawer.css';

interface Props {
  conversationId: string;
  onClose: () => void;
}

type Draft = { id: string; text: string; author: 'user' | 'agent'; source_turn: number | null };

const SLOTS: { key: ConversationStateSlot; label: string; hint: string; icon: LucideIcon }[] = [
  { key: 'goals', label: 'Goals', hint: 'What this conversation is trying to achieve', icon: Target },
  { key: 'decisions', label: 'Decisions', hint: 'Choices locked in', icon: CheckCircle2 },
  { key: 'open_threads', label: 'Open threads', hint: 'Unresolved questions and next steps', icon: ListTodo },
  { key: 'artifacts', label: 'Artifacts', hint: 'Documents, drafts, or outputs produced', icon: FileText },
  { key: 'narrative', label: 'Narrative', hint: 'Freeform notes that fit no named slot', icon: NotebookPen },
];

let _seq = 0;
const nextId = () => `d${++_seq}`;

function toDrafts(entries: StateEntry[]): Draft[] {
  return entries.map((e) => ({ id: nextId(), text: e.text, author: e.author, source_turn: e.source_turn }));
}

function hydrate(state: ConversationState): Record<ConversationStateSlot, Draft[]> {
  return {
    goals: toDrafts(state.goals),
    decisions: toDrafts(state.decisions),
    open_threads: toDrafts(state.open_threads),
    artifacts: toDrafts(state.artifacts),
    narrative: toDrafts(state.narrative),
  };
}

/** Compare-key for dirty detection (ignores volatile ids/timestamps). */
const slotKey = (rows: Draft[]) =>
  JSON.stringify(rows.map((r) => [r.text.trim(), r.author, r.source_turn]));

export function ConversationStateDrawer({ conversationId }: Props) {
  const { state, loading } = useConversationState(conversationId);
  const { notifySuccess, notifyError } = useNotify();

  const [drafts, setDrafts] = useState<Record<ConversationStateSlot, Draft[]> | null>(null);
  const [originals, setOriginals] = useState<Record<ConversationStateSlot, Draft[]> | null>(null);
  const [saving, setSaving] = useState<ConversationStateSlot | null>(null);

  // Hydrate the editable copy the first time the state loads. Later refreshes are
  // our own saves, reconciled in-place per slot, so we never clobber edits.
  useEffect(() => {
    if (state && drafts === null) {
      setDrafts(hydrate(state));
      setOriginals(hydrate(state));
    }
  }, [state, drafts]);

  if (loading && !drafts) {
    return <div className="conv-state"><p className="conv-state-loading">Loading state…</p></div>;
  }
  if (!drafts || !originals) {
    return <div className="conv-state"><p className="conv-state-loading">No state yet.</p></div>;
  }

  const editRow = (slot: ConversationStateSlot, id: string, text: string) =>
    setDrafts((d) => ({ ...d!, [slot]: d![slot].map((r) => (r.id === id ? { ...r, text } : r)) }));

  const removeRow = (slot: ConversationStateSlot, id: string) =>
    setDrafts((d) => ({ ...d!, [slot]: d![slot].filter((r) => r.id !== id) }));

  const addRow = (slot: ConversationStateSlot) =>
    setDrafts((d) => ({ ...d!, [slot]: [...d![slot], { id: nextId(), text: '', author: 'user', source_turn: null }] }));

  const saveSlot = async (slot: ConversationStateSlot) => {
    setSaving(slot);
    try {
      const entries = drafts[slot]
        .filter((r) => r.text.trim())
        .map((r) => ({ text: r.text.trim(), author: r.author, source_turn: r.source_turn }));
      const resp = await api.updateConversationState(conversationId, slot, entries);
      const reconciled = toDrafts(resp.state[slot]);
      setDrafts((d) => ({ ...d!, [slot]: reconciled }));
      setOriginals((o) => ({ ...o!, [slot]: reconciled.map((r) => ({ ...r })) }));
      // Nudge the composer badge (a separate instance) to refetch.
      window.dispatchEvent(new CustomEvent('conversationstate:updated', { detail: { conversationId } }));
      notifySuccess('State saved');
    } catch (err) {
      notifyError(err, 'Failed to save state');
    } finally {
      setSaving(null);
    }
  };

  return (
    <div className="conv-state">
      <header className="conv-state-head">
        <span className="conv-state-eyebrow">Conversation</span>
        <h2 className="conv-state-title">State</h2>
        <p className="conv-state-sub">
          The structured working memory the agent keeps for this conversation. It survives context
          compression and rides every turn. Your edits win — the agent reads this fresh each turn.
        </p>
      </header>

      {SLOTS.map(({ key, label, hint, icon: Icon }) => {
        const rows = drafts[key];
        const dirty = slotKey(rows) !== slotKey(originals[key]);
        return (
          <section key={key} className="conv-state-slot">
            <div className="conv-state-slot-head">
              <Icon size={15} className="conv-state-slot-icon" />
              <h3 className="conv-state-slot-label">{label}</h3>
              <span className="conv-state-slot-count">{rows.length}</span>
            </div>
            <p className="conv-state-slot-hint">{hint}</p>

            <div className="conv-state-rows">
              {rows.map((row) => (
                <div key={row.id} className="conv-state-row">
                  <Textarea
                    value={row.text}
                    onChange={(e) => editRow(key, row.id, e.target.value)}
                    placeholder="Empty — will be dropped on save"
                    rows={2}
                    className="conv-state-input"
                  />
                  <div className="conv-state-row-side">
                    {row.author === 'agent' && <span className="conv-state-badge">agent</span>}
                    <IconButton
                      size="xs"
                      tone="danger"
                      aria-label="Remove entry"
                      onClick={() => removeRow(key, row.id)}
                    >
                      <Trash2 size={14} />
                    </IconButton>
                  </div>
                </div>
              ))}
              {rows.length === 0 && <p className="conv-state-empty">Nothing here yet.</p>}
            </div>

            <div className="conv-state-slot-actions">
              <Button variant="ghost" size="sm" onClick={() => addRow(key)}>
                <Plus size={14} /> Add
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!dirty || saving === key}
                onClick={() => saveSlot(key)}
              >
                {saving === key ? 'Saving…' : 'Save'}
              </Button>
            </div>
          </section>
        );
      })}
    </div>
  );
}

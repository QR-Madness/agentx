/**
 * SkillsSection — the skill library inside Connectors & Tools.
 *
 * A skill is a named instruction pack (markdown body). Agents see only a
 * compact index (id — name: description) in their system prompt and load a
 * body on demand via the `use_skill` internal tool, so the library can grow
 * without growing per-turn prompt cost. Access mirrors MCP servers:
 * `allowed_agent_ids` null = all agents, else a whitelist.
 */
import { useState } from 'react';
import { createPortal } from 'react-dom';
import { GraduationCap, Loader2, Pencil, Plus, Trash2 } from 'lucide-react';
import { api, apiErrorMessage } from '../../lib/api';
import type { AgentSkill, AgentSkillInput } from '../../lib/api';
import type { SkillsState } from '../../lib/hooks';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useConfirm } from '../ui/ConfirmDialog';
import { Button, IconButton, Input, Switch, Textarea, Tooltip } from '../ui';

export function SkillsSection({ skillsState }: { skillsState: SkillsState }) {
  const { skills, loading, refresh } = skillsState;
  const { profiles } = useAgentProfile();
  const confirm = useConfirm();
  // undefined = closed; null = creating; a skill = editing it.
  const [editing, setEditing] = useState<AgentSkill | null | undefined>(undefined);
  const [busy, setBusy] = useState<string | null>(null);

  const toggleEnabled = async (s: AgentSkill) => {
    setBusy(s.id);
    try { await api.updateSkill(s.id, { enabled: !s.enabled }); await refresh(); }
    finally { setBusy(null); }
  };

  const toggleAgent = async (s: AgentSkill, agentId: string) => {
    const current = s.allowed_agent_ids;
    // Mirror the server AccessView: leaving "all" pre-fills everyone-but-one.
    const next = current == null
      ? profiles.map(p => p.agentId).filter(a => a !== agentId)
      : current.includes(agentId)
        ? current.filter(a => a !== agentId)
        : [...current, agentId];
    setBusy(s.id);
    try { await api.updateSkill(s.id, { allowed_agent_ids: next }); await refresh(); }
    finally { setBusy(null); }
  };

  const setAllowAll = async (s: AgentSkill) => {
    setBusy(s.id);
    try { await api.updateSkill(s.id, { allowed_agent_ids: null }); await refresh(); }
    finally { setBusy(null); }
  };

  const onDelete = async (s: AgentSkill) => {
    const ok = await confirm({
      title: `Delete skill "${s.name}"?`,
      body: 'Agents lose access immediately; this cannot be undone.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    setBusy(s.id);
    try { await api.deleteSkill(s.id); await refresh(); }
    finally { setBusy(null); }
  };

  return (
    <>
      <div className="toolkit-section-title">
        <div>
          <p className="toolkit-section-note" style={{ margin: 0 }}>
            Skills are know-how, not tools: agents see a compact index and load a
            skill's full instructions on demand with the <code>use_skill</code> tool.
          </p>
        </div>
        <Button variant="primary" size="sm" onClick={() => setEditing(null)}>
          <Plus size={14} /> New skill
        </Button>
      </div>

      {loading && skills.length === 0 ? (
        <div className="toolkit-empty"><Loader2 className="spin" /> Loading…</div>
      ) : skills.length === 0 ? (
        <div className="toolkit-empty">
          No skills yet. Click <strong>New skill</strong> to write your first instruction pack.
        </div>
      ) : (
        <div className="toolkit-card-grid">
          {skills.map(s => {
            const allowAll = s.allowed_agent_ids == null;
            return (
              <div key={s.id} className="toolkit-card" style={s.enabled ? undefined : { opacity: 0.65 }}>
                <div className="toolkit-card-header">
                  <GraduationCap size={16} />
                  <span className="name" title={s.name}>{s.name}</span>
                  <Tooltip content={s.enabled ? 'Enabled — listed in agents’ skills index' : 'Disabled — hidden from agents'}>
                    <span>
                      <Switch
                        checked={s.enabled}
                        onCheckedChange={() => toggleEnabled(s)}
                        disabled={busy === s.id}
                        aria-label={`Toggle ${s.name}`}
                      />
                    </span>
                  </Tooltip>
                </div>
                <div className="meta" style={{ fontFamily: 'var(--font-mono, monospace)' }}>{s.id}</div>
                <div className="meta" style={{ color: 'var(--text-secondary)' }}>
                  {s.description || 'No description — agents rely on this to pick the skill.'}
                </div>
                {s.tags.length > 0 && (
                  <div className="toolkit-chips">
                    {s.tags.map(t => <span key={t} className="toolkit-chip">#{t}</span>)}
                  </div>
                )}
                <div className="toolkit-chips toolkit-access-chips">
                  <button
                    className={`toolkit-chip ${allowAll ? 'solid' : ''}`}
                    onClick={() => setAllowAll(s)}
                    disabled={busy === s.id}
                  >All agents</button>
                  {profiles.map(p => {
                    const on = !allowAll && (s.allowed_agent_ids || []).includes(p.agentId);
                    return (
                      <button
                        key={p.agentId}
                        className={`toolkit-chip ${on ? 'solid' : ''}`}
                        onClick={() => toggleAgent(s, p.agentId)}
                        disabled={busy === s.id}
                      >{p.name}</button>
                    );
                  })}
                </div>
                <div className="toolkit-card-actions">
                  <Button variant="ghost" size="sm" onClick={() => setEditing(s)} disabled={busy !== null}>
                    <Pencil size={14} /> Edit
                  </Button>
                  <Tooltip content="Delete skill">
                    <IconButton
                      aria-label={`Delete ${s.name}`}
                      size="sm"
                      tone="danger"
                      onClick={() => onDelete(s)}
                      disabled={busy === s.id}
                    >
                      {busy === s.id ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />}
                    </IconButton>
                  </Tooltip>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {editing !== undefined && createPortal(
        <div className="toolkit-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setEditing(undefined); }}>
          <div className="toolkit-modal-card">
            <h3>{editing ? `Edit ${editing.name}` : 'New skill'}</h3>
            <SkillForm
              initial={editing}
              onCancel={() => setEditing(undefined)}
              onSaved={async () => { setEditing(undefined); await refresh(); }}
            />
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

function SkillForm({
  initial, onCancel, onSaved,
}: {
  initial: AgentSkill | null;
  onCancel: () => void;
  onSaved: () => Promise<void>;
}) {
  const [name, setName] = useState(initial?.name ?? '');
  const [description, setDescription] = useState(initial?.description ?? '');
  const [body, setBody] = useState(initial?.body ?? '');
  const [tags, setTags] = useState((initial?.tags ?? []).join(', '));
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) { setError('Name is required'); return; }
    setSubmitting(true);
    setError(null);
    const input: AgentSkillInput = {
      name: name.trim(),
      description: description.trim(),
      body,
      tags: tags.split(',').map(t => t.trim()).filter(Boolean),
    };
    try {
      if (initial) await api.updateSkill(initial.id, input);
      else await api.createSkill(input);
      await onSaved();
    } catch (err) {
      setError(apiErrorMessage(err));
      setSubmitting(false);
    }
  };

  return (
    <form className="toolkit-form" onSubmit={submit}>
      <label>
        <span>Name</span>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Structured Decision Brief" required disabled={submitting} />
      </label>
      <label>
        <span>Description</span>
        <Input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="One line agents use to decide when this skill applies"
          disabled={submitting}
        />
      </label>
      <label>
        <span>Instructions (markdown)</span>
        <Textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={12}
          placeholder={'The full know-how an agent should follow when it loads this skill…'}
          disabled={submitting}
        />
      </label>
      <label>
        <span>Tags (comma-separated)</span>
        <Input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="thinking, writing" disabled={submitting} />
      </label>
      {error && (
        <div className="toolkit-error-banner"><span>{error}</span></div>
      )}
      <div className="toolkit-modal-actions">
        <Button type="button" variant="ghost" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" disabled={submitting}>
          {submitting ? <Loader2 size={14} className="spin" /> : null}
          {initial ? 'Save changes' : 'Create skill'}
        </Button>
      </div>
    </form>
  );
}

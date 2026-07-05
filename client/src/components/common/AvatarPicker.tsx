/**
 * AvatarPicker — the consolidated icon chooser.
 *
 * A clickable avatar tile (with an optional per-agent accent aura + hover "edit"
 * affordance) opens a modal: a Browse | Generate segmented header, a search field,
 * a Recently-used row, a categorized icon grid, and a large live preview of the
 * focused icon. The Generate segment is a disabled seam for the future flow.
 */

import { useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Pencil, Sparkles, Search, Loader2, Wand2 } from 'lucide-react';
import {
  AVATAR_CATEGORIES,
  AVATAR_OPTIONS,
  getAvatarOption,
  searchAvatars,
  type AvatarOption,
} from '../../lib/avatars';
import type { AgentAccent } from '../../lib/agentAccent';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, SegmentedControl,
} from '../ui';
import { AgentAvatar } from './AgentAvatar';
import { api, apiErrorMessage } from '../../lib/api';
import './AvatarPicker.css';

const RECENTS_KEY = 'agentx:avatar-recents';
const RECENTS_MAX = 10;

function loadRecents(): string[] {
  try {
    const raw = localStorage.getItem(RECENTS_KEY);
    const ids = raw ? (JSON.parse(raw) as string[]) : [];
    return ids.filter(id => AVATAR_OPTIONS.some(o => o.id === id));
  } catch {
    return [];
  }
}

interface AvatarPickerProps {
  value: string;
  onChange: (id: string) => void;
  size?: 'sm' | 'md' | 'lg';
  accent?: AgentAccent;
  ariaLabel?: string;
  /** Controlled open (with onOpenChange) — lets a caller drive the picker without
   *  the built-in trigger (e.g. opened from a menu item). Uncontrolled if omitted. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Hide the built-in trigger button (for controlled use). */
  hideTrigger?: boolean;
}

export function AvatarPicker({ value, onChange, size = 'md', accent, ariaLabel, open: openProp, onOpenChange, hideTrigger }: AvatarPickerProps) {
  const [openInternal, setOpenInternal] = useState(false);
  const open = openProp ?? openInternal;
  const setOpen = (o: boolean) => { onOpenChange?.(o); if (openProp === undefined) setOpenInternal(o); };
  const [tab, setTab] = useState<'browse' | 'generate'>('browse');
  const [query, setQuery] = useState('');
  const [hovered, setHovered] = useState<string | null>(null);
  const [recents, setRecents] = useState<string[]>(loadRecents);
  const reduce = useReducedMotion();

  // Generate tab state.
  const [subject, setSubject] = useState('');
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [genRef, setGenRef] = useState<string | null>(null); // media:{ws}/{doc} of the last result

  const triggerPx = size === 'lg' ? 28 : size === 'sm' ? 16 : 20;
  const results = useMemo(() => searchAvatars(query), [query]);
  const grouped = useMemo(() => {
    return AVATAR_CATEGORIES.map(cat => ({
      ...cat,
      items: results.filter(o => o.category === cat.id),
    })).filter(g => g.items.length > 0);
  }, [results]);

  const previewId = hovered ?? value;
  const previewOpt = getAvatarOption(previewId);

  const pick = (id: string) => {
    onChange(id);
    const next = [id, ...recents.filter(r => r !== id)].slice(0, RECENTS_MAX);
    setRecents(next);
    try { localStorage.setItem(RECENTS_KEY, JSON.stringify(next)); } catch { /* ignore */ }
    setOpen(false);
  };

  const generate = async () => {
    const prompt = subject.trim();
    if (!prompt || generating) return;
    setGenerating(true);
    setGenError(null);
    try {
      const res = await api.generateAvatar({ subject_prompt: prompt });
      setGenRef(`media:${res.workspace_id}/${res.doc_id}`);
    } catch (err) {
      setGenError(apiErrorMessage(err));
    } finally {
      setGenerating(false);
    }
  };

  const useGenerated = () => {
    if (!genRef) return;
    onChange(genRef); // a media: ref — not an icon id, so it skips the recents row
    setOpen(false);
  };

  const auraStyle = accent
    ? ({ '--agent-accent': accent.accent, '--agent-soft': accent.soft } as React.CSSProperties)
    : undefined;

  const IconButton = ({ opt }: { opt: AvatarOption }) => {
    const Icon = opt.icon;
    return (
      <button
        type="button"
        className={`avatar-pick-cell ${value === opt.id ? 'selected' : ''}`}
        title={opt.label}
        onClick={() => pick(opt.id)}
        onMouseEnter={() => setHovered(opt.id)}
        onFocus={() => setHovered(opt.id)}
      >
        <Icon size={18} />
      </button>
    );
  };

  return (
    <>
      {!hideTrigger && (
        <button
          type="button"
          className={`avatar-trigger avatar-trigger--${size} ${accent ? 'has-accent' : ''}`}
          style={auraStyle}
          onClick={() => setOpen(true)}
          aria-label={ariaLabel ?? 'Choose avatar'}
        >
          <span className="avatar-trigger__aura" />
          <AgentAvatar avatar={value} size={triggerPx} className="avatar-trigger__icon" fill />
          <span className="avatar-trigger__edit"><Pencil size={size === 'lg' ? 14 : 11} /></span>
        </button>
      )}

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Choose an icon</DialogTitle>
          </DialogHeader>
          <div className="avatar-pick">
            <SegmentedControl
              size="sm"
              ariaLabel="Icon source"
              value={tab}
              onChange={(v) => setTab(v)}
              options={[
                { value: 'browse', label: 'Browse' },
                { value: 'generate', label: '✨ Generate' },
              ]}
            />

            {tab === 'browse' ? (
              <div className="avatar-pick__body">
                <aside className="avatar-pick__preview" style={auraStyle}>
                  <span className="avatar-pick__preview-aura" />
                  <AgentAvatar avatar={previewId} size={44} className="avatar-pick__preview-icon" />
                  <span className="avatar-pick__preview-label">{previewOpt?.label ?? '—'}</span>
                  <span className="avatar-pick__preview-cat">
                    {AVATAR_CATEGORIES.find(c => c.id === previewOpt?.category)?.label ?? ''}
                  </span>
                </aside>

                <div className="avatar-pick__main">
                  <div className="avatar-pick__search">
                    <Search size={14} />
                    <input
                      autoFocus
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search icons…"
                      aria-label="Search icons"
                    />
                  </div>

                  <div className="avatar-pick__scroll" onMouseLeave={() => setHovered(null)}>
                    {!query && recents.length > 0 && (
                      <div className="avatar-pick__group">
                        <div className="avatar-pick__group-label">Recently used</div>
                        <div className="avatar-pick__grid">
                          {recents.map(id => {
                            const opt = getAvatarOption(id);
                            return opt ? <IconButton key={`r-${id}`} opt={opt} /> : null;
                          })}
                        </div>
                      </div>
                    )}

                    {grouped.length === 0 && (
                      <div className="avatar-pick__empty">No icons match “{query}”.</div>
                    )}

                    {grouped.map(group => (
                      <div className="avatar-pick__group" key={group.id}>
                        <div className="avatar-pick__group-label">{group.label}</div>
                        <motion.div
                          className="avatar-pick__grid"
                          initial={reduce ? false : 'hidden'}
                          animate="show"
                          variants={{ show: { transition: { staggerChildren: 0.004 } } }}
                        >
                          {group.items.map(opt => (
                            <motion.div
                              key={opt.id}
                              variants={reduce ? {} : { hidden: { opacity: 0, scale: 0.8 }, show: { opacity: 1, scale: 1 } }}
                            >
                              <IconButton opt={opt} />
                            </motion.div>
                          ))}
                        </motion.div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="avatar-pick__generate">
                <div className="avatar-pick__generate-inner" style={{ gap: 12, width: '100%', maxWidth: 420 }}>
                  {genRef ? (
                    <AgentAvatar avatar={genRef} size={120} />
                  ) : (
                    <Sparkles size={28} />
                  )}
                  <div className="avatar-pick__generate-title">Generate an avatar</div>
                  <div className="avatar-pick__generate-sub">
                    Describe the agent — the app-wide style is set in Settings → Images. Uses
                    OpenRouter.
                  </div>
                  <textarea
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    rows={2}
                    placeholder="e.g. a gray-haired strategist with glasses"
                    className="ax-field ax-field--sm w-full resize-none"
                  />
                  {genError && <div className="text-xs text-error">{genError}</div>}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={generate}
                      disabled={!subject.trim() || generating}
                      className="inline-flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-sm text-fg-inverse shadow-sm transition hover:brightness-110 disabled:opacity-40"
                    >
                      {generating ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
                      {genRef ? 'Regenerate' : 'Generate'}
                    </button>
                    {genRef && (
                      <button
                        type="button"
                        onClick={useGenerated}
                        className="inline-flex h-8 items-center rounded-md border border-line px-3 text-sm text-fg transition-colors hover:border-line-strong"
                      >
                        Use this avatar
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

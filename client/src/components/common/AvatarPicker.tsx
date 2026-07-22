/**
 * AvatarPicker — the consolidated icon chooser.
 *
 * A clickable avatar tile (with an optional per-agent accent aura + hover "edit"
 * affordance) opens a modal: a Browse | Generate segmented header, a search field,
 * a Recently-used row, a categorized icon grid, and a large live preview of the
 * focused icon. The Generate segment is a disabled seam for the future flow.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Pencil, Sparkles, Search, Loader2, Wand2, Dices } from 'lucide-react';
import {
  AVATAR_CATEGORIES,
  AVATAR_OPTIONS,
  getAvatarOption,
  searchAvatars,
  type AvatarOption,
} from '../../lib/avatars';
import type { AgentAccent } from '../../lib/agentAccent';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, SegmentedControl,
} from '../ui';
import { AgentAvatar } from './AgentAvatar';
import { api, apiErrorMessage, type WorkspaceDocument } from '../../lib/api';
import './AvatarPicker.css';

const HOME_ID = 'ws_home';
/** doc id behind a `media:{ws}/{doc}` avatar ref (else null). */
function docIdOf(avatar?: string): string | null {
  return avatar && avatar.startsWith('media:') ? (avatar.split('/').pop() ?? null) : null;
}

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
  /** Profile-derived subject suggestion shown as the Generate placeholder
   *  (e.g. "Hazel — orchestration"). Purely a hint — empty still generates. */
  subjectSeed?: string;
}

export function AvatarPicker({ value, onChange, size = 'md', accent, ariaLabel, open: openProp, onOpenChange, hideTrigger, subjectSeed }: AvatarPickerProps) {
  const [openInternal, setOpenInternal] = useState(false);
  const open = openProp ?? openInternal;
  const [tab, setTab] = useState<'browse' | 'gallery' | 'generate'>('browse');
  const [query, setQuery] = useState('');
  const [hovered, setHovered] = useState<string | null>(null);
  const [recents, setRecents] = useState<string[]>(loadRecents);
  const reduce = useReducedMotion();

  // Generate tab state. Results are a candidate deck (media:{ws}/{doc} refs);
  // clicking a candidate applies it. `busy` distinguishes a single generation
  // from a 4-deal (sequential — the grid fills as images land).
  const [subject, setSubject] = useState('');
  const [busy, setBusy] = useState<'one' | 'deal' | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<string[]>([]);
  // Cancels the in-flight + remaining sequential generations (each is billed) as
  // soon as the user picks a candidate or closes the picker.
  const abortRef = useRef<AbortController | null>(null);

  // "Your avatars" gallery — reuse an image you already generated (esp. unused
  // ones) instead of paying to regenerate. Lazy-loaded on first view, refetched
  // each open so freshly generated faces appear.
  const { profiles } = useAgentProfile();
  const [gallery, setGallery] = useState<WorkspaceDocument[] | null>(null);
  const [galleryLoading, setGalleryLoading] = useState(false);

  const usageByDocId = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of profiles) {
      const id = docIdOf(p.avatar);
      if (id) m.set(id, p.name);
    }
    return m;
  }, [profiles]);

  const cancelGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(null);
  }, []);

  const setOpen = (o: boolean) => {
    if (!o) cancelGeneration(); // closing mid-deal must stop the billed loop
    onOpenChange?.(o);
    if (openProp === undefined) setOpenInternal(o);
  };

  const loadGallery = useCallback(async () => {
    setGalleryLoading(true);
    try {
      const { documents } = await api.listDocuments(HOME_ID);
      setGallery(documents.filter(d => d.filename.startsWith('avatars/')));
    } catch {
      setGallery([]);
    } finally {
      setGalleryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open && tab === 'gallery' && gallery === null && !galleryLoading) void loadGallery();
  }, [open, tab, gallery, galleryLoading, loadGallery]);
  // Drop the cache on close so the next open reflects any newly generated faces.
  useEffect(() => { if (!open) setGallery(null); }, [open]);

  const galleryOrdered = useMemo(() => {
    if (!gallery) return [];
    // Unused first — reusing an unassigned face is the point of this tab.
    return [...gallery].sort((a, b) => (usageByDocId.has(a.id) ? 1 : 0) - (usageByDocId.has(b.id) ? 1 : 0));
  }, [gallery, usageByDocId]);

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

  const generateBatch = async (count: 1 | 4) => {
    if (busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(count === 1 ? 'one' : 'deal');
    setGenError(null);
    setCandidates([]);
    // Sequential on purpose — respects provider rate limits; the grid fills
    // progressively. An empty subject is valid: the server template invents a
    // synthetic face (and varies it between generations). A cancel (pick/close)
    // breaks the loop *and* aborts the in-flight request, so we never bill for
    // images the user will never see.
    for (let i = 0; i < count; i++) {
      if (controller.signal.aborted) break;
      try {
        const res = await api.generateAvatar({ subject_prompt: subject.trim() }, controller.signal);
        if (controller.signal.aborted) break;
        setCandidates(prev => [...prev, `media:${res.workspace_id}/${res.doc_id}`]);
      } catch (err) {
        if (controller.signal.aborted) break; // user cancelled — not a real error
        setGenError(apiErrorMessage(err));
        break;
      }
    }
    // Leave state alone if a cancel already reset it (abortRef swapped to null).
    if (abortRef.current === controller) {
      abortRef.current = null;
      setBusy(null);
    }
  };

  const applyCandidate = (ref: string) => {
    cancelGeneration(); // stop any remaining sequential generations
    onChange(ref); // a media: ref — not an icon id, so it skips the recents row
    setOpen(false);
  };

  const auraStyle = accent
    ? ({ '--agent-accent': accent.accent, '--agent-soft': accent.soft } as React.CSSProperties)
    : undefined;

  const AvatarCell = ({ opt }: { opt: AvatarOption }) => {
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
            <DialogTitle>Choose an avatar</DialogTitle>
          </DialogHeader>
          <div className="avatar-pick">
            <SegmentedControl
              size="sm"
              ariaLabel="Avatar source"
              value={tab}
              onChange={(v) => setTab(v)}
              options={[
                { value: 'browse', label: 'Browse' },
                { value: 'gallery', label: 'Your avatars' },
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
                            return opt ? <AvatarCell key={`r-${id}`} opt={opt} /> : null;
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
                              <AvatarCell opt={opt} />
                            </motion.div>
                          ))}
                        </motion.div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : tab === 'gallery' ? (
              <div className="avatar-pick__gallery">
                {galleryLoading && gallery === null ? (
                  <div className="avatar-pick__empty">
                    <Loader2 size={16} className="animate-spin" /> Loading your avatars…
                  </div>
                ) : galleryOrdered.length === 0 ? (
                  <div className="avatar-pick__empty">
                    No saved avatars yet — generate one and it’ll live here for reuse.
                  </div>
                ) : (
                  <div className="avatar-pick__gallery-grid">
                    {galleryOrdered.map(doc => {
                      const ref = `media:${HOME_ID}/${doc.id}`;
                      const usedBy = usageByDocId.get(doc.id);
                      return (
                        <button
                          key={doc.id}
                          type="button"
                          className={`avatar-pick__gcell ${value === ref ? 'selected' : ''}`}
                          onClick={() => applyCandidate(ref)}
                          title={usedBy ? `In use by ${usedBy} — click to reuse` : 'Unused — click to use'}
                        >
                          <AgentAvatar avatar={ref} size={76} />
                          {usedBy ? (
                            <span className="avatar-pick__gcell-badge">{usedBy}</span>
                          ) : (
                            <span className="avatar-pick__gcell-badge avatar-pick__gcell-badge--free">unused</span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              <div className="avatar-pick__generate">
                <div className="avatar-pick__generate-inner" style={{ gap: 12, width: '100%', maxWidth: 460 }}>
                  {candidates.length === 0 && !busy && <Sparkles size={28} />}
                  <div className="avatar-pick__generate-title">Generate an avatar</div>
                  <div className="avatar-pick__generate-sub">
                    Describe the agent — or leave it empty and the style template invents a
                    synthetic face. The template lives in Settings → Images.
                  </div>
                  <textarea
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    rows={2}
                    placeholder={subjectSeed ? `e.g. ${subjectSeed}` : 'e.g. a gray-haired strategist with glasses'}
                    className="ax-field ax-field--sm w-full resize-none"
                  />

                  {(candidates.length > 0 || busy) && (
                    <div className="avatar-pick__cands">
                      {candidates.map(ref => (
                        <button
                          key={ref}
                          type="button"
                          className="avatar-pick__cand"
                          onClick={() => applyCandidate(ref)}
                          title="Use this avatar"
                        >
                          <AgentAvatar avatar={ref} size={104} />
                        </button>
                      ))}
                      {busy &&
                        Array.from({ length: (busy === 'deal' ? 4 : 1) - candidates.length }).map((_, i) => (
                          <div key={`pending-${i}`} className="avatar-pick__cand avatar-pick__cand--pending">
                            <Loader2 size={18} className="animate-spin" />
                          </div>
                        ))}
                    </div>
                  )}

                  {genError && <div className="text-xs text-error">{genError}</div>}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void generateBatch(1)}
                      disabled={busy !== null}
                      className="inline-flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-sm text-fg-inverse shadow-sm transition hover:brightness-110 disabled:opacity-40"
                    >
                      {busy === 'one' ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
                      {candidates.length > 0 ? 'Regenerate' : 'Generate'}
                    </button>
                    <button
                      type="button"
                      onClick={() => void generateBatch(4)}
                      disabled={busy !== null}
                      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line bg-surface-overlay px-3 text-sm text-fg transition-colors hover:border-line-strong disabled:opacity-40"
                    >
                      {busy === 'deal' ? <Loader2 size={14} className="animate-spin" /> : <Dices size={14} />}
                      Deal 4
                    </button>
                  </div>
                  <div className="avatar-pick__generate-sub">
                    Click a result to use it. Each image is billed to your image provider —
                    the cost lands in the usage ledger.
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

/**
 * Voice caption log — the accumulating, scrollable transcript of a voice-mode
 * exchange (your lines + the ambassador's spoken lines), shown live in
 * `VoiceSurface`. Kept as a pure reducer so the de-dupe / append rules are
 * testable without a DOM or a mic.
 *
 * Rules (bulletproofing):
 *  - blank text is ignored (no empty bubbles);
 *  - a line identical to the immediately-preceding same-role line is dropped
 *    (a briefing that re-plays, or a double-fired callback, never double-prints);
 *  - ids are stable per entry so React keys don't thrash on re-render.
 */

export type CaptionRole = 'you' | 'ambassador';

export interface Caption {
  id: string;
  role: CaptionRole;
  text: string;
}

export function appendCaption(list: Caption[], role: CaptionRole, text: string): Caption[] {
  const trimmed = text.trim();
  if (!trimmed) return list;
  const last = list[list.length - 1];
  if (last && last.role === role && last.text === trimmed) return list;
  const id = `${role}-${list.length}-${Date.now()}`;
  return [...list, { id, role, text: trimmed }];
}

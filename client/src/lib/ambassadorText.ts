/**
 * Think-block hygiene for ambassador text (render + speech + copy).
 *
 * The server now suppresses `<think>…</think>` reasoning at the stream, but
 * records persisted before that fix (and any future provider quirk) may still
 * carry blocks. Mirror of the API's `strip_think_blocks`: closed blocks go,
 * then an unterminated trailing block; if that would empty the text entirely
 * (a reply that was ALL reasoning) only the tags are removed — a visible
 * answer beats a blank one.
 */

const BLOCK_RE = /<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/g;
const OPEN_TAIL_RE = /<think(?:ing)?>[\s\S]*$/;
const TAGS_RE = /<\/?think(?:ing)?>/g;

export function stripThinkBlocks(text: string): string {
  if (!text || !text.includes('<think')) return text;
  const stripped = text.replace(BLOCK_RE, '').replace(OPEN_TAIL_RE, '').trim();
  if (stripped) return stripped;
  return text.replace(TAGS_RE, '').trim();
}

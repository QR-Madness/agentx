/**
 * Rotating, memory-flavored status lines for the consolidation drawer — shown as
 * the hero line while a run is in flight (the real stage/detail sits beneath).
 * Tasteful, not cringe. `nextMessage` avoids repeating the line it's given.
 */

export const CONSOLIDATION_MESSAGES: readonly string[] = [
  'Crystallizing your facts',
  'Remembering the past',
  'Connecting the dots',
  'Filing away the good bits',
  'Reminiscing',
  'Never forgetting about you',
  'Sorting the signal from the noise',
  'Weaving the memory graph',
  'Committing it all to memory',
  'Tidying the mind palace',
  'Getting my nap in',
  'Letting it sink in',
];

/** Pick a random message, never returning `previous` (unless the pool is size 1). */
export function nextMessage(previous?: string): string {
  const pool = CONSOLIDATION_MESSAGES;
  if (pool.length <= 1) return pool[0] ?? '';
  let pick = previous;
  while (pick === previous) {
    pick = pool[Math.floor(Math.random() * pool.length)];
  }
  return pick as string;
}

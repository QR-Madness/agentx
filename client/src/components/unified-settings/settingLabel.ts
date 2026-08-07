/**
 * One way to turn a setting's key into something readable.
 *
 * There were two: the search hook capitalised only the first letter of the
 * whole path, the Overview digest capitalised each segment — so the same
 * setting appeared as "Search · max results" in one list and "Search · Max
 * results" in the other, on the same screen. Both callers now use this.
 *
 * Manifest keys are the raw config paths (`search.max_results`) or memory field
 * names (`recall_candidate_pool`); neither is meant for someone who has never
 * opened the config file.
 */

function titleCase(segment: string): string {
  return segment.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
}

/** `search.max_results` → "Search · Max results". */
export function humanizeSettingKey(key: string): string {
  return key.split('.').map(titleCase).join(' · ');
}

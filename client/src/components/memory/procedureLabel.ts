// ─── Procedure headline formatting ─────────────────────────────────────────
// Distilled procedure triggers are stored condition-phrased by the distill
// prompt — they usually already begin with a condition word ("when presenting
// a recommendation"). Unconditionally prepending "When " produced the doubled
// "When when …" wart. Capitalize the trigger when it already leads with a
// condition word; otherwise prepend "When ".

const CONDITION_LEAD = /^(when|whenever|before|after|while|if|once|upon|during|as)\b/i;

/** Human-readable headline for a procedure: its trigger (de-doubled) or body. */
export function procedureHeadline(trigger?: string | null, body?: string | null): string {
  const t = (trigger ?? '').trim();
  if (!t) return (body ?? '').trim();
  if (CONDITION_LEAD.test(t)) return t.charAt(0).toUpperCase() + t.slice(1);
  return `When ${t}`;
}

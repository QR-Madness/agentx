/**
 * modelShortLabel — turn a fully-qualified model id into a friendly tail.
 *
 *   "openrouter:deepseek/deepseek-v4-flash" → "deepseek-v4-flash"
 *   "vercel:nvidia/nemotron-3-ultra-550b"   → "nemotron-3-ultra-550b"
 *   "gpt-4o"                                 → "gpt-4o"
 *
 * One shared derivation so the agent picker, the roster dossier, and the
 * profile-editor nav can't drift on how a model reads. Returns `null` for an
 * empty/unset model so callers can render an "inherits default" affordance
 * instead of a blank.
 */
export function modelShortLabel(model?: string | null): string | null {
  if (!model) return null;
  // Strip the provider prefix ("openrouter:…"), then the namespace ("deepseek/…").
  const afterProvider = model.includes(':') ? model.slice(model.indexOf(':') + 1) : model;
  const tail = afterProvider.includes('/')
    ? afterProvider.slice(afterProvider.lastIndexOf('/') + 1)
    : afterProvider;
  return tail || null;
}

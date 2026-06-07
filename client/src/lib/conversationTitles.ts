/**
 * Conversation title overrides — now a thin shim over the generalized
 * per-conversation metadata store (`conversationMeta.ts`), which folded the
 * title map in (with a one-time migration of the legacy `convTitles` storage).
 * Kept so existing imports keep working unchanged.
 */

export {
  getTitleOverride,
  setTitleOverride,
  clearTitleOverride,
  getDisplayTitle,
} from './conversationMeta';

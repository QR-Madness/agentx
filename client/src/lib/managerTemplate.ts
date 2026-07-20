// Agentic Organizations — the manager "report contract" tool template.
//
// Client-side MIRROR of `api/agentx_ai/agent/models.py::MANAGER_REPORT_ONLY_BLOCKED_TOOLS`
// (keep the two lists in sync — both are pinned by tests). The server merge on the
// role transition is authoritative; this mirror exists because the profile editor's
// form state never rehydrates after an autosave, so without a local merge the next
// PUT would send the pre-template blocked list and wipe the server's merge. Both
// merges are order-preserving unions — applying either (or both) converges.
export const MANAGER_BLOCKED_TOOLS: readonly string[] = [
  '_internal.create_document',
  '_internal.update_document',
  '_internal.append_to_document',
  '_internal.edit_document',
  '_internal.rename_document',
  '_internal.delete_document',
  '_internal.run_command',
  '_internal.write_file',
  '_internal.generate_image',
  '_internal.generate_speech',
];

/** Order-preserving union of the current blocked list with the manager template. */
export function mergeManagerTemplate(blocked: string[] | undefined | null): string[] {
  return [...new Set([...(blocked ?? []), ...MANAGER_BLOCKED_TOOLS])];
}

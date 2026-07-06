/**
 * PromptTemplatesSection — the prompt template library as a settings section
 * (prompts nav group). Renders the shared PromptLibraryBrowser as a pure
 * management surface (browse/search/create/edit/delete/reset — no insert
 * action). The in-chat PromptLibraryModal stays for quick access.
 */

import { Library } from 'lucide-react';
import { SectionHeader } from '../../ui';
import { PromptLibraryBrowser } from '../../prompt-library/PromptLibraryBrowser';

export default function PromptTemplatesSection() {
  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Library size={20} />}
        title="Template Library"
        description="Browse, create, and edit the reusable prompt templates available in chat and the profile editor."
      />
      <div className="plm-section-host">
        <PromptLibraryBrowser variant="section" />
      </div>
    </div>
  );
}

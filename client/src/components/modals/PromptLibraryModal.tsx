/**
 * PromptLibraryModal — quick-access prompt template library (in-chat modal,
 * profile editor, system-prompt "insert from library" dialog).
 *
 * Thin wrapper over the shared PromptLibraryBrowser; the unified-settings
 * Template Library section renders the same browser without insert/select.
 */

import {
  PromptLibraryBrowser,
  type PromptLibraryBrowserProps,
} from '../prompt-library/PromptLibraryBrowser';

interface PromptLibraryModalProps extends Omit<PromptLibraryBrowserProps, 'onClose' | 'variant'> {
  onClose: () => void;
  variant?: 'modal' | 'panel';
}

export function PromptLibraryModal({ variant = 'modal', ...props }: PromptLibraryModalProps) {
  return <PromptLibraryBrowser variant={variant} {...props} />;
}

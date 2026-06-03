/**
 * memoElement — memoize an element renderer on `element` identity only.
 *
 * Element renderers share the `ElementRenderProps` contract, which also carries
 * volatile interaction props (choice callbacks/flags). Heavy renderers (mermaid,
 * table) must not re-render when those change — only when their `element`
 * changes (which is stable within a message and changes only on amend).
 */

import { memo } from 'react';
import type { ComponentType } from 'react';
import type { ElementRenderProps } from './types';

export function memoElement(
  Component: ComponentType<ElementRenderProps>,
): ComponentType<ElementRenderProps> {
  return memo(Component, (a, b) => a.element === b.element);
}

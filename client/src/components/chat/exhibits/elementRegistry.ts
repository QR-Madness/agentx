/**
 * Element registry — maps an exhibit `Element.type` to its renderer.
 *
 * This is the allow-list / sandbox seam on the client: a type without a
 * registered renderer falls through to a safe source-as-code fallback in
 * `ExhibitBubble` (never raw HTML). Adding an element type = add its renderer
 * here (and the matching backend element model).
 */

import type { ComponentType } from 'react';
import type { ExhibitElement } from '../../../lib/exhibits';
import { MermaidElement } from './MermaidElement';

export interface ElementProps {
  content: string;
  title?: string;
}

export const elementRegistry: Partial<
  Record<ExhibitElement['type'], ComponentType<ElementProps>>
> = {
  mermaid: MermaidElement,
};

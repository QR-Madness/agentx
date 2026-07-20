/**
 * Element registry — maps an exhibit `Element.type` to its renderer.
 *
 * This is the allow-list / sandbox seam on the client: a type without a
 * registered renderer falls through to a safe source-as-code fallback in
 * `ExhibitBubble` (never raw HTML). Adding an element type = add its renderer
 * here (and the matching backend element model). Renderers share the
 * {@link ElementRenderProps} contract and narrow on `element.type`.
 */

import type { ComponentType } from 'react';
import type { ExhibitElement } from '../../../lib/exhibits';
import type { ElementRenderProps } from './types';
import { MermaidElement } from './MermaidElement';
import { ChoiceElement } from './ChoiceElement';
import { TableElement } from './TableElement';
import { CitationElement } from './CitationElement';
import { ImageElement } from './ImageElement';
import { AudioElement } from './AudioElement';
import { VideoElement } from './VideoElement';
import { TextElement } from './TextElement';

export type { ElementRenderProps };

export const elementRegistry: Partial<
  Record<ExhibitElement['type'], ComponentType<ElementRenderProps>>
> = {
  mermaid: MermaidElement,
  choice: ChoiceElement,
  table: TableElement,
  citation: CitationElement,
  image: ImageElement,
  audio: AudioElement,
  video: VideoElement,
  text: TextElement,
};

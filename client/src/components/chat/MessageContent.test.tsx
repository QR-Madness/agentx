import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { MessageContent } from './MessageContent';

/**
 * Locks the markdown plugin-ordering contract: rehype-raw (raw HTML like
 * `<br/>`), rehype-katex (math), and rehype-highlight (code) must all keep
 * working together. If the plugin order regresses, one of these breaks.
 */
describe('MessageContent', () => {
  it('renders raw <br/> inside a table cell as a line break', () => {
    const { container } = render(
      <MessageContent content={'| Col |\n| --- |\n| a<br/>b |'} />
    );
    expect(container.querySelector('table')).toBeTruthy();
    expect(container.querySelector('td br')).toBeTruthy();
  });

  it('still renders inline math via KaTeX', () => {
    const { container } = render(<MessageContent content={'mass-energy $x^2$ here'} />);
    expect(container.querySelector('.katex')).toBeTruthy();
  });

  it('still highlights fenced code', () => {
    const { container } = render(
      <MessageContent content={'```js\nconst x = 1;\n```'} />
    );
    expect(container.querySelector('pre code')).toBeTruthy();
  });
});

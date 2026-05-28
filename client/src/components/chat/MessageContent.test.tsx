import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { MessageContent, normalizeMathDelimiters } from './MessageContent';

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

  it('renders inline math with \\( \\) delimiters via KaTeX', () => {
    const { container } = render(<MessageContent content={'energy \\(E = mc^2\\) here'} />);
    expect(container.querySelector('.katex')).toBeTruthy();
  });

  it('renders display math with \\[ \\] delimiters via KaTeX', () => {
    const { container } = render(<MessageContent content={'\\[ \\int_0^1 x\\,dx \\]'} />);
    expect(container.querySelector('.katex')).toBeTruthy();
  });
});

describe('normalizeMathDelimiters', () => {
  it('converts \\( \\) to $ and \\[ \\] to $$', () => {
    expect(normalizeMathDelimiters('a \\(x^2\\) b')).toBe('a $x^2$ b');
    expect(normalizeMathDelimiters('\\[ y \\]')).toBe('$$y$$');
  });

  it('leaves backslash-bracket sequences inside code untouched', () => {
    expect(normalizeMathDelimiters('`\\(x\\)`')).toBe('`\\(x\\)`');
    expect(normalizeMathDelimiters('```\n\\[x\\]\n```')).toBe('```\n\\[x\\]\n```');
  });

  it('leaves existing $-delimited math alone', () => {
    expect(normalizeMathDelimiters('$x^2$ and $$y$$')).toBe('$x^2$ and $$y$$');
  });

  it('escapes currency $ so it does not collide with math delimiters', () => {
    expect(normalizeMathDelimiters('It costs $20 to run \\[\\frac{a}{b}\\].'))
      .toBe('It costs \\$20 to run $$\\frac{a}{b}$$.');
    // $$ display delimiters are left intact
    expect(normalizeMathDelimiters('$$y=2$$')).toBe('$$y=2$$');
  });
});

describe('MessageContent currency + math', () => {
  it('does not blank the message when currency mixes with math', () => {
    const { container } = render(
      <MessageContent content={'It costs $20 to compute \\(x^2\\).'} />
    );
    // math still renders…
    expect(container.querySelector('.katex')).toBeTruthy();
    // …and the currency text survives
    expect(container.textContent).toContain('$20');
  });
});

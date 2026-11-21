/**
 * keyframes — `@keyframes` names share ONE global namespace across every
 * stylesheet in the app. `styles/base.css` is imported into `@layer base`, so
 * an unlayered component stylesheet that defines the same name overrides it
 * outright; among unlayered copies the last-loaded one wins, and lazy chunks
 * load late. Before v0.21.295 the Settings sections' `.fade-in` never got
 * base.css's rise: it used a chat stylesheet's copy, and once Memory had been
 * opened, Memory's copy turned the rise into a drop.
 *
 * This test loads every `.css` file under `src/` as raw text and fails when a
 * keyframe name has more than one distinct definition. Identical copies (the
 * per-file `spin` helpers) are allowed.
 *
 * If this fails: give the component's animation a component-prefixed kebab-case
 * name (`message-bubble-in`, `auth-glow-pulse`). `styles/base.css` owns the
 * generic names (`fadeIn`, `slideIn`, `pulse`, `glow-pulse`).
 */

import { describe, it, expect } from 'vitest';

// Raw source of every stylesheet under src/ (root-absolute keys like `/src/styles/base.css`).
const sheets = import.meta.glob('/src/**/*.css', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/** Whitespace- and comment-insensitive form of a keyframes body, for comparison. */
function normalise(body: string): string {
  return body
    .replace(/\s+/g, ' ')
    .replace(/\s*([{};:,])\s*/g, '$1')
    .replace(/;}/g, '}')
    .trim();
}

/** Every `@keyframes` block in a stylesheet: its name and normalised body. */
function keyframes(css: string): Array<{ name: string; body: string }> {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const found: Array<{ name: string; body: string }> = [];
  for (const match of source.matchAll(/@keyframes\s+([\w-]+)\s*\{/g)) {
    const start = (match.index ?? 0) + match[0].length;
    let depth = 1;
    let end = start;
    while (end < source.length && depth > 0) {
      if (source[end] === '{') depth++;
      else if (source[end] === '}') depth--;
      end++;
    }
    found.push({ name: match[1] ?? '', body: normalise(source.slice(start, end - 1)) });
  }
  return found;
}

describe('CSS @keyframes names', () => {
  // name -> normalised body -> files defining it that way
  const definitions = new Map<string, Map<string, string[]>>();
  for (const [path, css] of Object.entries(sheets)) {
    for (const { name, body } of keyframes(css)) {
      const bodies = definitions.get(name) ?? new Map<string, string[]>();
      bodies.set(body, [...(bodies.get(body) ?? []), path.replace(/^\//, '')]);
      definitions.set(name, bodies);
    }
  }

  it('reads the stylesheets (an empty raw import would pass the real check vacuously)', () => {
    expect(definitions.has('fadeIn')).toBe(true);
    expect(definitions.size).toBeGreaterThan(20);
  });

  it('gives every keyframe name a single definition', () => {
    const conflicts = [...definitions]
      .filter(([, bodies]) => bodies.size > 1)
      .map(([name, bodies]) => `${name}: ${[...bodies.values()].map((files) => files.join(', ')).join('  vs  ')}`)
      .sort();

    expect(
      conflicts,
      `@keyframes defined differently in more than one place — rename the component's copy:\n` +
        conflicts.join('\n'),
    ).toEqual([]);
  });
});

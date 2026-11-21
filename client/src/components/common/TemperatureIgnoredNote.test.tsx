import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TemperatureIgnoredNote } from './TemperatureIgnoredNote';
import { temperatureIgnoredBy } from '../../lib/modelSampling';

const HINT = 'No effect on Anthropic models — they use their default sampling.';

describe('TemperatureIgnoredNote', () => {
  it.each([
    'anthropic:claude-3-5-sonnet-latest',
    'anthropic:claude-opus-4-1',
  ])('shows the hint for %s (Anthropic provider)', (model) => {
    render(<TemperatureIgnoredNote model={model} id="temp-note" />);
    // The id is what the slider's aria-describedby points at.
    expect(screen.getByText(HINT).closest('p')).toHaveAttribute('id', 'temp-note');
  });

  it.each([
    ['Claude through OpenRouter', 'openrouter:anthropic/claude-opus-4.1'],
    ['Claude through Vercel', 'vercel:anthropic/claude-sonnet-4'],
    ['another provider', 'openai:gpt-4o'],
    ['the system default (no model picked)', ''],
    ['an endpoint id that only contains "anthropic"', 'my-anthropic:claude-3-5-sonnet-latest'],
    ['a differently cased prefix (the API matches case-sensitively)', 'Anthropic:claude-3-5-sonnet-latest'],
  ])('hides the hint for %s', (_case, model) => {
    const { container } = render(<TemperatureIgnoredNote model={model} id="temp-note" />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe('temperatureIgnoredBy', () => {
  it('names the provider only for an anthropic: prefix', () => {
    expect(temperatureIgnoredBy('anthropic:claude-3-5-haiku-latest')).toBe('Anthropic');
    expect(temperatureIgnoredBy('openrouter:anthropic/claude-3.5-haiku')).toBeNull();
    expect(temperatureIgnoredBy('anthropic')).toBeNull(); // bare id: no provider prefix
    expect(temperatureIgnoredBy(undefined)).toBeNull();
    expect(temperatureIgnoredBy(null)).toBeNull();
  });
});

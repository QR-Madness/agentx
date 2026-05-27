import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';

vi.mock('../../lib/api', () => ({
  api: {
    listModels: vi.fn().mockResolvedValue({
      models: [
        { id: 'anthropic:claude', name: 'Claude', provider: 'anthropic' },
        { id: 'openai:gpt', name: 'GPT', provider: 'openai' },
      ],
    }),
  },
}));

import { ModelSelector, invalidateModelCache } from './ModelSelector';

/**
 * Regression: a parent that setState's inside `onProviderChange` must not loop.
 * The notify effect used to depend on the (inline → unstable) callback, so the
 * setState re-render produced a new callback that re-fired the effect forever
 * ("Maximum update depth exceeded"). The ref-based handler fires only on an
 * actual provider change.
 */
function LoopProneHarness({ onNotify }: { onNotify: () => void }) {
  const [provider, setProvider] = useState('');
  return (
    <div>
      <span data-testid="provider">{provider}</span>
      <ModelSelector
        value=""
        onChange={() => {}}
        onProviderChange={(p) => {
          onNotify();
          setProvider(p); // setState in the handler — the loop trigger
        }}
        compact
      />
    </div>
  );
}

describe('ModelSelector', () => {
  it('does not loop when onProviderChange setState\'s', async () => {
    invalidateModelCache();
    const onNotify = vi.fn();
    render(<LoopProneHarness onNotify={onNotify} />);

    // Provider auto-selects once models load; the handler settles.
    await waitFor(() => expect(screen.getByTestId('provider')).toHaveTextContent('anthropic'));
    // Bounded calls — a loop would be in the thousands (or throw max-depth).
    expect(onNotify.mock.calls.length).toBeLessThanOrEqual(3);
  });
});

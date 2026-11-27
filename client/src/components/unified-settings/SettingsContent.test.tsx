import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { useEffect, useState } from 'react';
import { SettingsContent } from './SettingsContent';

/**
 * Two stand-in sections. The second renders its control only after a delay,
 * the way a real section waits on its lazy chunk and its own settings fetch
 * (Web Search took about 480 ms after mounting on a local server).
 */
function Showing() {
  return <div data-setting="config:showing.key">Showing</div>;
}

function Arriving() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setReady(true), 400);
    return () => clearTimeout(timer);
  }, []);
  return ready
    ? <div data-setting="config:arriving.key">Arriving</div>
    : <p>Loading…</p>;
}

// Read at render time, not when the factory runs, so the stand-ins exist.
vi.mock('./sections', () => ({
  findSectionById: (id: string) => ({
    id,
    label: id,
    component: id === 'showing' ? Showing : Arriving,
  }),
}));

const flashed = (key: string) =>
  document.querySelector(`[data-setting="${key}"]`)?.classList.contains('setting-flash') ?? false;

describe('SettingsContent — landing on a setting', () => {
  it('lands on a setting in another section once that section renders it', async () => {
    // The jump every cross-section link makes: an Overview row, a search hit,
    // the Model Providers nudge. The outgoing section is still exiting when the
    // focus arrives, so the search must follow the section that mounts next,
    // and keep looking until its control has rendered.
    const { rerender } = render(<SettingsContent activeSection="showing" />);
    rerender(
      <SettingsContent activeSection="arriving" focus={{ key: 'config:arriving.key', seq: 1 }} />
    );

    await screen.findByText('Arriving', {}, { timeout: 3000 });
    await waitFor(() => expect(flashed('config:arriving.key')).toBe(true), { timeout: 3000 });
  });

  it('lands on a setting in the section that opens with it', async () => {
    // A deep link that opens Settings straight onto a control: the area isn't
    // mounted when the focus first arrives.
    render(
      <SettingsContent activeSection="showing" focus={{ key: 'config:showing.key', seq: 0 }} />
    );
    await waitFor(() => expect(flashed('config:showing.key')).toBe(true), { timeout: 3000 });
  });
});

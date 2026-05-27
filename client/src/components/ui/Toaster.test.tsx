import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { NotificationProvider, useNotify } from '../../contexts/NotificationContext';
import { Toaster } from './Toaster';

// Toaster portals into #toast-root (created in index.html at runtime); recreate it here.
let toastRoot: HTMLElement;
beforeEach(() => {
  toastRoot = document.createElement('div');
  toastRoot.id = 'toast-root';
  document.body.appendChild(toastRoot);
});
afterEach(() => {
  toastRoot.remove();
});

function Trigger() {
  const { notifyError } = useNotify();
  return <button onClick={() => notifyError({ message: 'Provider down', status: 502, kind: 'upstream' })}>boom</button>;
}

describe('Toaster', () => {
  it('renders an error toast with the backend message and kind-derived title', () => {
    render(
      <NotificationProvider>
        <Trigger />
        <Toaster />
      </NotificationProvider>
    );

    act(() => {
      screen.getByText('boom').click();
    });

    expect(screen.getByText('Provider down')).toBeInTheDocument();
    expect(screen.getByText('Provider unavailable')).toBeInTheDocument(); // titleForKind('upstream')
  });
});

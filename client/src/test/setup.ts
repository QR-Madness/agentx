/**
 * Vitest setup — runs before each test file.
 * Adds jest-dom matchers (toBeInTheDocument, toHaveClass, …) and cleans up the
 * React tree between tests.
 */

import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// jsdom doesn't implement ResizeObserver; Radix components that measure
// (e.g. Slider via @radix-ui/react-use-size) need it present.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

afterEach(() => {
  cleanup();
});

/**
 * Vitest setup — runs before each test file.
 * Adds jest-dom matchers (toBeInTheDocument, toHaveClass, …) and cleans up the
 * React tree between tests.
 */

import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

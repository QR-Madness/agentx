import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge } from './Badge';

describe('Badge', () => {
  it('renders with the default neutral variant', () => {
    render(<Badge>new</Badge>);
    const el = screen.getByText('new');
    expect(el).toHaveClass('ax-badge', 'ax-badge--neutral');
  });

  it('applies the requested variant', () => {
    render(<Badge variant="danger">err</Badge>);
    expect(screen.getByText('err')).toHaveClass('ax-badge--danger');
  });
});

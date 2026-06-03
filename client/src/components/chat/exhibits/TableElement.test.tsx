import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TableElement } from './TableElement';
import type { TableElement as TableElementType } from '../../../lib/exhibits';

const element: TableElementType = {
  type: 'table',
  columns: ['Model', 'Cost'],
  rows: [
    ['opus', '0.40'],
    ['haiku', '0.05'],
  ],
};

function renderTable() {
  return render(<TableElement element={element} messageId="m1" />);
}

describe('TableElement', () => {
  it('renders headers and cells', () => {
    renderTable();
    expect(screen.getByRole('columnheader', { name: /Model/ })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'opus' })).toBeInTheDocument();
  });

  it('sorts rows when a header is clicked', async () => {
    renderTable();
    await userEvent.click(screen.getByRole('button', { name: /Cost/ }));
    const firstDataCell = screen.getAllByRole('row')[1];
    // ascending by Cost -> haiku (0.05) first
    expect(within(firstDataCell).getByText('haiku')).toBeInTheDocument();
  });

  it('opens a modal on Expand', async () => {
    renderTable();
    await userEvent.click(screen.getByRole('button', { name: /Expand/ }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});

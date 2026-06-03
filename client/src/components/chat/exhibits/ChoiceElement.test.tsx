import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChoiceElement } from './ChoiceElement';
import type { ChoiceElement as ChoiceElementType } from '../../../lib/exhibits';

const element: ChoiceElementType = {
  type: 'choice',
  prompt: 'Which DB?',
  options: ['PostgreSQL', 'Neo4j'],
};

describe('ChoiceElement', () => {
  it('submits the clicked option with the owning message id', async () => {
    const onSubmitChoice = vi.fn();
    render(
      <ChoiceElement element={element} messageId="m1" onSubmitChoice={onSubmitChoice} />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Neo4j' }));
    expect(onSubmitChoice).toHaveBeenCalledWith('Neo4j', 'm1');
  });

  it('renders resolved (disabled, chosen marked) once answered', () => {
    render(
      <ChoiceElement element={element} messageId="m1" answeredValue="Neo4j" />,
    );
    const chosen = screen.getByRole('button', { name: 'Neo4j' });
    const other = screen.getByRole('button', { name: 'PostgreSQL' });
    expect(chosen).toBeDisabled();
    expect(other).toBeDisabled();
    expect(chosen).toHaveAttribute('aria-pressed', 'true');
    expect(other).toHaveAttribute('aria-pressed', 'false');
  });

  it('is inert while a turn is in flight (busy)', async () => {
    const onSubmitChoice = vi.fn();
    render(
      <ChoiceElement element={element} messageId="m1" busy onSubmitChoice={onSubmitChoice} />,
    );
    const btn = screen.getByRole('button', { name: 'Neo4j' });
    expect(btn).toBeDisabled();
    await userEvent.click(btn);
    expect(onSubmitChoice).not.toHaveBeenCalled();
  });
});

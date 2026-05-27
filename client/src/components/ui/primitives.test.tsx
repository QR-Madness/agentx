import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Switch } from './Switch';
import { Checkbox } from './Checkbox';
import { Label } from './Label';
import { Select, SelectTrigger, SelectValue } from './Select';

describe('Switch', () => {
  it('renders a switch reflecting checked state', () => {
    render(<Switch checked aria-label="wifi" onCheckedChange={() => {}} />);
    const sw = screen.getByRole('switch', { name: 'wifi' });
    expect(sw).toBeInTheDocument();
    expect(sw).toHaveAttribute('data-state', 'checked');
  });
});

describe('Checkbox', () => {
  it('renders an unchecked checkbox by default', () => {
    render(<Checkbox aria-label="agree" />);
    const cb = screen.getByRole('checkbox', { name: 'agree' });
    expect(cb).toBeInTheDocument();
    expect(cb).toHaveAttribute('data-state', 'unchecked');
  });
});

describe('Label', () => {
  it('renders its text and associates via htmlFor', () => {
    render(<Label htmlFor="name">Name</Label>);
    const label = screen.getByText('Name');
    expect(label).toHaveAttribute('for', 'name');
  });
});

describe('Select', () => {
  it('renders a closed trigger with placeholder', () => {
    render(
      <Select>
        <SelectTrigger aria-label="model">
          <SelectValue placeholder="Pick a model" />
        </SelectTrigger>
      </Select>
    );
    expect(screen.getByText('Pick a model')).toBeInTheDocument();
  });
});

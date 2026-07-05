import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Switch } from './Switch';
import { Checkbox } from './Checkbox';
import { Label } from './Label';
import { Select, SelectTrigger, SelectValue } from './Select';
import { Slider } from './Slider';
import { IconButton } from './IconButton';
import { StatusDot } from './StatusDot';
import { Input } from './Field';

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

describe('Slider', () => {
  it('renders a slider reflecting its value', () => {
    render(<Slider value={[0.3]} min={0} max={1} step={0.1} aria-label="weight" />);
    const slider = screen.getByRole('slider', { name: 'weight' });
    expect(slider).toBeInTheDocument();
    expect(slider).toHaveAttribute('aria-valuenow', '0.3');
  });
});

describe('IconButton', () => {
  it('renders a transparent-chrome icon button with required aria-label', () => {
    render(<IconButton aria-label="Delete file" />);
    const btn = screen.getByRole('button', { name: 'Delete file' });
    expect(btn).toHaveClass('ax-iconbtn');
    expect(btn).toHaveAttribute('type', 'button');
  });

  it('applies size, tone and active variant classes', () => {
    render(<IconButton aria-label="Retry" size="sm" tone="danger" active />);
    const btn = screen.getByRole('button', { name: 'Retry' });
    expect(btn).toHaveClass('ax-iconbtn--sm');
    expect(btn).toHaveClass('ax-iconbtn--danger');
    expect(btn).toHaveClass('ax-iconbtn--active');
  });
});

describe('StatusDot', () => {
  it('renders a decorative dot with the tone class', () => {
    const { container } = render(<StatusDot tone="warning" />);
    const dot = container.querySelector('.status-dot');
    expect(dot).toHaveClass('warning');
    expect(dot).toHaveAttribute('aria-hidden', 'true');
  });
});

describe('Input icon slot', () => {
  it('keeps bare-input markup when no icon is given', () => {
    const { container } = render(<Input aria-label="Name" />);
    expect(container.querySelector('input')).toHaveClass('ax-field');
    expect(container.querySelector('.ax-inputwrap')).toBeNull();
  });

  it('switches to the field-wrapper layout with an icon', () => {
    const { container } = render(<Input aria-label="Search" icon={<svg data-testid="i" />} />);
    expect(container.querySelector('.ax-inputwrap')).not.toBeNull();
    expect(container.querySelector('input')).toHaveClass('ax-inputwrap__input');
  });
});

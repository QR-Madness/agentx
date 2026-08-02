import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SliderField } from './SliderField';
import { NumberField } from './NumberField';
import { ToggleField } from './ToggleField';
import { PromptField } from './PromptField';
import { SelectField } from './SelectField';

describe('SliderField', () => {
  it('renders label + formatted readout and exposes a slider', () => {
    render(<SliderField label="Temperature" value={0.2} min={0} max={1} step={0.05} onChange={vi.fn()} />);
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('0.20')).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveAttribute('aria-valuenow', '0.2');
  });
});

describe('NumberField', () => {
  it('parses input and applies the fallback on empty', () => {
    const onChange = vi.fn();
    render(<NumberField label="Max Tokens" value={2000} fallback={2000} onChange={onChange} />);
    const input = screen.getByDisplayValue('2000');
    fireEvent.change(input, { target: { value: '500' } });
    expect(onChange).toHaveBeenLastCalledWith(500);
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(2000);
  });
});

describe('ToggleField', () => {
  it('renders label + badge and fires onChange', () => {
    const onChange = vi.fn();
    render(
      <ToggleField
        checked={false}
        onChange={onChange}
        label="Hybrid Search"
        badge={{ text: 'Recommended', variant: 'success' }}
        hint="combines keyword + vector"
      />
    );
    expect(screen.getByText('Hybrid Search')).toBeInTheDocument();
    expect(screen.getByText('Recommended')).toBeInTheDocument();
    expect(screen.getByText('combines keyword + vector')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: /Hybrid Search/ }));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe('SelectField', () => {
  const depthOptions = [
    { value: '', label: 'Provider default' },
    { value: 'basic', label: 'Basic' },
  ];

  it('renders an empty-valued option without throwing', () => {
    render(
      <SelectField label="Search Depth" value="" options={depthOptions} onChange={vi.fn()} />
    );
    expect(screen.getByText('Search Depth')).toBeInTheDocument();
    expect(screen.getByText('Provider default')).toBeInTheDocument();
  });

  it('reports the empty value back to the caller as an empty string', () => {
    const onChange = vi.fn();
    render(
      <SelectField label="Search Depth" value="basic" options={depthOptions} onChange={onChange} />
    );
    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('option', { name: 'Provider default' }));
    expect(onChange).toHaveBeenCalledWith('');
  });
});

describe('PromptField', () => {
  it('shows the empty hint and calls onReset', () => {
    const onReset = vi.fn();
    render(<PromptField label="System Prompt" value="" onChange={vi.fn()} onReset={onReset} />);
    expect(screen.getByText('Leave empty to use default prompt')).toBeInTheDocument();
    fireEvent.click(screen.getByTitle('Reset to default'));
    expect(onReset).toHaveBeenCalled();
  });
});

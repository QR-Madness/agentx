import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SliderField } from './SliderField';
import { NumberField } from './NumberField';
import { ToggleField } from './ToggleField';
import { PromptField } from './PromptField';
import { SelectField } from './SelectField';
import { TextField } from './TextField';

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

describe('field accessibility', () => {
  // Five of the eight primitives rendered a <Label> with no htmlFor and a
  // control with no id, so the label announced nothing. NumberField was worse:
  // its accessible name resolved to `title` — the hint sentence, not the
  // setting's name. These assert the label is what gets announced.
  it('NumberField is named by its label, not its hint', () => {
    render(
      <NumberField
        label="Max Results"
        value={5}
        title="Results returned per search (1–20)"
        onChange={vi.fn()}
      />
    );
    expect(screen.getByRole('spinbutton', { name: 'Max Results' })).toBeInTheDocument();
  });

  it('TextField is named by its label', () => {
    render(<TextField label="Cross-Encoder Model" value="bge" onChange={vi.fn()} />);
    expect(screen.getByRole('textbox', { name: 'Cross-Encoder Model' })).toBeInTheDocument();
  });

  it('SelectField is named by its label', () => {
    render(
      <SelectField
        label="Search Depth"
        value="basic"
        options={[{ value: 'basic', label: 'Basic' }]}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByRole('combobox', { name: 'Search Depth' })).toBeInTheDocument();
  });

  it('describes a control by its hint', () => {
    render(<NumberField label="Candidate Pool" value={50} hint="How many the reranker scores" onChange={vi.fn()} />);
    expect(screen.getByRole('spinbutton', { name: 'Candidate Pool' }))
      .toHaveAccessibleDescription('How many the reranker scores');
  });
});

describe('manifest chrome', () => {
  const binding = {
    entry: {
      key: 'recall_candidate_pool',
      store: 'memory' as const,
      type: 'int',
      default: 50,
      value: 120,
      secret: false,
      writable_via: '/api/memory/recall-settings',
    },
    defaultValue: 50,
    isModified: true,
    min: 10,
    max: 200,
    help: { summary: 'How many candidates the reranker scores.' },
  };

  it('marks a changed setting and resets it to the shipped default', () => {
    const onReset = vi.fn();
    render(
      <NumberField label="Candidate Pool" value={120} binding={binding} onReset={onReset} onChange={vi.fn()} />
    );
    expect(screen.getByLabelText('Changed from the default')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reset Candidate Pool to default' }));
    expect(onReset).toHaveBeenCalled();
  });

  it('takes bounds from the manifest when the caller gives none', () => {
    render(<NumberField label="Candidate Pool" value={120} binding={binding} onChange={vi.fn()} />);
    const input = screen.getByRole('spinbutton', { name: 'Candidate Pool' });
    expect(input).toHaveAttribute('min', '10');
    expect(input).toHaveAttribute('max', '200');
  });

  it('offers help when the setting has authored prose', () => {
    render(<NumberField label="Candidate Pool" value={50} binding={binding} onChange={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'About Candidate Pool' })).toBeInTheDocument();
  });

  it('renders nothing extra for an unbound control', () => {
    render(<NumberField label="Candidate Pool" value={50} onChange={vi.fn()} />);
    expect(screen.queryByLabelText('Changed from the default')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^About/ })).not.toBeInTheDocument();
  });
});

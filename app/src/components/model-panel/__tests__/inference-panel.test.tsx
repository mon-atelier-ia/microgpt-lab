import { render, cleanup } from '@testing-library/react';
import { describe, it, expect, afterEach } from 'vitest';
import { InferencePanel } from '../inference-panel';

afterEach(cleanup);

describe('InferencePanel', () => {
  it('shows placeholder text when words is empty', () => {
    const { getByText } = render(<InferencePanel words={[]} colorVar="a" />);
    expect(getByText(/Générez des mots après l'entraînement/i)).toBeDefined();
  });

  it('renders list items for each word', () => {
    const { getAllByRole } = render(
      <InferencePanel words={['alice', 'bob', 'carol']} colorVar="a" />,
    );
    const items = getAllByRole('listitem');
    expect(items).toHaveLength(3);
    expect(items[0].textContent).toBe('alice');
    expect(items[1].textContent).toBe('bob');
    expect(items[2].textContent).toBe('carol');
  });

  it('shows correct word count in footer', () => {
    const { getByRole } = render(<InferencePanel words={['alice', 'bob']} colorVar="b" />);
    const status = getByRole('status');
    expect(status.textContent).toContain('2 mots');
  });

  it('shows singular mot when one word', () => {
    const { getByRole } = render(<InferencePanel words={['alice']} colorVar="a" />);
    const status = getByRole('status');
    expect(status.textContent).toContain('1 mot');
    expect(status.textContent).not.toContain('1 mots');
  });

  it('shows temperature when provided', () => {
    const { getByRole } = render(<InferencePanel words={[]} colorVar="a" temperature={0.75} />);
    const status = getByRole('status');
    expect(status.textContent).toContain('t=0.75');
  });

  it('does not show temperature when not provided', () => {
    const { getByRole } = render(<InferencePanel words={[]} colorVar="a" />);
    const status = getByRole('status');
    expect(status.textContent).not.toContain('t=');
  });
});

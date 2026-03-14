import { render, cleanup } from '@testing-library/react';
import { describe, it, expect, afterEach } from 'vitest';
import { InferencePanel } from '../inference-panel';
import type { FeedbackResult } from '../../../lib/training-feedback';

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

function makeFeedback(level: FeedbackResult['level'], message: string, hint = ''): FeedbackResult {
  return { level, message, hint, memorization: 0, quality: 0, diversity: 0 };
}

function renderWithFeedback(level: FeedbackResult['level'], message: string) {
  return render(
    <InferencePanel words={['x']} colorVar="a" feedback={makeFeedback(level, message)} />,
  );
}

const BADGE_REGEX = /modèle|généralisation|créativité|capacité|bruit/i;
const SWEET_SPOT_MSG = 'Bonne généralisation !';

describe('InferencePanel — feedback badge', () => {
  it('shows no badge when feedback is null', () => {
    const { queryByText } = render(
      <InferencePanel words={['alice']} colorVar="a" feedback={null} />,
    );
    expect(queryByText(BADGE_REGEX)).toBeNull();
  });

  it('shows no badge when level is untrained', () => {
    const { queryByText } = render(
      <InferencePanel words={[]} colorVar="a" feedback={makeFeedback('untrained', '')} />,
    );
    expect(queryByText(BADGE_REGEX)).toBeNull();
  });

  it('shows random badge', () => {
    const msg = '🎲 Le modèle génère du bruit';
    const { getByText } = renderWithFeedback('random', msg);
    expect(getByText(msg)).toBeDefined();
  });

  it('shows learning badge', () => {
    const msg = '📈 Le modèle apprend…';
    const { getByText } = renderWithFeedback('learning', msg);
    expect(getByText(msg)).toBeDefined();
  });

  it('shows sweet-spot with message, hint, and ARIA', () => {
    const hint = 'Bravo ! Mots crédibles !';
    const { getByText, getAllByRole } = render(
      <InferencePanel
        words={['x']}
        colorVar="a"
        feedback={makeFeedback('sweet-spot', SWEET_SPOT_MSG, hint)}
      />,
    );
    expect(getByText(SWEET_SPOT_MSG)).toBeDefined();
    expect(getByText(hint)).toBeDefined();
    const statuses = getAllByRole('status');
    const badgeStatus = statuses.find((s) => s.textContent?.includes(SWEET_SPOT_MSG));
    expect(badgeStatus).toBeDefined();
    expect(badgeStatus?.getAttribute('aria-live')).toBe('polite');
  });

  it('shows low-diversity badge', () => {
    const msg = '🔁 Manque de créativité';
    const { getByText } = renderWithFeedback('low-diversity', msg);
    expect(getByText(msg)).toBeDefined();
  });

  it('shows overfitting badge', () => {
    const msg = '🧠 Le modèle mémorise';
    const { getByText } = renderWithFeedback('overfitting', msg);
    expect(getByText(msg)).toBeDefined();
  });

  it('shows underpowered badge', () => {
    const msg = '⚡ Capacité limitée';
    const { getByText } = renderWithFeedback('underpowered', msg);
    expect(getByText(msg)).toBeDefined();
  });
});

import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ErrorBanner } from '../error-banner';

describe('ErrorBanner', () => {
  it('returns null when message is null', () => {
    const { container } = render(<ErrorBanner message={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders with role="alert" when message is provided', () => {
    render(<ErrorBanner message="Something went wrong" />);
    const alert = screen.getByRole('alert');
    expect(alert).toBeDefined();
    expect(alert.textContent).toBe('Something went wrong');
  });
});

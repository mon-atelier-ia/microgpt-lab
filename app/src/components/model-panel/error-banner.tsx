type ErrorBannerProps = { message: string | null };

export function ErrorBanner({ message }: ErrorBannerProps) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className="rounded px-3 py-2 text-xs font-medium"
      style={{ background: 'var(--error)', color: 'var(--surface-0)' }}
    >
      {message}
    </div>
  );
}

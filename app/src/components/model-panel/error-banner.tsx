type ErrorBannerProps = { message: string | null };

export function ErrorBanner({ message }: ErrorBannerProps) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded bg-error px-3 py-2 text-xs font-medium text-surface-0">
      {message}
    </div>
  );
}

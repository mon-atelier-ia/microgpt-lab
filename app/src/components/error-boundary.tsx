import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

type Props = { children: ReactNode };
type State = { error: Error | null; resetKey: number };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, resetKey: 0 };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  handleReset = () => {
    this.setState((prev) => ({ error: null, resetKey: prev.resetKey + 1 }));
  };

  render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          className="flex min-h-screen flex-col items-center justify-center gap-4 bg-surface-0 p-8 text-text-primary"
        >
          <h1 className="text-lg font-bold text-error">Une erreur est survenue</h1>
          <p className="max-w-md text-center text-sm text-text-secondary">
            {this.state.error.message}
          </p>
          <button
            className="rounded-md bg-surface-1 px-4 py-2 text-sm hover:bg-surface-2"
            onClick={this.handleReset}
          >
            Réessayer
          </button>
        </div>
      );
    }
    // resetKey forces full child remount → workers re-initialize
    return <div key={this.state.resetKey}>{this.props.children}</div>;
  }
}

import '@testing-library/react';

// jsdom does not implement ResizeObserver (used by Radix UI)
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

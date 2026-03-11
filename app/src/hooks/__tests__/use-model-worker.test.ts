import { renderHook, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { StepResult } from '../../lib/types';

// ---------------------------------------------------------------------------
// Mock PRESETS so initModel doesn't need real data files
// ---------------------------------------------------------------------------
vi.mock('../../data/presets', () => ({
  PRESETS: [
    {
      id: 'test-preset',
      name: 'Test',
      description: 'Test preset',
      load: async () => ['word1', 'word2'],
    },
  ],
}));

// ---------------------------------------------------------------------------
// MockWorker — captures onmessage, records postMessage / terminate calls
// ---------------------------------------------------------------------------
class MockWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();

  simulateMessage(data: unknown) {
    this.onmessage?.({ data } as MessageEvent);
  }
}

let mockWorkerInstance: MockWorker;

beforeEach(() => {
  mockWorkerInstance = new MockWorker();
  function WorkerStub(_url: unknown, _opts?: unknown) {
    return mockWorkerInstance;
  }
  vi.stubGlobal('Worker', WorkerStub);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

async function getHook() {
  const { useModelWorker } = await import('../use-model-worker');
  return useModelWorker;
}

// ---------------------------------------------------------------------------
// Outbound messages (hook → worker)
// ---------------------------------------------------------------------------
describe('useModelWorker — outbound messages', () => {
  it('initModel sends { type: init, datasetText, config }', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    const params = {
      datasetId: 'test-preset',
      n_embd: 32,
      n_head: 2,
      n_layer: 2,
      block_size: 8,
      lr: 0.01,
      temperature: 1.0,
      trainSteps: 100,
    };
    await act(async () => {
      await result.current.initModel(params);
    });
    expect(mockWorkerInstance.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'init', datasetText: 'word1\nword2', config: params }),
    );
  });

  it('train sets trainState to "training" and sends { type: train, n_steps }', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      result.current.train(50);
    });
    expect(result.current.trainState).toBe('training');
    expect(mockWorkerInstance.postMessage).toHaveBeenCalledWith({ type: 'train', n_steps: 50 });
  });

  it('generate sends { type: generate, temperature, n_samples }', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      result.current.generate(0.8, 10);
    });
    expect(mockWorkerInstance.postMessage).toHaveBeenCalledWith({
      type: 'generate',
      temperature: 0.8,
      n_samples: 10,
    });
  });

  it('setLr sends { type: set_lr, lr }', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      result.current.setLr(0.001);
    });
    expect(mockWorkerInstance.postMessage).toHaveBeenCalledWith({ type: 'set_lr', lr: 0.001 });
  });

  it('on cleanup sends { type: dispose } and calls terminate()', async () => {
    const useModelWorker = await getHook();
    const { unmount } = renderHook(() => useModelWorker());
    unmount();
    expect(mockWorkerInstance.postMessage).toHaveBeenCalledWith({ type: 'dispose' });
    expect(mockWorkerInstance.terminate).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Inbound messages (worker → hook state)
// ---------------------------------------------------------------------------
describe('useModelWorker — inbound messages', () => {
  it('{ type: step } updates the steps array after buffer flush', async () => {
    vi.useFakeTimers();
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    const stepData: StepResult = { step: 1, loss: 2.5, word: 'foo', lr: 0.01 };
    act(() => {
      mockWorkerInstance.simulateMessage({ type: 'step', data: stepData });
    });
    await act(async () => {
      vi.advanceTimersByTime(150);
    });
    expect(result.current.steps).toHaveLength(1);
    expect(result.current.steps[0]).toEqual(stepData);
    vi.useRealTimers();
  });

  it('{ type: train_done } sets trainState to "trained"', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      mockWorkerInstance.simulateMessage({ type: 'train_done' });
    });
    expect(result.current.trainState).toBe('trained');
  });

  it('{ type: generated } updates words array', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      mockWorkerInstance.simulateMessage({ type: 'generated', words: ['alpha', 'beta'] });
    });
    expect(result.current.words).toEqual(['alpha', 'beta']);
  });

  it('{ type: error } sets errorMessage and trainState to "error"', async () => {
    const useModelWorker = await getHook();
    const { result } = renderHook(() => useModelWorker());
    act(() => {
      mockWorkerInstance.simulateMessage({ type: 'error', message: 'something broke' });
    });
    expect(result.current.trainState).toBe('error');
    expect(result.current.errorMessage).toBe('something broke');
  });
});

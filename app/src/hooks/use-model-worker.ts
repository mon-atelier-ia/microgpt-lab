import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from 'react';
import type {
  ModelParams,
  StepResult,
  TrainState,
  WorkerMessage,
  WorkerResponse,
} from '../lib/types';
import { PRESETS } from '../data/presets';

const MAX_STEPS = 5000;

type StepBuffer = MutableRefObject<StepResult[]>;
type TimerRef = MutableRefObject<ReturnType<typeof setTimeout> | null>;

function useStepBuffer(setSteps: Dispatch<SetStateAction<StepResult[]>>) {
  const stepBufferRef = useRef<StepResult[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushSteps = useCallback(() => {
    flushTimerRef.current = null;
    const buffer = stepBufferRef.current;
    if (buffer.length === 0) return;
    stepBufferRef.current = [];
    setSteps((prev) => {
      const merged = [...prev, ...buffer];
      return merged.length > MAX_STEPS ? merged.slice(merged.length - MAX_STEPS) : merged;
    });
  }, [setSteps]);

  const clearBuffer = useCallback(() => {
    stepBufferRef.current = [];
    if (flushTimerRef.current !== null) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  return { stepBufferRef, flushTimerRef, flushSteps, clearBuffer };
}

type WorkerDeps = {
  setTrainState: (s: TrainState) => void;
  setSteps: Dispatch<SetStateAction<StepResult[]>>;
  setWords: (w: string[]) => void;
  setErrorMessage: (m: string | null) => void;
  stepBufferRef: StepBuffer;
  flushTimerRef: TimerRef;
  flushSteps: () => void;
};

function makeHandler(d: WorkerDeps) {
  return (e: MessageEvent<WorkerResponse>) => {
    const msg = e.data;
    switch (msg.type) {
      case 'ready':
        d.setTrainState('idle');
        break;
      case 'step':
        d.stepBufferRef.current.push(msg.data);
        if (d.flushTimerRef.current === null) {
          d.flushTimerRef.current = setTimeout(d.flushSteps, 100);
        }
        break;
      case 'train_done':
        d.flushSteps();
        d.setTrainState('trained');
        break;
      case 'generated':
        d.setWords(msg.words);
        break;
      case 'error':
        d.setTrainState('error');
        d.setErrorMessage(msg.message);
        console.error('Worker error:', msg.message);
        break;
    }
  };
}

export function useModelWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [trainState, setTrainState] = useState<TrainState>('idle');
  const [steps, setSteps] = useState<StepResult[]>([]);
  const [words, setWords] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const { stepBufferRef, flushTimerRef, flushSteps, clearBuffer } = useStepBuffer(setSteps);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/model-worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    worker.onmessage = makeHandler({
      setTrainState,
      setSteps,
      setWords,
      setErrorMessage,
      stepBufferRef,
      flushTimerRef,
      flushSteps,
    });
    return () => {
      clearBuffer();
      worker.postMessage({ type: 'dispose' });
      worker.terminate();
    };
  }, [flushSteps, clearBuffer, stepBufferRef, flushTimerRef]);

  const send = useCallback((msg: WorkerMessage) => {
    workerRef.current?.postMessage(msg);
  }, []);

  const initModel = useCallback(
    async (params: ModelParams) => {
      const preset = PRESETS.find((p) => p.id === params.datasetId);
      if (!preset) return;
      clearBuffer();
      setSteps([]);
      setWords([]);
      setTrainState('idle');
      setErrorMessage(null);
      const data = await preset.load();
      send({ type: 'init', datasetText: data.join('\n'), config: params });
    },
    [send, clearBuffer],
  );

  const train = useCallback(
    (n_steps: number) => {
      setTrainState('training');
      send({ type: 'train', n_steps });
    },
    [send],
  );

  const setLr = useCallback(
    (lr: number) => {
      send({ type: 'set_lr', lr });
    },
    [send],
  );

  const generate = useCallback(
    (temperature: number, n_samples: number) => {
      send({ type: 'generate', temperature, n_samples });
    },
    [send],
  );

  return { trainState, steps, words, errorMessage, initModel, train, setLr, generate };
}

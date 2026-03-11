import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ModelParams,
  StepResult,
  TrainState,
  WorkerMessage,
  WorkerResponse,
} from '../lib/types';
import { PRESETS } from '../data/presets';

export function useModelWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [trainState, setTrainState] = useState<TrainState>('idle');
  const [steps, setSteps] = useState<StepResult[]>([]);
  const [words, setWords] = useState<string[]>([]);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/model-worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      switch (msg.type) {
        case 'ready':
          setTrainState('idle');
          break;
        case 'step':
          setSteps((prev) => [...prev, msg.data]);
          break;
        case 'train_done':
          setTrainState('trained');
          break;
        case 'generated':
          setWords(msg.words);
          break;
        case 'error':
          setTrainState('error');
          console.error('Worker error:', msg.message);
          break;
      }
    };

    return () => worker.terminate();
  }, []);

  const send = useCallback((msg: WorkerMessage) => {
    workerRef.current?.postMessage(msg);
  }, []);

  const initModel = useCallback(
    (params: ModelParams) => {
      const preset = PRESETS.find((p) => p.id === params.datasetId);
      if (!preset) return;
      setSteps([]);
      setWords([]);
      setTrainState('idle');
      send({
        type: 'init',
        datasetText: preset.data.join('\n'),
        config: params,
      });
    },
    [send],
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

  return { trainState, steps, words, initModel, train, setLr, generate };
}

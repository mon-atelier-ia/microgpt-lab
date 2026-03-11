import { useModelWorker } from '../hooks/use-model-worker';
import { ModelPanel } from './model-panel/model-panel';

export type CompareViewProps = {
  workerHandleA: ReturnType<typeof useModelWorker>;
};

export function CompareView({ workerHandleA }: CompareViewProps) {
  const workerHandleB = useModelWorker();

  return (
    <div style={{ display: 'flex', gap: '1rem' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <ModelPanel layout="vertical" colorVar="a" workerHandle={workerHandleA} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <ModelPanel layout="vertical" colorVar="b" workerHandle={workerHandleB} />
      </div>
    </div>
  );
}

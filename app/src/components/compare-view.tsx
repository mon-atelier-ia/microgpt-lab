import { useModelWorker } from '../hooks/use-model-worker';
import { ModelPanel } from './model-panel/model-panel';

export type CompareViewProps = {
  workerHandleA: ReturnType<typeof useModelWorker>;
};

export function CompareView({ workerHandleA }: CompareViewProps) {
  const workerHandleB = useModelWorker();

  return (
    <div className="flex flex-col gap-4 md:flex-row md:gap-4">
      <div className="min-w-0 md:flex-1">
        <ModelPanel layout="vertical" colorVar="a" workerHandle={workerHandleA} />
      </div>
      <div className="min-w-0 md:flex-1">
        <ModelPanel layout="vertical" colorVar="b" workerHandle={workerHandleB} />
      </div>
    </div>
  );
}

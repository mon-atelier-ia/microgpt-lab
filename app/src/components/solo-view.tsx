import { type WorkerHandle } from '../hooks/use-model-worker';
import { ModelPanel } from './model-panel/model-panel';

export type SoloViewProps = {
  workerHandle: WorkerHandle;
};

export function SoloView({ workerHandle }: SoloViewProps) {
  return <ModelPanel layout="horizontal" colorVar="a" workerHandle={workerHandle} />;
}

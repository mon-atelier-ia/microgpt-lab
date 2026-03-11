import { useModelWorker } from '../hooks/use-model-worker';
import { ModelPanel } from './model-panel/model-panel';

export type SoloViewProps = {
  workerHandle: ReturnType<typeof useModelWorker>;
};

export function SoloView({ workerHandle }: SoloViewProps) {
  return <ModelPanel layout="horizontal" colorVar="a" workerHandle={workerHandle} />;
}

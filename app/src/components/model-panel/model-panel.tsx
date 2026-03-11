import { useEffect, useState } from 'react';
import type { ModelParams } from '../../lib/types';
import { DEFAULT_PARAMS } from '../../lib/types';
import { useModelWorker } from '../../hooks/use-model-worker';
import { ParamsPanel } from './params-panel';
import { LossPanel } from './loss-panel';
import { InferencePanel } from './inference-panel';
import { ErrorBanner } from './error-banner';
import { Button } from '../ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogTitle,
} from '../ui/alert-dialog';

type WorkerHandle = ReturnType<typeof useModelWorker>;

export type ModelPanelProps = {
  colorVar: 'a' | 'b';
  layout: 'horizontal' | 'vertical';
  workerHandle?: WorkerHandle;
};

export function ModelPanel({ colorVar, layout, workerHandle }: ModelPanelProps) {
  if (workerHandle) {
    return <ModelPanelInner colorVar={colorVar} layout={layout} handle={workerHandle} />;
  }
  return <ModelPanelWithOwnWorker colorVar={colorVar} layout={layout} />;
}

function ModelPanelWithOwnWorker(props: {
  colorVar: 'a' | 'b';
  layout: 'horizontal' | 'vertical';
}) {
  const handle = useModelWorker();
  return <ModelPanelInner {...props} handle={handle} />;
}

function ModelPanelInner({
  colorVar,
  layout,
  handle,
}: {
  colorVar: 'a' | 'b';
  layout: 'horizontal' | 'vertical';
  handle: WorkerHandle;
}) {
  const { trainState, steps, words, errorMessage, initModel, train, setLr, generate } = handle;

  const [params, setParams] = useState<ModelParams>({ ...DEFAULT_PARAMS });
  const [pendingArch, setPendingArch] = useState<ModelParams | null>(null);

  // Auto-init on mount with default params
  useEffect(() => {
    initModel(params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleParamsChange(next: ModelParams) {
    const lrChanged = next.lr !== params.lr;
    const archChanged =
      next.datasetId !== params.datasetId ||
      next.n_embd !== params.n_embd ||
      next.n_head !== params.n_head ||
      next.n_layer !== params.n_layer ||
      next.block_size !== params.block_size;

    if (archChanged && trainState === 'trained') {
      setPendingArch(next);
      return;
    }
    if (archChanged) {
      setParams(next);
      initModel(next);
    } else {
      setParams(next);
      if (lrChanged) {
        setLr(next.lr);
      }
    }
  }

  function confirmArchChange() {
    if (pendingArch) {
      setParams(pendingArch);
      initModel(pendingArch);
      setPendingArch(null);
    }
  }

  const isHorizontal = layout === 'horizontal';

  const containerClass = isHorizontal ? 'flex flex-row gap-2 min-w-0' : 'flex flex-col gap-2';

  return (
    <div className={containerClass}>
      <ErrorBanner message={errorMessage} />
      <div className={isHorizontal ? 'flex-[0_0_40%] min-w-0' : 'flex-[0_0_35%]'}>
        <ParamsPanel
          params={params}
          onParamsChange={handleParamsChange}
          onTrain={() => train(params.trainSteps)}
          onGenerate={() => generate(params.temperature, 10)}
          trainState={trainState}
          colorVar={colorVar}
        />
      </div>
      <div className={isHorizontal ? 'flex-[0_0_25%] min-w-0' : 'flex-[0_0_25%]'}>
        <LossPanel steps={steps} colorVar={colorVar} />
      </div>
      <div className={isHorizontal ? 'flex-[1_1_35%] min-w-0' : 'flex-[1_1_40%]'}>
        <InferencePanel words={words} colorVar={colorVar} temperature={params.temperature} />
      </div>

      <AlertDialog
        open={pendingArch !== null}
        onOpenChange={(open) => !open && setPendingArch(null)}
      >
        <AlertDialogContent>
          <AlertDialogTitle className="text-sm font-semibold">Reset training?</AlertDialogTitle>
          <AlertDialogDescription className="mt-2 text-xs text-text-secondary">
            Changing architecture parameters will reset all training progress.
          </AlertDialogDescription>
          <div className="mt-4 flex justify-end gap-2">
            <AlertDialogCancel asChild>
              <Button size="sm" variant="ghost">
                Cancel
              </Button>
            </AlertDialogCancel>
            <AlertDialogAction asChild>
              <Button size="sm" onClick={confirmArchChange} className="bg-error text-surface-0">
                Reset &amp; apply
              </Button>
            </AlertDialogAction>
          </div>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

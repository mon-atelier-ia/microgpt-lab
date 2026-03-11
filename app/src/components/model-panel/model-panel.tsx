import { useEffect, useRef, useState } from 'react';
import type { ColorVar, ModelParams } from '../../lib/types';
import { DEFAULT_N_SAMPLES } from '../../lib/constants';
import { isArchChange } from '../../lib/validation';
import { useModelWorker } from '../../hooks/use-model-worker';
import type { WorkerHandle } from '../../hooks/use-model-worker';
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

export type ModelPanelProps = {
  colorVar: ColorVar;
  layout: 'horizontal' | 'vertical';
  workerHandle?: WorkerHandle;
};

export function ModelPanel({ colorVar, layout, workerHandle }: ModelPanelProps) {
  if (workerHandle) {
    return <ModelPanelInner colorVar={colorVar} layout={layout} handle={workerHandle} />;
  }
  return <ModelPanelWithOwnWorker colorVar={colorVar} layout={layout} />;
}

type ModelPanelInnerProps = {
  colorVar: ColorVar;
  layout: 'horizontal' | 'vertical';
  handle: WorkerHandle;
};

function ModelPanelWithOwnWorker(props: Omit<ModelPanelInnerProps, 'handle'>) {
  const handle = useModelWorker();
  return <ModelPanelInner {...props} handle={handle} />;
}

function ModelPanelInner({ colorVar, layout, handle }: ModelPanelInnerProps) {
  const {
    trainState,
    steps,
    words,
    errorMessage,
    params,
    setParams,
    initModel,
    train,
    setLr,
    generate,
  } = handle;
  const [pendingArch, setPendingArch] = useState<ModelParams | null>(null);
  const initialized = useRef(false);

  // Auto-init on mount with default params
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      initModel(params);
    }
  }, [initModel, params]);

  function handleParamsChange(next: ModelParams) {
    const lrChanged = next.lr !== params.lr;
    const archChanged = isArchChange(params, next);

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
  const containerClass = isHorizontal ? 'flex flex-row gap-3 min-w-0' : 'flex flex-col gap-3';
  const glowClass = colorVar === 'a' ? 'panel-glow-a' : 'panel-glow-b';

  return (
    <div className={containerClass}>
      <ErrorBanner message={errorMessage} />
      <div className={isHorizontal ? 'flex-[0_0_40%] min-w-0' : 'flex-[0_0_35%]'}>
        <ParamsPanel
          params={params}
          onParamsChange={handleParamsChange}
          onTrain={() => train(params.trainSteps)}
          onGenerate={() => generate(params.temperature, DEFAULT_N_SAMPLES)}
          trainState={trainState}
          colorVar={colorVar}
          glowClass={glowClass}
        />
      </div>
      <div className={isHorizontal ? 'flex-[0_0_25%] min-w-0' : 'flex-[0_0_25%]'}>
        <LossPanel steps={steps} colorVar={colorVar} glowClass={glowClass} />
      </div>
      <div className={isHorizontal ? 'flex-[1_1_35%] min-w-0' : 'flex-[1_1_40%]'}>
        <InferencePanel
          words={words}
          colorVar={colorVar}
          temperature={params.temperature}
          glowClass={glowClass}
        />
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

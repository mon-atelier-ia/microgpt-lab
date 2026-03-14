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

function ConfirmResetDialog({
  open,
  onOpenChange,
  onConfirm,
  title,
  description,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
  title: string;
  description: string;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogTitle className="text-sm font-semibold">{title}</AlertDialogTitle>
        <AlertDialogDescription className="mt-2 text-xs text-text-secondary">
          {description}
        </AlertDialogDescription>
        <div className="mt-4 flex justify-end gap-2">
          <AlertDialogCancel asChild>
            <Button size="sm" variant="ghost">
              Annuler
            </Button>
          </AlertDialogCancel>
          <AlertDialogAction asChild>
            <Button size="sm" onClick={onConfirm} className="bg-error text-surface-0">
              Réinitialiser
            </Button>
          </AlertDialogAction>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  );
}

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

function useParamsHandler(handle: WorkerHandle) {
  const { trainState, params, setParams, initModel, setLr, resetModel } = handle;
  const [pendingArch, setPendingArch] = useState<ModelParams | null>(null);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);

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
      if (lrChanged) setLr(next.lr);
    }
  }

  function confirmArchChange() {
    if (!pendingArch) return;
    setParams(pendingArch);
    initModel(pendingArch);
    setPendingArch(null);
  }

  function handleReset() {
    if (trainState === 'trained' || trainState === 'error') {
      setResetDialogOpen(true);
    } else {
      resetModel();
    }
  }

  function confirmReset() {
    setResetDialogOpen(false);
    resetModel();
  }

  return {
    pendingArch,
    setPendingArch,
    resetDialogOpen,
    setResetDialogOpen,
    handleParamsChange,
    confirmArchChange,
    handleReset,
    confirmReset,
  };
}

function ModelPanelInner({ colorVar, layout, handle }: ModelPanelInnerProps) {
  const { trainState, steps, words, errorMessage, params, initModel, train, generate } = handle;
  const ph = useParamsHandler(handle);
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      initModel(params);
    }
  }, [initModel, params]);

  const isHorizontal = layout === 'horizontal';
  const containerClass = isHorizontal
    ? 'flex flex-col gap-3 lg:flex-row lg:min-w-0'
    : 'flex flex-col gap-3';
  const glowClass = colorVar === 'a' ? 'panel-glow-a' : 'panel-glow-b';

  return (
    <div className={containerClass}>
      <ErrorBanner message={errorMessage} />
      <div className={isHorizontal ? 'lg:flex-[0_0_40%] lg:min-w-0' : ''}>
        <ParamsPanel
          params={params}
          onParamsChange={ph.handleParamsChange}
          onTrain={() => train(params.trainSteps)}
          onGenerate={() => generate(params.temperature, DEFAULT_N_SAMPLES)}
          onReset={ph.handleReset}
          trainState={trainState}
          colorVar={colorVar}
          glowClass={glowClass}
        />
      </div>
      <div className={isHorizontal ? 'lg:flex-[0_0_25%] lg:min-w-0' : ''}>
        <LossPanel steps={steps} colorVar={colorVar} glowClass={glowClass} />
      </div>
      <div className={isHorizontal ? 'lg:flex-[1_1_35%] lg:min-w-0' : ''}>
        <InferencePanel
          words={words}
          colorVar={colorVar}
          temperature={params.temperature}
          glowClass={glowClass}
        />
      </div>

      <ConfirmResetDialog
        open={ph.pendingArch !== null}
        onOpenChange={(open) => !open && ph.setPendingArch(null)}
        onConfirm={ph.confirmArchChange}
        title="Réinitialiser l'entraînement ?"
        description="Modifier les paramètres d'architecture réinitialisera la progression."
      />
      <ConfirmResetDialog
        open={ph.resetDialogOpen}
        onOpenChange={ph.setResetDialogOpen}
        onConfirm={ph.confirmReset}
        title="Réinitialiser le modèle ?"
        description="Le modèle sera recréé de zéro. La progression et les mots générés seront perdus."
      />
    </div>
  );
}

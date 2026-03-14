import { useEffect, useMemo, useRef, useState } from 'react';
import type { ColorVar, ModelParams } from '../../lib/types';
import { DEFAULT_N_SAMPLES, HEAD_OPTIONS } from '../../lib/constants';
import { isArchChange, validHeadCounts } from '../../lib/validation';
import { useModelWorker } from '../../hooks/use-model-worker';
import type { WorkerHandle } from '../../hooks/use-model-worker';
import { computeMemorization, computeQuality, computeDiversity } from '../../lib/training-metrics';
import { computeFeedback, type FeedbackResult } from '../../lib/training-feedback';
import { useLossData } from '../../hooks/use-loss-data';
import { ParamsPanel } from './params-panel';
import { LossPanel } from './loss-panel';
import { InferencePanel } from './inference-panel';
import { ErrorBanner } from './error-banner';
import { ConfirmResetDialog } from '../ui/confirm-reset-dialog';

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

  function handleParamsChange(incoming: ModelParams) {
    // Auto-correct n_head when n_embd changes and current head count is invalid
    const next = { ...incoming };
    if (incoming.n_embd !== params.n_embd) {
      const heads = validHeadCounts(incoming.n_embd, [...HEAD_OPTIONS]);
      if (!heads.includes(next.n_head)) {
        next.n_head = heads[heads.length - 1] ?? 1;
      }
    }

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

function useFeedback(handle: WorkerHandle, lastEma: number | null): FeedbackResult | null {
  const { words, steps, params, datasetProfile } = handle;

  return useMemo(() => {
    if (!datasetProfile || words.length === 0) return null;
    const memorization = computeMemorization(words, datasetProfile);
    const quality = computeQuality(words, datasetProfile);
    const diversity = computeDiversity(words);
    return computeFeedback(
      { memorization, quality, diversity },
      {
        temperature: params.temperature,
        lossEma: lastEma,
        totalSteps: steps.length,
        wordCount: words.length,
        datasetSize: datasetProfile.words.length,
        vocabSize: datasetProfile.vocabSize,
        modelConfig: {
          n_embd: params.n_embd,
          n_head: params.n_head,
          n_layer: params.n_layer,
          block_size: params.block_size,
        },
      },
    );
  }, [words, steps.length, params, datasetProfile, lastEma]);
}

function ModelPanelInner({ colorVar, layout, handle }: ModelPanelInnerProps) {
  const { trainState, steps, words, errorMessage, params, initModel, train, generate } = handle;
  const ph = useParamsHandler(handle);
  const initialized = useRef(false);
  const lossData = useLossData(steps, colorVar);
  const feedback = useFeedback(handle, lossData.lastEma);

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
        <LossPanel steps={steps} colorVar={colorVar} glowClass={glowClass} lossData={lossData} />
      </div>
      <div className={isHorizontal ? 'lg:flex-[1_1_35%] lg:min-w-0' : ''}>
        <InferencePanel
          words={words}
          colorVar={colorVar}
          temperature={params.temperature}
          glowClass={glowClass}
          feedback={feedback}
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

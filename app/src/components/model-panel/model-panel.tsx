import { useEffect, useState } from 'react';
import type { ModelParams } from '../../lib/types';
import { DEFAULT_PARAMS } from '../../lib/types';
import { useModelWorker } from '../../hooks/use-model-worker';
import { ParamsPanel } from './params-panel';
import { LossPanel } from './loss-panel';
import { InferencePanel } from './inference-panel';

export type ModelPanelProps = {
  colorVar: 'a' | 'b';
  layout: 'horizontal' | 'vertical';
  workerHandle?: ReturnType<typeof useModelWorker>;
};

export function ModelPanel({ colorVar, layout, workerHandle }: ModelPanelProps) {
  const ownWorker = useModelWorker();
  const { trainState, steps, words, initModel, train, setLr, generate } = workerHandle ?? ownWorker;

  const [params, setParams] = useState<ModelParams>({ ...DEFAULT_PARAMS });

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

    setParams(next);

    if (archChanged) {
      initModel(next);
    } else if (lrChanged) {
      setLr(next.lr);
    }
  }

  const isHorizontal = layout === 'horizontal';

  const containerStyle = isHorizontal
    ? { display: 'flex', flexDirection: 'row' as const, gap: '0.5rem', minWidth: 0 }
    : { display: 'flex', flexDirection: 'column' as const, gap: '0.5rem' };

  const paramsStyle = isHorizontal ? { flex: '0 0 40%', minWidth: 0 } : { flex: '0 0 35%' };
  const lossStyle = isHorizontal ? { flex: '0 0 25%', minWidth: 0 } : { flex: '0 0 25%' };
  const inferenceStyle = isHorizontal ? { flex: '1 1 35%', minWidth: 0 } : { flex: '1 1 40%' };

  return (
    <div style={containerStyle}>
      <div style={paramsStyle}>
        <ParamsPanel
          params={params}
          onParamsChange={handleParamsChange}
          onTrain={() => train(200)}
          onGenerate={() => generate(params.temperature, 10)}
          trainState={trainState}
          colorVar={colorVar}
        />
      </div>
      <div style={lossStyle}>
        <LossPanel steps={steps} colorVar={colorVar} />
      </div>
      <div style={inferenceStyle}>
        <InferencePanel words={words} colorVar={colorVar} />
      </div>
    </div>
  );
}

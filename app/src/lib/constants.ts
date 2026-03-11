import type { ModelParams } from './types';

export const DEFAULT_PARAMS: ModelParams = {
  datasetId: 'prenoms-simple',
  n_embd: 16,
  n_head: 4,
  n_layer: 1,
  block_size: 16,
  lr: 0.01,
  temperature: 0.8,
  trainSteps: 200,
};

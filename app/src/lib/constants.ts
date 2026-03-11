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

export const N_EMBD_OPTIONS = [8, 16, 32] as const;
export const N_LAYER_OPTIONS = [1, 2, 4] as const;
export const BLOCK_SIZE_OPTIONS = [8, 16, 32, 64] as const;
export const TRAIN_STEPS_OPTIONS = [100, 200, 500, 1000, 2000] as const;
export const HEAD_OPTIONS = [1, 2, 4, 8] as const;
export const DEFAULT_N_SAMPLES = 10;

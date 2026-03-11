export type Mode = 'solo' | 'compare';

export type ColorVar = 'a' | 'b';

export type ModelParams = {
  datasetId: string;
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
  lr: number;
  temperature: number;
  trainSteps: number;
};

export type StepResult = {
  step: number;
  loss: number;
  word: string;
  lr: number;
};

export type TrainState = 'idle' | 'training' | 'trained' | 'error';

export type WorkerMessage =
  | { type: 'init'; datasetText: string; config: ModelParams }
  | { type: 'train'; n_steps: number }
  | { type: 'set_lr'; lr: number }
  | { type: 'generate'; temperature: number; n_samples: number }
  | { type: 'dispose' };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'step'; data: StepResult }
  | { type: 'train_done' }
  | { type: 'generated'; words: string[] }
  | { type: 'error'; message: string };

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

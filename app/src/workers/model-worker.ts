import init, { WasmGpt } from '@wasm/microgpt_wasm';
import type { WorkerMessage, WorkerResponse } from '../lib/types';

let gpt: WasmGpt | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

async function handleInit(
  datasetText: string,
  config: { n_embd: number; n_head: number; n_layer: number; block_size: number },
) {
  await init();
  if (gpt) gpt.free();
  gpt = WasmGpt.new_with_config(
    datasetText,
    config.n_embd,
    config.n_head,
    config.n_layer,
    config.block_size,
  );
  post({ type: 'ready' });
}

function requireGpt(): WasmGpt | null {
  if (!gpt) {
    post({ type: 'error', message: 'Model not initialized' });
    return null;
  }
  return gpt;
}

function validateField(value: number, name: string): boolean {
  if (!isPositiveFinite(value)) {
    post({ type: 'error', message: `Invalid ${name}: ${value}` });
    return false;
  }
  return true;
}

let trainRemaining = 0;

function handleTrain(n_steps: number) {
  const g = requireGpt();
  if (!g) return;
  trainRemaining = n_steps;
  trainChunk();
}

function trainChunk() {
  if (!gpt || trainRemaining <= 0) {
    post({ type: 'train_done' });
    return;
  }
  const chunkSize = Math.min(10, trainRemaining);
  for (let i = 0; i < chunkSize; i++) {
    // train_step returns { step, loss, word, lr } as any from WASM
    const result = gpt.train_step() as { step: number; loss: number; word: string; lr: number };
    post({ type: 'step', data: result });
    trainRemaining--;
  }
  if (trainRemaining > 0) {
    setTimeout(trainChunk, 0);
  } else {
    post({ type: 'train_done' });
  }
}

function handleGenerate(temperature: number, n_samples: number) {
  const g = requireGpt();
  if (!g) return;
  const vocab = JSON.parse(g.vocab_tokens()) as string[];
  const bos = g.bos();
  const words: string[] = [];

  for (let s = 0; s < n_samples; s++) {
    let prefix = new Uint32Array([bos]);
    let word = '';
    for (let t = 0; t < 20; t++) {
      const probs = g.compute_probs(prefix, temperature);
      const token = sampleFromProbs(probs);
      if (token === bos) break;
      word += vocab[token];
      const newPrefix = new Uint32Array(prefix.length + 1);
      newPrefix.set(prefix);
      newPrefix[prefix.length] = token;
      prefix = newPrefix;
    }
    if (word.length > 0) words.push(word);
  }
  post({ type: 'generated', words });
}

function sampleFromProbs(probs: Float64Array): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

function isPositiveFinite(v: number): boolean {
  return Number.isFinite(v) && v > 0;
}

function validateConfig(c: {
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
}): string | null {
  if (
    !isPositiveFinite(c.n_embd) ||
    !isPositiveFinite(c.n_head) ||
    !isPositiveFinite(c.n_layer) ||
    !isPositiveFinite(c.block_size)
  ) {
    return `Invalid config: n_embd=${c.n_embd} n_head=${c.n_head} n_layer=${c.n_layer} block_size=${c.block_size}`;
  }
  if (c.n_embd % c.n_head !== 0) {
    return `n_embd (${c.n_embd}) must be divisible by n_head (${c.n_head})`;
  }
  return null;
}

async function dispatch(msg: WorkerMessage) {
  switch (msg.type) {
    case 'init': {
      const err = validateConfig(msg.config);
      if (err) {
        post({ type: 'error', message: err });
        return;
      }
      await handleInit(msg.datasetText, msg.config);
      return;
    }
    case 'train':
      if (!validateField(msg.n_steps, 'n_steps')) return;
      handleTrain(msg.n_steps);
      return;
    case 'set_lr':
      if (!validateField(msg.lr, 'lr')) return;
      if (gpt) gpt.set_lr(msg.lr);
      return;
    case 'generate':
      if (!validateField(msg.temperature, 'temperature')) return;
      handleGenerate(msg.temperature, msg.n_samples);
      return;
    case 'dispose':
      if (gpt) {
        gpt.free();
        gpt = null;
      }
      return;
  }
}

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  try {
    await dispatch(e.data);
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

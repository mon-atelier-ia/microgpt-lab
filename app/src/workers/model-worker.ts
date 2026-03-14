import init, { WasmGpt } from '@wasm/microgpt_wasm';
import type { WorkerMessage, WorkerResponse } from '../lib/types';
import {
  isPositiveFinite,
  isStepResult,
  sampleFromProbs,
  validateConfig,
} from '../lib/worker-utils';

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

function handleTrain(n_steps: number) {
  const g = requireGpt();
  if (!g) return;
  for (let i = 0; i < n_steps; i++) {
    const raw = g.train_step();
    if (!isStepResult(raw)) {
      post({ type: 'error', message: 'Unexpected train_step result' });
      return;
    }
    post({ type: 'step', data: raw });
  }
  post({ type: 'train_done' });
}

function handleGenerate(temperature: number, n_samples: number) {
  const g = requireGpt();
  if (!g) return;
  const rawVocab: unknown = JSON.parse(g.vocab_tokens());
  if (!Array.isArray(rawVocab) || rawVocab.some((t) => typeof t !== 'string')) {
    post({ type: 'error', message: 'Invalid vocab format from WASM' });
    return;
  }
  const vocab = rawVocab as string[];
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
    // After a WASM crash, the instance is corrupted — force re-init
    gpt = null;
    const raw = err instanceof Error ? err.message : String(err);
    const message =
      raw === 'unreachable' || raw.includes('RuntimeError')
        ? 'Le modèle WASM a planté (mémoire insuffisante ou panic interne). Réinitialisez le modèle.'
        : raw;
    post({ type: 'error', message });
  }
};

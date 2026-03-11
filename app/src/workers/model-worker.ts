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

let trainRemaining = 0;

function handleTrain(n_steps: number) {
  if (!gpt) {
    post({ type: 'error', message: 'Model not initialized' });
    return;
  }
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
  if (!gpt) {
    post({ type: 'error', message: 'Model not initialized' });
    return;
  }
  const vocab = JSON.parse(gpt.vocab_tokens()) as string[];
  const bos = gpt.bos();
  const words: string[] = [];

  for (let s = 0; s < n_samples; s++) {
    let prefix = new Uint32Array([bos]);
    let word = '';
    for (let t = 0; t < 20; t++) {
      const probs = gpt.compute_probs(prefix, temperature);
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

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  try {
    switch (e.data.type) {
      case 'init':
        await handleInit(e.data.datasetText, e.data.config);
        break;
      case 'train':
        handleTrain(e.data.n_steps);
        break;
      case 'set_lr':
        if (gpt) gpt.set_lr(e.data.lr);
        break;
      case 'generate':
        handleGenerate(e.data.temperature, e.data.n_samples);
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

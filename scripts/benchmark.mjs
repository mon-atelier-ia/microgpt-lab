/**
 * Empirical benchmark: loss EMA + match ratio at checkpoints for each dataset.
 * Runs WASM directly in Node.js — no browser needed.
 *
 * Usage: node scripts/benchmark.mjs
 */
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(__dirname, '../app/wasm-pkg/microgpt_wasm_bg.wasm');

// Dynamic import of wasm-bindgen glue
const wasm = await import('../app/wasm-pkg/microgpt_wasm.js');
const wasmBytes = readFileSync(wasmPath);
await wasm.default(wasmBytes);

// Dataset loaders — import the raw TS modules via a simpler approach: read the source and extract the array
// Actually, the data files export arrays. Let's import them via dynamic import with a tsx loader...
// Simpler: just hardcode small inline datasets or read the .ts files and eval.
// Even simpler: use the wasm-pkg which has all we need, and load datasets from the TS source files.

// We'll use a different approach: load the dataset text files directly
// The datasets are TypeScript arrays. Let's extract them via regex from the source files.

function loadDataset(filename, exportName) {
  const src = readFileSync(resolve(__dirname, `../app/src/data/${filename}`), 'utf-8');
  // Match: export const foo = ['a', 'b', ...] or ["a", "b", ...]
  const match = src.match(/export\s+const\s+\w+(?::\s*\w+\[\])?\s*=\s*\[([\s\S]*)\]/);
  if (!match) throw new Error(`Cannot parse dataset ${filename}`);
  // Extract quoted strings
  const items = [];
  const re = /['"]([^'"]+)['"]/g;
  let m;
  while ((m = re.exec(match[1])) !== null) items.push(m[1]);
  return items;
}

const datasets = [
  { id: 'prenoms-simple', name: 'Prénoms FR (50)', file: 'prenoms-simple.ts', export: 'prenomsSimple' },
  { id: 'prenoms', name: 'Prénoms FR (1000)', file: 'prenoms.ts', export: 'prenoms' },
  { id: 'baby-names', name: 'Baby Names EN (1000)', file: 'baby-names.ts', export: 'babyNames' },
  { id: 'dinosaures', name: 'Dinosaures (1530)', file: 'dinosaures.ts', export: 'dinosaures' },
  { id: 'pokemon-fr', name: 'Pokémon FR (1022)', file: 'pokemon-fr.ts', export: 'pokemonFr' },
  { id: 'names-en', name: 'Names EN (8000)', file: 'names-en.ts', export: 'namesEn' },
  { id: 'prenoms-insee', name: 'Prénoms FR (33k)', file: 'prenoms-insee.ts', export: 'prenomsInsee' },
];

const checkpoints = [200, 500, 1000, 2000, 3000, 5000];

function sampleFromProbs(probs) {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

function generateWords(gpt, vocab, bos, count, temperature, blockSize = 16) {
  const words = [];
  for (let s = 0; s < count; s++) {
    let prefix = new Uint32Array([bos]);
    let word = '';
    for (let t = 0; t < blockSize - 1; t++) {
      const probs = gpt.compute_probs(prefix, temperature);
      const token = sampleFromProbs(probs);
      if (token === bos) break;
      word += vocab[token];
      // Keep prefix within block_size to avoid index out of bounds
      if (prefix.length >= blockSize) {
        prefix = prefix.slice(prefix.length - blockSize + 1);
      }
      const np = new Uint32Array(prefix.length + 1);
      np.set(prefix);
      np[prefix.length] = token;
      prefix = np;
    }
    if (word.length > 0) words.push(word);
  }
  return words;
}

const TEMPS = [0.5, 0.8, 1.2];
const N_SAMPLES = 50;
const N_RUNS = 3;
const MAX_STEPS = 5000;

// Compute random baseline match rate for a dataset
function randomBaseline(vocab, bos, datasetSet, blockSize) {
  // Generate 200 random words (no model) to estimate accidental match rate
  const words = [];
  const chars = vocab.filter((_, i) => i !== bos);
  for (let s = 0; s < 200; s++) {
    let word = '';
    const len = 1 + Math.floor(Math.random() * (blockSize - 1));
    for (let t = 0; t < len; t++) {
      word += chars[Math.floor(Math.random() * chars.length)];
    }
    if (word.length > 0) words.push(word);
  }
  const matches = words.filter(w => datasetSet.has(w.toLowerCase())).length;
  return (matches / words.length * 100).toFixed(1);
}

console.log('# Benchmark — Loss EMA + Match Ratio per Dataset\n');
console.log(`Config: n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01`);
console.log(`Method: ${MAX_STEPS} steps, ${N_SAMPLES} samples × ${N_RUNS} runs per checkpoint, temperatures ${TEMPS.join('/')}\n`);

const allResults = [];

for (const ds of datasets) {
  const datasetWords = loadDataset(ds.file, ds.export);
  const text = datasetWords.join('\n');
  const datasetSet = new Set(datasetWords.map(w => w.toLowerCase()));

  const gpt = wasm.WasmGpt.new_with_config(text, 16, 4, 1, 16);
  const vocab = JSON.parse(gpt.vocab_tokens());
  const bos = gpt.bos();

  const baseline = randomBaseline(vocab, bos, datasetSet, 16);
  console.log(`## ${ds.name} (${datasetWords.length} words, random baseline: ${baseline}%)\n`);
  console.log('| Step | Loss EMA | t=0.5 match% (avg) | t=0.8 match% (avg) | t=1.2 match% (avg) | Samples (t=0.8) |');
  console.log('|------|----------|--------------------|--------------------|--------------------|--------------------|');

  let ema = null;
  const alpha = 0.1;
  let nextCheckpoint = 0;
  const dsResults = [];

  for (let step = 1; step <= MAX_STEPS; step++) {
    const raw = gpt.train_step();
    ema = ema === null ? raw.loss : alpha * raw.loss + (1 - alpha) * ema;

    if (step === checkpoints[nextCheckpoint]) {
      const tempResults = {};
      let sampleWords = [];

      for (const temp of TEMPS) {
        let totalMatches = 0;
        let totalGenerated = 0;

        for (let run = 0; run < N_RUNS; run++) {
          const generated = generateWords(gpt, vocab, bos, N_SAMPLES, temp, 16);
          const matches = generated.filter(w => datasetSet.has(w.toLowerCase())).length;
          totalMatches += matches;
          totalGenerated += generated.length;
          if (temp === 0.8 && run === 0) sampleWords = generated.slice(0, 5);
        }

        const avgRatio = totalGenerated > 0 ? Math.round(totalMatches / totalGenerated * 100) : 0;
        tempResults[temp] = { ratio: avgRatio, total: totalGenerated, matches: totalMatches };
      }

      const row = {
        step,
        ema: ema.toFixed(4),
        tempResults,
        samples: sampleWords,
      };
      dsResults.push(row);

      const t05 = tempResults[0.5];
      const t08 = tempResults[0.8];
      const t12 = tempResults[1.2];
      console.log(`| ${step} | ${row.ema} | ${t05.ratio}% (${t05.matches}/${t05.total}) | ${t08.ratio}% (${t08.matches}/${t08.total}) | ${t12.ratio}% (${t12.matches}/${t12.total}) | ${row.samples.join(', ')} |`);

      nextCheckpoint++;
      if (nextCheckpoint >= checkpoints.length) break;
    }
  }

  gpt.free();
  allResults.push({ dataset: ds.name, size: datasetWords.length, baseline, results: dsResults });
  console.log('');
}

console.log('---\nDone.');

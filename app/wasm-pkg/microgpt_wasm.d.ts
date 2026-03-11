/* tslint:disable */
/* eslint-disable */

export class WasmGpt {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * BOS token id.
     */
    bos(): number;
    /**
     * Forward pass on prefix, return probs as Float64Array.
     * Direct plug-in for useInferenceEngine's ComputeProbs type.
     */
    compute_probs(prefix_ids: Uint32Array, temperature: number): Float64Array;
    /**
     * Model config as a JS object: { n_embd, n_head, n_layer, block_size, head_dim, vocab_size }.
     */
    config(): any;
    /**
     * Return the current learning rate.
     */
    current_lr(): number;
    /**
     * Full forward trace at query_pos for head_idx.
     * Returns a JS object with all intermediate vectors for Ch4 animation.
     */
    forward_trace(token_ids: Uint32Array, query_pos: number, head_idx: number): any;
    /**
     * Forward + backward at `selected_pos`, returning per-position
     * embedding gradients via Rust autodiff.
     *
     * Uses scalar engine (value.rs) — weights are synced from tensor model.
     */
    forward_with_grads(token_ids: Uint32Array, selected_pos: number): any;
    /**
     * Get lm_head row for token_id. Used by Ch5 for gradient display.
     */
    lm_head_row(token_id: number): Float64Array;
    /**
     * Create model from newline-separated (or comma-separated) names.
     */
    constructor(names_text: string);
    /**
     * Create model with custom hyperparameters.
     * Returns an error if `n_embd` is not divisible by `n_head`.
     */
    static new_with_config(names_text: string, n_embd: number, n_head: number, n_layer: number, block_size: number): WasmGpt;
    /**
     * Reset model with new dataset.
     */
    reset(names_text: string): void;
    /**
     * Reset model weights for streaming re-training.
     */
    reset_training(): void;
    /**
     * Set the learning rate for subsequent training steps.
     */
    set_lr(lr: number): void;
    /**
     * Batch-train the model for `n_steps` without capturing trace data.
     * Uses tensor engine (~100× faster than scalar).
     * After this call, the model is trained and ready for inference/forward_trace.
     */
    train(n_steps: number): void;
    /**
     * Run one training step using tensor engine.
     * Returns JS object: { step, loss, word, lr }.
     */
    train_step(): any;
    /**
     * Run one training step with full gradient + parameter capture.
     * Uses tensor engine for forward/backward/Adam.
     */
    train_step_traced(): any;
    /**
     * Return training metadata: optimizer config + tracked parameter list
     * + initial step (step 0, pre-training state).
     */
    training_meta(): any;
    /**
     * Character list (for display). Returns JSON string array.
     */
    vocab_tokens(): string;
    /**
     * Get wpe row for pos_id (reads from tensor model).
     */
    wpe_row(pos_id: number): Float64Array;
    /**
     * Get wte row for token_id (reads from tensor model).
     */
    wte_row(token_id: number): Float64Array;
}

/**
 * Install panic hook once so Rust panics show readable stack traces
 * in the browser console instead of "unreachable".
 */
export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgpt_free: (a: number, b: number) => void;
    readonly wasmgpt_bos: (a: number) => number;
    readonly wasmgpt_compute_probs: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmgpt_config: (a: number) => any;
    readonly wasmgpt_current_lr: (a: number) => number;
    readonly wasmgpt_forward_trace: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmgpt_forward_with_grads: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmgpt_lm_head_row: (a: number, b: number) => [number, number, number, number];
    readonly wasmgpt_new: (a: number, b: number) => [number, number, number];
    readonly wasmgpt_new_with_config: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly wasmgpt_reset: (a: number, b: number, c: number) => [number, number];
    readonly wasmgpt_reset_training: (a: number) => void;
    readonly wasmgpt_set_lr: (a: number, b: number) => void;
    readonly wasmgpt_train: (a: number, b: number) => void;
    readonly wasmgpt_train_step: (a: number) => [number, number, number];
    readonly wasmgpt_train_step_traced: (a: number) => [number, number, number];
    readonly wasmgpt_training_meta: (a: number) => [number, number, number];
    readonly wasmgpt_vocab_tokens: (a: number) => [number, number];
    readonly wasmgpt_wpe_row: (a: number, b: number) => [number, number, number, number];
    readonly wasmgpt_wte_row: (a: number, b: number) => [number, number, number, number];
    readonly init_panic_hook: () => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

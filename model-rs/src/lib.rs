//! Minimal GPT library in pure Rust — faithful port of Karpathy's microgpt.py
//! with scalar autograd, zero dependencies.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod data;
pub mod inference;
pub mod model;
pub mod rng;
pub mod train;
pub mod value;

/// Forward pass through the transformer.
pub mod forward;
/// Core tensor operations (linear, softmax, rmsnorm).
pub mod ops;
/// Tensor autograd engine — matrix-level autodiff (~25 nodes vs ~6000).
pub mod tensor;
/// Tensor-level forward pass (replaces `forward.rs` for training/inference).
pub mod tensor_forward;
/// Tensor-level model: weights + Adam state.
pub mod tensor_model;
/// Tensor-level training step.
pub mod tensor_train;

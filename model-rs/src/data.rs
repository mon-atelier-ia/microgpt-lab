//! Character-level vocabulary and tokenization.

use std::collections::{BTreeSet, HashMap};

/// Character-level vocabulary with string-to-id mapping.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Ordered list of tokens (chars + BOS).
    pub tokens: Vec<String>,
    /// String → token id mapping.
    pub stoi: HashMap<String, usize>,
    bos: usize,
}

impl Vocab {
    /// BOS token id — also used as EOS (matches Karpathy's gist).
    pub fn bos(&self) -> usize {
        self.bos
    }
    /// Total vocabulary size (unique chars + BOS).
    pub fn size(&self) -> usize {
        self.tokens.len()
    }
}

/// Build character-level vocabulary from documents.
/// BOS is the last token (id = number of unique chars).
pub fn build_vocab(docs: &[&str]) -> Vocab {
    let mut chars = BTreeSet::new();
    for d in docs {
        for c in d.chars() {
            chars.insert(c);
        }
    }
    let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    let bos = tokens.len();
    tokens.push("<BOS>".to_string());
    let stoi: HashMap<String, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    Vocab { tokens, stoi, bos }
}

/// Tokenize a document: [BOS, chars..., BOS] truncated to block_size.
pub fn tokenize(doc: &str, vocab: &Vocab, block_size: usize) -> Vec<usize> {
    let bos = vocab.bos();
    std::iter::once(bos)
        .chain(doc.chars().map(|c| vocab.stoi[&c.to_string()]))
        .chain(std::iter::once(bos))
        .take(block_size + 1) // +1 because we need block_size input-target pairs
        .collect()
}

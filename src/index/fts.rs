//! Full-Text Search (FTS) index implementation
//!
//! This module provides full-text search capabilities using inverted indexing
//! for fast content search across dictionary entries.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::index::{FtsConfig, Index, IndexConfig, IndexStats};
use crate::traits::{DictError, Result};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Document ID for FTS operations
type DocId = u32;

/// Term ID for token indexing
type TermId = u32;

/// Token with position information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Token {
    /// Token text
    text: String,
    /// Term ID
    term_id: TermId,
    /// Position in document
    position: u32,
    /// Document frequency
    doc_freq: u32,
}

/// Search result with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsSearchResult {
    /// Document ID
    pub doc_id: DocId,
    /// Document key (word)
    pub key: String,
    /// Relevance score
    pub score: f32,
    /// Snippet highlighting positions
    pub highlights: Vec<(usize, usize)>,
}

/// Inverted index entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InvertedIndexEntry {
    /// Term ID
    term_id: TermId,
    /// Term text
    term: String,
    /// Documents containing this term
    postings: Vec<Posting>,
    /// Total document frequency
    doc_freq: u32,
    /// Total term frequency
    term_freq: u32,
}

/// Posting list entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    /// Document ID
    doc_id: DocId,
    /// Term frequency in document
    term_freq: u32,
    /// Positions where term occurs
    positions: Vec<u32>,
}

/// Document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    /// Document ID
    doc_id: DocId,
    /// Document key (word)
    key: String,
    /// Document content
    content: Vec<u8>,
    /// Document length in tokens
    doc_length: u32,
}

/// FTS Index implementation using inverted indexing
pub struct FtsIndex {
    /// Inverted index: term -> posting list
    inverted_index: HashMap<String, InvertedIndexEntry>,
    /// Forward index: doc_id -> document
    documents: HashMap<DocId, Document>,
    /// Next available term ID
    next_term_id: TermId,
    /// Next available document ID
    next_doc_id: DocId,
    /// Stop words set
    stop_words: HashSet<String>,
    /// Term statistics
    term_stats: HashMap<String, u32>,
    /// Index configuration
    config: FtsConfig,
    /// Index statistics
    stats: IndexStats,
    /// Thread-safe access
    lock: Arc<RwLock<()>>,
}

impl FtsIndex {
    /// Create a new FTS index
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            documents: HashMap::new(),
            next_term_id: 1,
            next_doc_id: 1,
            stop_words: HashSet::new(),
            term_stats: HashMap::new(),
            config: FtsConfig::default(),
            stats: IndexStats {
                entries: 0,
                size: 0,
                build_time: 0,
                version: "1.0".to_string(),
                config: IndexConfig::default(),
            },
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Create FTS index with custom configuration
    pub fn with_config(config: FtsConfig) -> Self {
        let stop_words = HashSet::from_iter(config.stop_words.clone());

        Self {
            inverted_index: HashMap::new(),
            documents: HashMap::new(),
            next_term_id: 1,
            next_doc_id: 1,
            stop_words,
            term_stats: HashMap::new(),
            config,
            stats: IndexStats {
                entries: 0,
                size: 0,
                build_time: 0,
                version: "1.0".to_string(),
                config: IndexConfig::default(),
            },
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Tokenize text into terms
    fn tokenize(&self, text: &str) -> Vec<(String, u32)> {
        // Create regex for word extraction
        let word_pattern = Regex::new(r"\p{L}+|\p{N}+").expect("Failed to create word regex");

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut position = 0u32;

        for mat in word_pattern.find_iter(text) {
            let token = mat.as_str().to_lowercase();

            // Skip stop words
            if self.stop_words.contains(&token) {
                continue;
            }

            // Filter by length
            if token.len() >= self.config.min_token_len && token.len() <= self.config.max_token_len
            {
                tokens.push(token);
                positions.push(position);
                position += 1;
            }
        }

        tokens.into_iter().zip(positions).collect()
    }

    /// Add a document to the index
    fn add_document(&mut self, key: String, content: &[u8]) -> Result<DocId> {
        let _guard = self.lock.write();

        let content_str = String::from_utf8_lossy(content);
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Tokenize the content
        let tokens = self.tokenize(&content_str);

        // Track term frequencies
        let mut term_freqs = HashMap::new();
        for (term, position) in &tokens {
            *term_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        // Create document
        let document = Document {
            doc_id,
            key: key.clone(),
            content: content.to_vec(),
            doc_length: tokens.len() as u32,
        };

        self.documents.insert(doc_id, document);

        // Update inverted index
        for (term, term_freq) in term_freqs {
            let positions: Vec<u32> = tokens
                .iter()
                .filter_map(|(t, pos)| if *t == term { Some(*pos) } else { None })
                .collect();

            // Assign a stable term_id per term. If the term already exists, reuse its id.
            let entry = self.inverted_index.entry(term.clone()).or_insert_with(|| {
                let term_id = self.next_term_id;
                self.next_term_id = self.next_term_id.saturating_add(1);
                InvertedIndexEntry {
                    term_id,
                    term: term.clone(),
                    postings: Vec::new(),
                    doc_freq: 0,
                    term_freq: 0,
                }
            });

            entry.postings.push(Posting {
                doc_id,
                term_freq,
                positions,
            });
            entry.doc_freq = entry.doc_freq.saturating_add(1);
            entry.term_freq = entry.term_freq.saturating_add(term_freq);
        }

        Ok(doc_id)
    }

    /// Search for documents containing the query
    pub fn search(&self, query: &str) -> Result<Vec<(String, f32)>> {
        let _guard = self.lock.read();

        let query_tokens = self.tokenize(query);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate IDF for query terms
        let total_docs = self.documents.len() as f32;
        let mut query_term_weights = HashMap::new();

        for (term, _) in &query_tokens {
            if let Some(entry) = self.inverted_index.get(term) {
                let idf = (total_docs / (entry.doc_freq as f32 + 1.0)).ln() + 1.0;
                query_term_weights.insert(term, idf);
            }
        }

        // Score documents
        let mut doc_scores = HashMap::<DocId, f32>::new();

        for (term, idf) in query_term_weights {
            if let Some(entry) = self.inverted_index.get(term.as_str()) {
                for posting in &entry.postings {
                    let tf = 1.0 + (posting.term_freq as f32).ln(); // Log TF
                    let score = idf * tf;
                    *doc_scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }

        // Sort results by score
        let mut results: Vec<_> = doc_scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.documents
                    .get(&doc_id)
                    .map(|doc| (doc.key.clone(), score))
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Get terms starting with prefix
    pub fn prefix_search(&self, prefix: &str) -> Result<Vec<String>> {
        let _guard = self.lock.read();

        let mut terms = Vec::new();
        for term in self.inverted_index.keys() {
            if term.starts_with(prefix) {
                terms.push(term.clone());
            }
        }
        terms.sort();

        Ok(terms)
    }

    /// Get term frequency
    pub fn term_frequency(&self, term: &str) -> u32 {
        self.inverted_index
            .get(term)
            .map(|e| e.term_freq)
            .unwrap_or(0)
    }

    /// Get document frequency
    pub fn document_frequency(&self, term: &str) -> u32 {
        self.inverted_index
            .get(term)
            .map(|e| e.doc_freq)
            .unwrap_or(0)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.inverted_index.len()
    }

    /// Get average document length
    pub fn avg_doc_length(&self) -> f32 {
        if self.documents.is_empty() {
            return 0.0;
        }
        let total_length: u32 = self.documents.values().map(|doc| doc.doc_length).sum();
        total_length as f32 / self.documents.len() as f32
    }

    /// Get suggestions for misspelled query
    pub fn suggest_spelling(&self, query: &str) -> Result<Vec<String>> {
        let _guard = self.lock.read();

        // Simple edit distance suggestion
        let mut suggestions = Vec::new();

        for term in self.inverted_index.keys() {
            let distance = self.edit_distance(query, term);
            if distance <= 2 {
                suggestions.push((term.clone(), distance));
            }
        }

        suggestions.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(suggestions
            .into_iter()
            .take(10)
            .map(|(term, _)| term)
            .collect())
    }

    /// Calculate edit distance between two strings
    fn edit_distance(&self, s1: &str, s2: &str) -> u32 {
        let (m, n) = (s1.len(), s2.len());
        let mut dp = vec![vec![0u32; n + 1]; m + 1];

        for i in 0..=m {
            dp[i][0] = i as u32;
        }
        for j in 0..=n {
            dp[0][j] = j as u32;
        }

        for i in 1..=m {
            for j in 1..=n {
                if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Get highlighted snippet for a document
    pub fn get_snippet(&self, doc_id: DocId, query: &str, max_length: usize) -> Option<String> {
        let _guard = self.lock.read();

        let doc = self.documents.get(&doc_id)?;
        let content = String::from_utf8_lossy(&doc.content);
        let query_tokens: Vec<_> = self.tokenize(query).into_iter().map(|(t, _)| t).collect();

        // Find first occurrence
        let mut first_pos = content.len();
        for token in &query_tokens {
            if let Some(pos) = content.to_lowercase().find(token) {
                first_pos = first_pos.min(pos);
            }
        }

        if first_pos == content.len() {
            return None;
        }

        // Generate snippet
        let start = first_pos.saturating_sub(max_length / 2);
        let end = (first_pos + max_length).min(content.len());
        let snippet = content[start..end].to_string();

        Some(snippet)
    }

    /// Validate index integrity
    pub fn validate(&self) -> Result<bool> {
        let _guard = self.lock.read();

        // Check consistency between forward and inverted indexes
        for (doc_id, doc) in &self.documents {
            // Check that document exists in inverted index
            let content = String::from_utf8_lossy(&doc.content);
            let tokens = self.tokenize(&content);

            for (term, _) in tokens {
                if let Some(entry) = self.inverted_index.get(&term) {
                    let has_doc = entry.postings.iter().any(|p| p.doc_id == *doc_id);
                    if !has_doc {
                        return Ok(false);
                    }
                }
            }
        }

        // Check posting list consistency
        for entry in self.inverted_index.values() {
            // Check document frequency
            let actual_doc_freq = entry.postings.len() as u32;
            if actual_doc_freq != entry.doc_freq {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> &IndexStats {
        &self.stats
    }
}

impl Index for FtsIndex {
    const INDEX_TYPE: &'static str = "fts";

    fn build(&mut self, entries: &[(String, Vec<u8>)], config: &IndexConfig) -> Result<()> {
        let start_time = Instant::now();

        // Clear existing data
        self.inverted_index.clear();
        self.documents.clear();
        self.next_term_id = 1;
        self.next_doc_id = 1;
        self.term_stats.clear();

        // Set configuration
        self.config = config.fts_config.clone();

        // Optional resource guard: if max_memory is configured, use it as a soft bound.
        // We don't change the API: on likely overflow risk we return an InsufficientMemory-style error.
        if let Some(max_mem) = config.max_memory {
            // Very rough heuristic: assume average (key+content) contributes.
            // This keeps behavior backwards compatible while protecting against obvious DoS inputs.
            let estimated: u64 = entries
                .iter()
                .map(|(k, v)| (k.len() + v.len()) as u64)
                .sum();
            if estimated > max_mem.saturating_mul(4) {
                return Err(DictError::IndexError(
                    "FTS index build aborted: estimated input too large for configured max_memory"
                        .to_string(),
                ));
            }
        }

        // Build index from entries
        for (key, content) in entries {
            self.add_document(key.clone(), content)?;
        }

        self.stats.entries = entries.len() as u64;
        self.stats.build_time = start_time.elapsed().as_millis() as u64;
        self.stats.size = self.inverted_index.len() as u64 * 100; // Estimated size

        // Validate the index
        if !self.validate()? {
            self.clear();
            return Err(DictError::IndexError(
                "FTS index validation failed; index discarded".to_string(),
            ));
        }

        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).map_err(|e| DictError::IoError(e.to_string()))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| DictError::MmapError(e.to_string()))?
        };

        // Deserialize the index
        let serialized_data = &mmap[..];
        let (inverted_index, documents, term_stats): (
            HashMap<String, InvertedIndexEntry>,
            HashMap<DocId, Document>,
            HashMap<String, u32>,
        ) = bincode::deserialize(serialized_data)
            .map_err(|e| DictError::SerializationError(e.to_string()))?;

        self.inverted_index = inverted_index;
        self.documents = documents;
        self.term_stats = term_stats;
        self.stats.size = std::fs::metadata(path)?.len();

        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|e| DictError::IoError(e.to_string()))?;

        // Serialize the FTS index
        let data = (&self.inverted_index, &self.documents, &self.term_stats);

        let serialized =
            bincode::serialize(&data).map_err(|e| DictError::SerializationError(e.to_string()))?;

        let mut file = file;
        file.write_all(&serialized)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        Ok(())
    }

    fn stats(&self) -> &IndexStats {
        &self.stats
    }

    fn is_built(&self) -> bool {
        !self.inverted_index.is_empty() && !self.documents.is_empty()
    }

    fn clear(&mut self) {
        self.inverted_index.clear();
        self.documents.clear();
        self.term_stats.clear();
        self.next_term_id = 1;
        self.next_doc_id = 1;
        self.stats = IndexStats {
            entries: 0,
            size: 0,
            build_time: 0,
            version: "1.0".to_string(),
            config: IndexConfig::default(),
        };
    }

    fn verify(&self) -> Result<bool> {
        self.validate()
    }
}

impl Default for FtsIndex {
    fn default() -> Self {
        Self::new()
    }
}

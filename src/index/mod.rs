//! High-performance indexing system for dictionary operations
//!
//! This module provides B-TREE and Full-Text Search (FTS) indexing capabilities
//! to accelerate dictionary lookups and search operations.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

use crate::traits::{DictError, Result};
use serde::{Deserialize, Serialize};

pub mod btree;
pub mod fts;

/// Common index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of entries indexed
    pub entries: u64,
    /// Index file size in bytes
    pub size: u64,
    /// Index build time in milliseconds
    pub build_time: u64,
    /// Index version
    pub version: String,
    /// Index configuration
    pub config: IndexConfig,
}

/// Configuration for index operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// B-TREE order (branching factor)
    pub btree_order: Option<usize>,
    /// FTS analyzer settings
    pub fts_config: FtsConfig,
    /// Compression settings
    pub compression: Option<CompressionConfig>,
    /// Whether to build index in memory first
    pub build_in_memory: bool,
    /// Maximum memory usage during build (bytes)
    pub max_memory: Option<u64>,
}

/// Full-Text Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsConfig {
    /// Minimum token length for indexing
    pub min_token_len: usize,
    /// Maximum token length for indexing
    pub max_token_len: usize,
    /// Whether to use stemming
    pub use_stemming: bool,
    /// Stop words to ignore during indexing
    pub stop_words: Vec<String>,
    /// Analyzer language
    pub language: Option<String>,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9 for gzip, 1-19 for zstd)
    pub level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// GZIP compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstandard compression
    Zstd,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        CompressionAlgorithm::Zstd
    }
}

/// Trait that defines common index operations
pub trait Index: Send + Sync {
    /// Index type identifier
    const INDEX_TYPE: &'static str;

    /// Build the index from entries
    fn build(&mut self, entries: &[(String, Vec<u8>)], config: &IndexConfig) -> Result<()>;

    /// Load index from file
    fn load(&mut self, path: &Path) -> Result<()>;

    /// Save index to file
    fn save(&self, path: &Path) -> Result<()>;

    /// Get index statistics
    fn stats(&self) -> &IndexStats;

    /// Check if index is built
    fn is_built(&self) -> bool;

    /// Clear the index
    fn clear(&mut self);

    /// Verify index integrity
    fn verify(&self) -> Result<bool>;
}

/// Error types specific to index operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexError {
    /// Index corruption detected
    CorruptedIndex(String),
    /// Index version mismatch
    VersionMismatch { expected: String, found: String },
    /// Index not built
    NotBuilt(String),
    /// Index I/O error
    IoError(String),
    /// Index configuration error
    ConfigError(String),
    /// Index too large for memory
    InsufficientMemory(String),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::CorruptedIndex(msg) => write!(f, "Corrupted index: {}", msg),
            IndexError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            IndexError::NotBuilt(name) => write!(f, "Index '{}' not built", name),
            IndexError::IoError(msg) => write!(f, "Index I/O error: {}", msg),
            IndexError::ConfigError(msg) => write!(f, "Index configuration error: {}", msg),
            IndexError::InsufficientMemory(msg) => write!(f, "Insufficient memory: {}", msg),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<IndexError> for DictError {
    fn from(err: IndexError) -> Self {
        DictError::IndexError(err.to_string())
    }
}

/// Manager for multiple indexes
pub struct IndexManager {
    /// B-TREE index
    btree: Option<btree::BTreeIndex>,
    /// FTS index
    fts: Option<fts::FtsIndex>,
    /// Index configuration
    config: IndexConfig,
    /// Paths to index files
    paths: HashMap<&'static str, PathBuf>,
    /// Index statistics
    stats: IndexStats,
}

impl IndexManager {
    /// Create a new index manager
    pub fn new(config: IndexConfig) -> Self {
        let stats_config = config.clone();
        Self {
            btree: None,
            fts: None,
            config,
            paths: HashMap::new(),
            stats: IndexStats {
                entries: 0,
                size: 0,
                build_time: 0,
                version: "1.0".to_string(),
                config: stats_config,
            },
        }
    }

    /// Build both B-TREE and FTS indexes
    pub fn build_all(&mut self, entries: &[(String, Vec<u8>)]) -> Result<()> {
        let start_time = std::time::Instant::now();
        let entries_count = entries.len() as u64;

        // Build B-TREE index
        if self.btree.is_none() {
            self.btree = Some(btree::BTreeIndex::new());
        }
        if let Some(ref mut btree) = self.btree {
            btree.build(entries, &self.config)?;
            self.stats.size += btree.stats().size;
        }

        // Build FTS index
        if self.fts.is_none() {
            self.fts = Some(fts::FtsIndex::new());
        }
        if let Some(ref mut fts) = self.fts {
            fts.build(entries, &self.config)?;
            self.stats.size += fts.stats().size;
        }

        self.stats.entries = entries_count;
        self.stats.build_time = start_time.elapsed().as_millis() as u64;

        Ok(())
    }

    /// Load indexes from files
    pub fn load_all(&mut self, base_path: &Path, extensions: &[(&str, &str)]) -> Result<()> {
        for (index_type, extension) in extensions {
            let index_path = base_path.with_extension(extension);

            match *index_type {
                "btree" => {
                    if !index_path.exists() {
                        return Err(DictError::FileNotFound(index_path.display().to_string()));
                    }
                    if self.btree.is_none() {
                        self.btree = Some(btree::BTreeIndex::new());
                    }
                    if let Some(ref mut btree) = self.btree {
                        btree.load(&index_path)?;
                        self.stats.size += btree.stats().size;
                    }
                }
                "fts" => {
                    if !index_path.exists() {
                        return Err(DictError::FileNotFound(index_path.display().to_string()));
                    }
                    if self.fts.is_none() {
                        self.fts = Some(fts::FtsIndex::new());
                    }
                    if let Some(ref mut fts) = self.fts {
                        fts.load(&index_path)?;
                        self.stats.size += fts.stats().size;
                    }
                }
                _ => {
                    return Err(DictError::Internal(format!(
                        "Unknown index type: {}",
                        index_type
                    )))
                }
            }
        }

        Ok(())
    }

    /// Save indexes to files
    pub fn save_all(&self, base_path: &Path, extensions: &[(&str, &str)]) -> Result<()> {
        for (index_type, extension) in extensions {
            let index_path = base_path.with_extension(extension);

            match *index_type {
                "btree" => {
                    if let Some(ref btree) = self.btree {
                        btree.save(&index_path)?;
                    }
                }
                "fts" => {
                    if let Some(ref fts) = self.fts {
                        fts.save(&index_path)?;
                    }
                }
                _ => {
                    return Err(DictError::Internal(format!(
                        "Unknown index type: {}",
                        index_type
                    )))
                }
            }
        }

        Ok(())
    }

    /// Binary search for a key using B-TREE index
    pub fn binary_search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>> {
        if let Some(ref btree) = self.btree {
            btree.search(key)
        } else {
            Err(DictError::IndexError(
                "B-TREE index not available".to_string(),
            ))
        }
    }

    /// Search using FTS index
    pub fn fulltext_search(&self, query: &str) -> Result<Vec<(String, f32)>> {
        if let Some(ref fts) = self.fts {
            fts.search(query)
        } else {
            Err(DictError::IndexError("FTS index not available".to_string()))
        }
    }

    /// Get all statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Check if indexes are built
    pub fn is_built(&self) -> bool {
        self.btree.as_ref().map(|b| b.is_built()).unwrap_or(false)
            && self.fts.as_ref().map(|f| f.is_built()).unwrap_or(false)
    }

    /// Clear all indexes
    pub fn clear(&mut self) {
        if let Some(ref mut btree) = self.btree {
            btree.clear();
        }
        if let Some(ref mut fts) = self.fts {
            fts.clear();
        }
        self.stats = IndexStats {
            entries: 0,
            size: 0,
            build_time: 0,
            version: "1.0".to_string(),
            config: self.config.clone(),
        };
    }

    /// Verify all indexes
    pub fn verify(&self) -> Result<bool> {
        let mut all_valid = true;

        if let Some(ref btree) = self.btree {
            if !btree.verify()? {
                all_valid = false;
            }
        }

        if let Some(ref fts) = self.fts {
            if !fts.verify()? {
                all_valid = false;
            }
        }

        Ok(all_valid)
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            btree_order: Some(256),
            fts_config: FtsConfig {
                min_token_len: 3,
                max_token_len: 64,
                use_stemming: true,
                stop_words: vec!["the".to_string(), "and".to_string(), "or".to_string()],
                language: Some("en".to_string()),
            },
            compression: Some(CompressionConfig {
                algorithm: CompressionAlgorithm::default(),
                level: 6,
            }),
            build_in_memory: true,
            max_memory: Some(1_000_000_000), // 1GB
        }
    }
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            min_token_len: 3,
            max_token_len: 64,
            use_stemming: true,
            stop_words: vec!["the".to_string(), "and".to_string(), "or".to_string()],
            language: Some("en".to_string()),
        }
    }
}

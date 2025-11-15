//! Core dictionary trait definitions and types
//!
//! This module provides the foundational trait and types that all dictionary
//! implementations must satisfy. It includes basic lookup operations, batch
//! operations, search functionality, and performance-optimized methods.

use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::ops::Range;
use std::path::Path;
use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

/// Result type for dictionary operations
pub type Result<T> = StdResult<T, DictError>;

/// Search result containing word and entry data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching word/term
    pub word: String,
    /// The dictionary entry data
    pub entry: Vec<u8>,
    /// Optional relevance score for search operations
    pub score: Option<f32>,
    /// Optional highlight information
    pub highlights: Option<Vec<(usize, usize)>>,
}

/// Batch lookup result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Original word that was searched
    pub word: String,
    /// Entry data if found
    pub entry: Option<Vec<u8>>,
    /// Error if lookup failed
    pub error: Option<DictError>,
}

/// Dictionary entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    /// Dictionary name
    pub name: String,
    /// Dictionary format version
    pub version: String,
    /// Total number of entries
    pub entries: u64,
    /// Dictionary description
    pub description: Option<String>,
    /// Dictionary author/publisher
    pub author: Option<String>,
    /// Dictionary language
    pub language: Option<String>,
    /// File size in bytes
    pub file_size: u64,
    /// Creation date
    pub created: Option<String>,
    /// Optional flags indicating available indexes
    pub has_btree: bool,
    pub has_fts: bool,
}

/// Error types for dictionary operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DictError {
    /// File not found
    FileNotFound(String),
    /// Invalid file format
    InvalidFormat(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// I/O error
    IoError(String),
    /// Memory mapping error
    MmapError(String),
    /// Index error
    IndexError(String),
    /// Decompression error
    DecompressionError(String),
    /// Serialization error
    SerializationError(String),
    /// Internal error with message
    Internal(String),
}

impl fmt::Display for DictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DictError::FileNotFound(path) => write!(f, "File not found: {}", path),
            DictError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            DictError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            DictError::IoError(msg) => write!(f, "I/O error: {}", msg),
            DictError::MmapError(msg) => write!(f, "Memory mapping error: {}", msg),
            DictError::IndexError(msg) => write!(f, "Index error: {}", msg),
            DictError::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
            DictError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            DictError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for DictError {}

impl From<std::io::Error> for DictError {
    fn from(err: std::io::Error) -> Self {
        DictError::IoError(err.to_string())
    }
}

/// Configuration options for dictionary initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictConfig {
    /// Whether to load B-TREE index for fast key lookups from existing on-disk index
    pub load_btree: bool,
    /// Whether to load full-text search index from existing on-disk index
    pub load_fts: bool,
    /// Whether to enable memory mapping for large files
    pub use_mmap: bool,
    /// Cache size for entries (number of entries to cache)
    pub cache_size: usize,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Custom encoding (auto-detect if None)
    pub encoding: Option<String>,
    /// Whether to (re)build and persist a B-TREE index when missing or outdated.
    /// Implementations must keep core index behavior unchanged and only orchestrate
    /// build/save for their own formats.
    pub build_btree: bool,
    /// Whether to (re)build and persist a full-text search index when missing or outdated.
    pub build_fts: bool,
}

impl Default for DictConfig {
    fn default() -> Self {
        Self {
            load_btree: true,
            load_fts: true,
            use_mmap: true,
            cache_size: 1000,
            batch_size: 100,
            encoding: None,
            // By default, allow building on-disk indexes when missing.
            build_btree: true,
            build_fts: true,
        }
    }
}

/// Iterator over dictionary entries
pub struct EntryIterator<'a, K> {
    pub keys: std::vec::IntoIter<K>,
    pub dictionary: &'a dyn Dict<K>,
}

impl<'a, K: Hash + Eq + Clone + fmt::Display> Iterator for EntryIterator<'a, K> {
    type Item = Result<(K, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.keys
            .next()
            .and_then(|key| match self.dictionary.get(&key) {
                Ok(entry) => Some(Ok((key, entry))),
                Err(e) => Some(Err(e)),
            })
    }
}

/// Core trait that defines all dictionary operations
pub trait Dict<K>: Send + Sync
where
    K: Hash + Eq + Clone + fmt::Display,
{
    /// Get the dictionary metadata
    fn metadata(&self) -> &DictMetadata;

    /// Check if the dictionary contains a specific key
    fn contains(&self, key: &K) -> Result<bool>;

    /// Get a specific entry by key
    fn get(&self, key: &K) -> Result<Vec<u8>>;

    /// Get multiple entries in a batch
    fn get_multiple(&self, keys: &[K]) -> Result<Vec<BatchResult>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(BatchResult {
                word: format!("{}", key),
                entry: self.get(key).ok(),
                error: self.get(key).err(),
            });
        }
        Ok(results)
    }

    /// Get a batch of entries for better performance
    fn get_batch(&self, keys: &[K], batch_size: Option<usize>) -> Result<Vec<BatchResult>> {
        let batch_size = batch_size.unwrap_or(100);
        let mut results = Vec::new();

        for chunk in keys.chunks(batch_size) {
            let mut chunk_results = self.get_multiple(chunk)?;
            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    /// Get all keys in the dictionary
    fn keys(&self) -> Result<Vec<K>> {
        let mut keys = Vec::new();
        for item in self.iter()? {
            keys.push(item?.0);
        }
        Ok(keys)
    }

    /// Get all values in the dictionary
    fn values(&self) -> Result<Vec<Vec<u8>>> {
        let mut values = Vec::new();
        for item in self.iter()? {
            values.push(item?.1);
        }
        Ok(values)
    }

    /// Search for entries that match a prefix
    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>>;

    /// Search for entries using fuzzy matching
    fn search_fuzzy(&self, query: &str, max_distance: Option<u32>) -> Result<Vec<SearchResult>>;

    /// Full-text search across all entries
    fn search_fulltext(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>>;

    /// Get entries by index range
    fn get_range(&self, range: Range<usize>) -> Result<Vec<(K, Vec<u8>)>>;

    /// Iterate over all entries
    fn iter(&self) -> Result<EntryIterator<K>>;

    /// Get an iterator for prefix matches
    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(K, Vec<u8>)>> + Send>>;

    /// Get the number of entries in the dictionary
    fn len(&self) -> usize;

    /// Check if the dictionary is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the file paths for the dictionary files
    fn file_paths(&self) -> &[std::path::PathBuf];

    /// Force reload of indexes if they exist
    fn reload_indexes(&mut self) -> Result<()>;

    /// Clear any cached data
    fn clear_cache(&mut self);

    /// Get statistics about the dictionary
    fn stats(&self) -> DictStats {
        DictStats {
            total_entries: self.len() as u64,
            cache_hit_rate: 0.0, // Default implementation
            memory_usage: 0,     // Default implementation
            index_sizes: HashMap::new(),
        }
    }

    /// Build indexes for this dictionary
    fn build_indexes(&mut self) -> Result<()>;
}

/// Statistics about dictionary performance and usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictStats {
    /// Total number of entries
    pub total_entries: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Estimated memory usage in bytes
    pub memory_usage: u64,
    /// Size of different indexes
    pub index_sizes: HashMap<String, u64>,
}

/// Trait for building dictionaries (for creating new dictionary files)
pub trait DictBuilder<K> {
    /// Add an entry to the dictionary
    fn add_entry(&mut self, key: K, entry: &[u8]) -> Result<()>;

    /// Build and write the dictionary to disk
    fn build(&mut self, output_path: &Path, config: Option<DictConfig>) -> Result<()>;

    /// Set metadata for the dictionary
    fn set_metadata(&mut self, metadata: DictMetadata);

    /// Get the current number of entries
    fn len(&self) -> usize;

    /// Check if the builder is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for high-performance operations
pub trait HighPerformanceDict<K>: Dict<K>
where
    K: Hash + Eq + Clone + fmt::Display,
{
    /// Get entry using binary search on the key index
    fn binary_search_get(&self, key: &K) -> Result<Vec<u8>>;

    /// Stream large result sets without loading everything into memory
    fn stream_search(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>>;
}

/// Trait for dictionary formats
pub trait DictFormat<K> {
    /// Format name
    const FORMAT_NAME: &'static str;

    /// Format version
    const FORMAT_VERSION: &'static str;

    /// Check if a path contains a valid dictionary of this format
    fn is_valid_format(path: &Path) -> Result<bool>;

    /// Load dictionary from file(s)
    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<K> + Send + Sync>>;
}

/// Constants for different dictionary formats
pub const FORMAT_MDICT: &str = "mdict";
pub const FORMAT_STARDICT: &str = "stardict";
pub const FORMAT_ZIM: &str = "zim";

/// File extensions for different dictionary components
pub const EXT_DICT: &str = ".dict";
pub const EXT_IDX: &str = ".idx";
pub const EXT_INFO: &str = ".info";
pub const EXT_BTREE: &str = ".btree";
pub const EXT_FTS: &str = ".fts";

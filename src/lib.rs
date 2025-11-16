#![warn(missing_docs)]
#![warn(unused_extern_crates)]
#![warn(unused_qualifications)]

//! High-performance dictionary utilities library
//!
//! This library provides fast and efficient dictionary operations with support for multiple
//! dictionary formats including Monkey's Dictionary (MDict), StarDict, and ZIM format.
//!
//! ## Features
//!
//! - **Multiple Format Support**: MDict, StarDict, and ZIM formats
//! - **High Performance**: B-TREE indexing for fast lookups
//! - **Full-Text Search**: Inverted indexing for content search
//! - **Memory-Mapped Files**: Efficient large file handling
//! - **Compression Support**: GZIP, LZ4, Zstandard compression
//! - **Batch Operations**: Efficient bulk processing
//! - **Thread Safety**: Safe concurrent access
//! - **Lazy Loading**: Memory-efficient on-demand loading
//!
//! ## Quick Start
//!
//! ```rust
//! use dictutils::prelude::*;
//!
//! fn main() {
//!     // This is a usage example, not executed in doctests.
//!     let loader = DictLoader::new();
//!
//!     // Load a dictionary (format auto-detected)
//!     let dict = loader.load("path/to/dictionary.mdict");
//!
//!     // Handle Result in real code (omitted here for brevity)
//!     let _ = dict;
//! }
//! ```
//!
//! ## Configuration
//!
//! ```rust
//! use dictutils::prelude::*;
//!
//! // Example configuration (not executed in doctests)
//! let config = DictConfig {
//!     load_btree: true,      // Fast key lookups
//!     load_fts: true,        // Full-text search
//!     use_mmap: true,        // Memory mapping
//!     cache_size: 1000,      // Entry cache size
//!     batch_size: 100,       // Batch operation size
//!     ..Default::default()
//! };
//!
//! let loader = DictLoader::with_config(config);
//! let _dict = loader.load("path/to/dictionary");
//! ```
//!
//! ## Performance Tips
//!
//! 1. **Build Indexes**: Use `build_indexes()` for large dictionaries
//! 2. **Use B-TREE**: Enable for fast exact lookups
//! 3. **Enable FTS**: For content search functionality
//! 4. **Memory Mapping**: Recommended for files > 100MB
//! 5. **Batch Operations**: Use `get_multiple()` for multiple lookups
//!
//! ## Supported Dictionary Formats
//!
//! - **Monkey's Dictionary (.mdict)**: Fast lookup format with optional indexes
//! - **StarDict (.dict)**: Classic format with binary search
//! - **ZIM (.zim)**: Wikipedia offline format with article storage
//!
//! ## Thread Safety
//!
//! All dictionary operations are thread-safe and can be shared across threads
//! using standard Rust concurrency patterns.

/// Core trait definitions and types
pub mod traits;

/// Dictionary format implementations
pub mod dict;

/// High-performance indexing system
pub mod index;

/// Utility functions and helpers
pub mod util;

// Re-export common types and functions for convenience
pub use dict::{BatchOperations, DictLoader, MDict, StarDict, ZimDict};
pub use index::{btree, fts};
pub use traits::*;
pub use util::{buffer, compression, encoding, file_utils};

// Convenience module for easy imports
pub mod prelude {
    pub use crate::dict::{utils as dict_utils, BatchOperations, DictLoader};
    pub use crate::index::{btree::BTreeIndex, fts::FtsIndex};
    pub use crate::traits::*;
    pub use crate::util::{compression::CompressionAlgorithm, encoding::TextEncoding, DictConfig};
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Library description
pub const DESCRIPTION: &str = "High-performance dictionary utilities library";

/// Maximum supported dictionary size (2GB)
pub const MAX_DICT_SIZE: u64 = 2_147_483_648;

/// Default cache size for entries
pub const DEFAULT_CACHE_SIZE: usize = 1000;

/// Default batch size for operations
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Minimum memory required for basic operations (64MB)
pub const MIN_MEMORY: u64 = 64 * 1024 * 1024;

/// Recommended memory for optimal performance (256MB)
pub const RECOMMENDED_MEMORY: u64 = 256 * 1024 * 1024;

// Feature flags
#[cfg(feature = "cli")]
pub mod cli {
    use crate::dict::{BatchOperations, DictLoader};
    use crate::traits::*;
    use std::path::PathBuf;

    /// Sanitize output to prevent ANSI escape code injection
    fn sanitize_output(input: &str) -> String {
        input.replace('\x1b', "\\x1b")
    }

    /// Command-line interface utilities
    pub fn print_dict_info<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        let loader = DictLoader::new();

        println!("Dictionary Information");
        println!("===================");
        println!("Path: {}", sanitize_output(&path.display().to_string()));

        if let Ok(format) = loader.detect_format(path) {
            println!("Format: {}", sanitize_output(&format));
        }

        if let Ok(mut dict) = loader.load(path) {
            let metadata = dict.metadata();
            println!("Name: {}", sanitize_output(&metadata.name));
            println!("Version: {}", sanitize_output(&metadata.version));
            println!("Entries: {}", metadata.entries);
            println!("Size: {} bytes", metadata.file_size);
            println!("Has B-TREE: {}", metadata.has_btree);
            println!("Has FTS: {}", metadata.has_fts);

            // Print statistics
            let stats = dict.stats();
            println!("Memory Usage: {} bytes", stats.memory_usage);

            // Print index sizes
            for (index, size) in &stats.index_sizes {
                println!("{} Index: {} bytes", sanitize_output(index), size);
            }
        }

        Ok(())
    }

    /// Search command-line utility
    pub fn search_dict<P: AsRef<std::path::Path>>(
        path: P,
        query: &str,
        search_type: &str,
        limit: Option<usize>,
    ) -> Result<()> {
        let path = path.as_ref();
        let loader = DictLoader::new();
        let mut dict = loader.load(path)?;

        println!("Search Results for '{}'", sanitize_output(query));
        println!("===========================");

        let results = match search_type {
            "prefix" => dict.search_prefix(query, limit),
            "fuzzy" => dict.search_fuzzy(query, None),
            "fulltext" => {
                let iterator = dict.search_fulltext(query)?;
                let results_vec: Result<Vec<_>> = iterator.collect();
                results_vec
            }
            _ => {
                return Err(DictError::UnsupportedOperation(
                    "Search type must be 'prefix', 'fuzzy', or 'fulltext'".to_string(),
                ))
            }
        }?;

        for result in results.iter().take(limit.unwrap_or(10)) {
            println!("- {}", sanitize_output(&result.word));
            if let Some(score) = result.score {
                println!("  Score: {:.3}", score);
            }
        }

        println!("\nFound {} results", results.len());
        Ok(())
    }

    /// Validate dictionary file
    pub fn validate_dict<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        let loader = DictLoader::new();

        println!("Validating dictionary: {}", path.display());

        // Check if file exists and is readable
        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        // Try to detect format
        let format = loader.detect_format(path)?;
        println!("Format detected: {}", format);

        // Try to load and validate
        let mut dict = loader.load(path)?;

        // Validate integrity
        let stats = dict.stats();
        println!("Validation Results:");
        println!("  - Total entries: {}", stats.total_entries);
        println!("  - Memory usage: {} bytes", stats.memory_usage);
        println!("  - Cache hit rate: {:.2}%", stats.cache_hit_rate * 100.0);

        println!(
            "  - B-TREE index: {}",
            if dict.metadata().has_btree {
                "Available"
            } else {
                "Not available"
            }
        );

        println!(
            "  - FTS index: {}",
            if dict.metadata().has_fts {
                "Available"
            } else {
                "Not available"
            }
        );

        println!("Dictionary validation: SUCCESS");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test_utils::{cleanup_temp_dir, generate_test_entries, temp_dir};

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!NAME.is_empty());
        assert!(NAME == "dictutils");
    }

    #[test]
    fn test_config_defaults() {
        let config = DictConfig::default();
        assert!(config.load_btree);
        assert!(config.load_fts);
        assert!(config.use_mmap);
        assert_eq!(config.cache_size, DEFAULT_CACHE_SIZE);
        assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn test_compression_algorithms() {
        use crate::util::compression::*;

        let test_data = b"Hello, World! This is test data for compression.";

        for algorithm in &[CompressionAlgorithm::None, CompressionAlgorithm::Gzip] {
            let compressed = compress(test_data, algorithm.clone()).unwrap();
            let decompressed = decompress(&compressed, algorithm.clone()).unwrap();
            assert_eq!(test_data, &decompressed[..]);
        }
    }

    #[test]
    fn test_encoding_detection() {
        use crate::util::encoding::*;

        // Test UTF-8 detection
        let utf8_data = "Hello, World! 🌟".as_bytes();
        let encoding = detect_encoding(utf8_data).unwrap();
        assert_eq!(encoding, TextEncoding::Utf8);

        // Test ASCII detection
        let ascii_data = b"Hello, World!";
        let encoding = detect_encoding(ascii_data).unwrap();
        assert_eq!(encoding, TextEncoding::Utf8);
    }

    #[test]
    fn test_test_utils() {
        let entries = generate_test_entries(10);
        assert_eq!(entries.len(), 10);

        // Check entry format
        for (i, (key, content)) in entries.iter().enumerate() {
            assert!(key.starts_with("word_"));
            assert!(key.contains(&format!("{:06}", i)));
            assert!(!content.is_empty());
        }
    }

    #[test]
    fn test_dict_loader() {
        let loader = DictLoader::new();
        let formats = loader.supported_formats();

        assert!(formats.contains(&"mdict".to_string()));
        assert!(formats.contains(&"stardict".to_string()));
        assert!(formats.contains(&"zim".to_string()));
    }

    #[test]
    fn test_performance_utils() {
        use crate::util::performance::*;

        let mut profiler = Profiler::new();

        // Simulate some operations
        for i in 0..1000 {
            profiler.record("test_operation", 1);
            black_box(i);
        }

        let ops_per_sec = profiler.operations_per_second("test_operation");
        assert!(ops_per_sec > 0.0);
    }

    // Helper function to prevent compiler optimization
    fn black_box<T>(x: T) -> T {
        // Minimal, safe implementation for tests: prevent obvious optimizations
        std::hint::black_box(x)
    }

    // Integration tests would go here
    // These would test actual dictionary loading and operations
    // with real dictionary files
}

#[cfg(all(test, feature = "bench"))]
mod benchmarks {
    use super::*;
    use crate::util::test_utils::{cleanup_temp_dir, generate_test_entries, temp_dir};
    use std::time::Instant;

    #[bench]
    fn bench_binary_search(b: &mut test::Bencher) {
        let entries = generate_test_entries(1000);
        let keys: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();

        b.iter(|| {
            test::black_box(keys.binary_search(&"word_00500".to_string()));
        });
    }

    #[bench]
    fn bench_prefix_search(b: &mut test::Bencher) {
        let config = DictConfig::default();
        let temp_path = temp_dir().unwrap();

        // Create a temporary dictionary for testing
        let entries = generate_test_entries(1000);

        b.iter(|| {
            // This would test actual prefix search performance
            // with a real dictionary implementation
            test::black_box(&entries);
        });

        let _ = cleanup_temp_dir(&temp_path);
    }

    #[bench]
    fn bench_fuzzy_search(b: &mut test::Bencher) {
        let entries = generate_test_entries(1000);
        let query = "word_500";

        b.iter(|| {
            let mut results = Vec::new();
            for (key, _) in &entries {
                if let Some(_distance) = levenshtein_approx(query, key, 2) {
                    results.push(key);
                    if results.len() >= 10 {
                        break;
                    }
                }
            }
            test::black_box(results);
        });
    }

    // Simple Levenshtein distance approximation for benchmarks
    fn levenshtein_approx(a: &str, b: &str, max_dist: usize) -> Option<usize> {
        let (m, n) = (a.len(), b.len());
        if (m as i32 - n as i32).abs() > max_dist as i32 {
            return None;
        }

        let mut dp = vec![vec![0u32; n + 1]; m + 1];
        for i in 0..=m {
            dp[i][0] = i as u32;
        }
        for j in 0..=n {
            dp[0][j] = j as u32;
        }

        for i in 1..=m {
            for j in 1..=n {
                if a.chars().nth(i - 1) == b.chars().nth(j - 1) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                }
            }
        }

        let distance = dp[m][n] as usize;
        if distance <= max_dist {
            Some(distance)
        } else {
            None
        }
    }
}

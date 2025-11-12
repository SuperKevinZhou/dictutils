//! Test utilities and mock data generation for dictionary testing
//!
//! This module provides utilities for creating test dictionaries, mock data,
//! and helper functions for testing dictionary operations.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

use crate::traits::{DictConfig, DictMetadata, DictError, Result};
use crate::util::compression::{CompressionAlgorithm, compress};
use crate::util::encoding::{TextEncoding, convert_to_utf8};

/// Generate test entries for dictionary testing
pub fn generate_test_entries(count: usize) -> Vec<(String, Vec<u8>)> {
    let mut entries = Vec::with_capacity(count);
    
    for i in 0..count {
        let key = format!("word_{:06}", i);
        let content = format!("This is the definition for word {} with some additional content to make it realistic.", i);
        let content_bytes = content.into_bytes();
        entries.push((key, content_bytes));
    }
    
    entries
}

/// Generate test entries with specific patterns for testing
pub fn generate_patterned_entries(pattern: &str, count: usize) -> Vec<(String, Vec<u8>)> {
    let mut entries = Vec::with_capacity(count);
    
    for i in 0..count {
        let key = format!("{}_{}", pattern, i);
        let content = format!("Definition for {} item {} with detailed explanation.", pattern, i);
        let content_bytes = content.into_bytes();
        entries.push((key, content_bytes));
    }
    
    entries
}

/// Create a temporary directory for testing
pub fn temp_dir() -> Result<TempDir> {
    tempdir().map_err(|e| DictError::IoError(format!("Failed to create temp dir: {}", e)))
}

/// Clean up temporary directory
pub fn cleanup_temp_dir(temp_dir: &TempDir) -> Result<()> {
    drop(temp_dir);
    Ok(())
}

/// Create a mock MDict file for testing
pub fn create_mock_mdict(path: &Path, entries: &[(String, Vec<u8>)]) -> Result<()> {
    use std::io::{BufWriter, Seek, SeekFrom};

    let file = File::create(path)
        .map_err(|e| DictError::IoError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write magic header
    writer.write_all(b"MDict")?;
    writer.write_all(b"Version 2.0.0")?;
    
    // Write entry count
    let entry_count = entries.len() as u64;
    writer.write_all(&entry_count.to_le_bytes())?;
    
    // Write offsets (simplified for testing)
    let key_index_offset = 100u64;
    let value_index_offset = 100u64;
    let key_block_offset = 200u64;
    let value_block_offset = 300u64;
    
    writer.write_all(&key_index_offset.to_le_bytes())?;
    writer.write_all(&value_index_offset.to_le_bytes())?;
    writer.write_all(&key_block_offset.to_le_bytes())?;
    writer.write_all(&value_block_offset.to_le_bytes())?;
    
    // Write encoding and compression
    writer.write_all(&0u32.to_le_bytes())?; // UTF-8
    writer.write_all(&3u32.to_le_bytes())?; // Zstandard
    
    // Write file size and checksum
    let file_size = writer.seek(SeekFrom::Current(0))?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?; // Dummy checksum
    
    // Write metadata
    writer.write_all(&1u32.to_le_bytes())?; // One metadata entry
    writer.write_all(&4u32.to_le_bytes())?; // "name" length
    writer.write_all(b"name")?;
    writer.write_all(&9u32.to_le_bytes())?; // "test_dict" length
    writer.write_all(b"test_dict")?;
    
    writer.flush()?;
    
    Ok(())
}

/// Create a mock StarDict file for testing
pub fn create_mock_stardict(path: &Path, entries: &[(String, Vec<u8>)]) -> Result<()> {
    use std::io::{BufWriter, Seek, SeekFrom};

    let file = File::create(path)
        .map_err(|e| DictError::IoError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write magic header
    writer.write_all(b"StarDict's dict")?;
    writer.write_all(b"3.0.0\0")?;
    
    // Write entry count
    let entry_count = entries.len() as u32;
    writer.write_all(&entry_count.to_le_bytes())?;
    
    // Write offsets (simplified for testing)
    let table_offset = 100u32;
    let data_offset = 200u32;
    let data_length = 1024u32;
    let file_size = 2048u32;
    
    writer.write_all(&table_offset.to_le_bytes())?;
    writer.write_all(&data_offset.to_le_bytes())?;
    writer.write_all(&data_length.to_le_bytes())?;
    writer.write_all(&file_size.to_le_bytes())?;
    
    // Write compression flag
    writer.write_all(&0u32.to_le_bytes())?; // No compression
    
    // Write features
    writer.write_all(&0u32.to_le_bytes())?;
    
    writer.flush()?;
    
    Ok(())
}

/// Create a mock ZIM file for testing
pub fn create_mock_zim(path: &Path, entries: &[(String, Vec<u8>)]) -> Result<()> {
    use std::io::BufWriter;

    let file = File::create(path)
        .map_err(|e| DictError::IoError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write ZIM magic header
    writer.write_all(b"ZIM\x01")?;
    
    // Write header structure (simplified)
    writer.write_all(&0u32.to_le_bytes())?; // version
    writer.write_all(&0u16.to_le_bytes())?; // flags
    writer.write_all(&0u32.to_le_bytes())?; // mimeTypeCounter
    writer.write_all(&0u32.to_le_bytes())?; // urlPtrPos
    writer.write_all(&0u32.to_le_bytes())?; // titlePtrPos
    writer.write_all(&0u32.to_le_bytes())?; // clusterPtrPos
    writer.write_all(&0u32.to_le_bytes())?; // headerPos
    writer.write_all(&0u16.to_le_bytes())?; // checksumPos
    writer.write_all(&0u16.to_le_bytes())?; // guid
    
    writer.flush()?;
    
    Ok(())
}

/// Create a test dictionary with specified format and entries
pub fn create_test_dict<P: AsRef<Path>>(path: P, format: &str, entries: &[(String, Vec<u8>)]) -> Result<()> {
    let path = path.as_ref();
    
    match format {
        "mdict" => create_mock_mdict(path, entries),
        "stardict" => create_mock_stardict(path, entries),
        "zim" => create_mock_zim(path, entries),
        _ => Err(DictError::UnsupportedOperation(format!("Unsupported format: {}", format))),
    }
}

/// Generate random text for testing
pub fn generate_random_text(length: usize) -> String {
    let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    let mut text = String::new();
    
    for _ in 0..length {
        let idx = rand::random::<usize>() % chars.len();
        text.push(chars.chars().nth(idx).unwrap());
    }
    
    text
}

/// Generate test metadata
pub fn generate_test_metadata(name: &str) -> DictMetadata {
    DictMetadata {
        name: name.to_string(),
        version: "1.0".to_string(),
        entries: 100,
        description: Some("Test dictionary".to_string()),
        author: Some("Test Author".to_string()),
        language: Some("English".to_string()),
        file_size: 1024,
        created: Some("2023-01-01".to_string()),
        has_btree: true,
        has_fts: true,
    }
}

/// Get test configuration
pub fn test_config() -> DictConfig {
    DictConfig {
        load_btree: true,
        load_fts: true,
        use_mmap: false, // Disable for testing
        cache_size: 100,
        batch_size: 50,
        encoding: None,
    }
}

/// Create test data with Unicode content
pub fn generate_unicode_entries(count: usize) -> Vec<(String, Vec<u8>)> {
    let mut entries = Vec::with_capacity(count);
    
    let unicode_texts = vec![
        "Hello 世界 🌍",
        "Привет мир",
        "مرحبا بالعالم", 
        "こんにちは世界",
        "안녕하세요 세계",
        "Γεια σου κόσμε",
    ];
    
    for i in 0..count {
        let key = format!("unicode_word_{:03}", i);
        let text = unicode_texts[i % unicode_texts.len()];
        let content = format!("Definition for {}: {}", key, text);
        let content_bytes = content.into_bytes();
        entries.push((key, content_bytes));
    }
    
    entries
}

/// Benchmark helper for generating large datasets
pub fn generate_large_dataset(size: usize) -> Vec<(String, Vec<u8>)> {
    let mut entries = Vec::with_capacity(size);
    
    for i in 0..size {
        let key = format!("benchmark_key_{:08}", i);
        let content = format!("This is a benchmark entry number {} with substantial content to test memory usage and performance characteristics during dictionary operations.", i);
        let content_bytes = content.into_bytes();
        entries.push((key, content_bytes));
    }
    
    entries
}

/// Test file utilities
pub mod file_helpers {
    use super::*;
    use std::fs::read_to_string;
    
    /// Read file content as string for testing
    pub fn read_test_file(path: &Path) -> Result<String> {
        read_to_string(path)
            .map_err(|e| DictError::IoError(e.to_string()))
    }
    
    /// Check if file exists and is readable
    pub fn is_test_file_readable(path: &Path) -> bool {
        path.exists() && path.is_file() && path.metadata().map(|m| m.len()).unwrap_or(0) > 0
    }
    
    /// Get temporary test files directory
    pub fn test_files_dir() -> Result<PathBuf> {
        let temp_dir = tempdir()
            .map_err(|e| DictError::IoError(format!("Failed to create temp dir: {}", e)))?;
        Ok(temp_dir.into_path())
    }
}

/// Memory testing utilities
pub mod memory_helpers {
    use std::mem::{size_of, size_of_val};
    use std::alloc::{GlobalAlloc, Layout, System};
    
    /// Get approximate memory usage of entries
    pub fn estimate_entries_memory(entries: &[(String, Vec<u8>)]) -> usize {
        let mut total = 0;
        
        for (key, value) in entries {
            total += size_of_val(key);
            total += size_of_val(value);
            total += key.len();
            total += value.len();
        }
        
        total
    }
    
    /// Track memory allocation for testing
    pub struct MemoryTracker {
        start_allocated: usize,
        start_deallocated: usize,
    }
    
    impl MemoryTracker {
        pub fn new() -> Self {
            Self {
                start_allocated: 0,
                start_deallocated: 0,
            }
        }
        
        pub fn start(&mut self) {
            // This would track actual memory allocations
            // For now, just reset counters
            self.start_allocated = 0;
            self.start_deallocated = 0;
        }
        
        pub fn get_current_usage(&self) -> usize {
            // Return estimated current usage
            0
        }
    }
    
    impl Default for MemoryTracker {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Performance testing utilities
pub mod perf_helpers {
    use std::time::{Duration, Instant};
    
    /// Simple performance timer
    pub struct PerfTimer {
        start: Instant,
    }
    
    impl PerfTimer {
        pub fn new() -> Self {
            Self {
                start: Instant::now(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }
    }
    
    impl Default for PerfTimer {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Mock dictionary builder for testing
pub struct MockDictBuilder {
    entries: Vec<(String, Vec<u8>)>,
    metadata: DictMetadata,
    compression: CompressionAlgorithm,
    encoding: TextEncoding,
}

impl MockDictBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            metadata: generate_test_metadata("mock_dict"),
            compression: CompressionAlgorithm::None,
            encoding: TextEncoding::Utf8,
        }
    }
    
    pub fn add_entry(&mut self, key: String, value: Vec<u8>) -> &mut Self {
        self.entries.push((key, value));
        self.metadata.entries = self.entries.len() as u64;
        self
    }
    
    pub fn set_compression(&mut self, compression: CompressionAlgorithm) -> &mut Self {
        self.compression = compression;
        self
    }
    
    pub fn set_encoding(&mut self, encoding: TextEncoding) -> &mut Self {
        self.encoding = encoding;
        self
    }
    
    pub fn set_metadata(&mut self, metadata: DictMetadata) -> &mut Self {
        self.metadata = metadata;
        self
    }
    
    pub fn build(self) -> (Vec<(String, Vec<u8>)>, DictMetadata) {
        (self.entries, self.metadata)
    }
}

impl Default for MockDictBuilder {
    fn default() -> Self {
        Self::new()
    }
}
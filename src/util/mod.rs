//! Utility functions for dictionary operations
//!
//! This module provides utility functions for file operations, compression,
//! encoding detection, and other common operations used by dictionary implementations.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::result::Result as StdResult;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use serde::{Deserialize, Serialize};
use zstd::{Decoder, Encoder};

use crate::traits::{DictError, Result};

pub mod compression;
pub mod encoding;

/// File utility functions
pub mod file_utils {
    use super::*;

    /// Read entire file into memory
    pub fn read_file(path: &Path) -> Result<Vec<u8>> {
        let mut file = File::open(path).map_err(|e| DictError::IoError(e.to_string()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        Ok(buffer)
    }

    /// Read file with memory mapping
    pub fn read_file_mmap(path: &Path) -> Result<memmap2::Mmap> {
        let file = File::open(path).map_err(|e| DictError::IoError(e.to_string()))?;

        unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| DictError::MmapError(e.to_string()))
        }
    }

    /// Write data to file with atomic operations
    pub fn write_file_atomic(path: &Path, data: &[u8]) -> Result<()> {
        let temp_path = path.with_extension("tmp");
        let mut file = File::create(&temp_path).map_err(|e| DictError::IoError(e.to_string()))?;

        file.write_all(data)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        file.sync_all()
            .map_err(|e| DictError::IoError(e.to_string()))?;

        std::fs::rename(&temp_path, path).map_err(|e| DictError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Get file size
    pub fn file_size(path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(path).map_err(|e| DictError::IoError(e.to_string()))?;

        Ok(metadata.len())
    }

    /// Check if file exists and is readable
    pub fn is_readable(path: &Path) -> bool {
        File::open(path).is_ok()
    }

    /// Create directory if it doesn't exist
    pub fn ensure_dir(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| DictError::IoError(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Calculate CRC32 checksum
    pub fn crc32(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    /// Verify file integrity with CRC32
    pub fn verify_crc32(path: &Path, expected_crc: u32) -> Result<bool> {
        let data = read_file(path)?;
        let actual_crc = crc32(&data);
        Ok(actual_crc == expected_crc)
    }
}

/// Buffer utilities for efficient I/O operations
pub mod buffer {
    use super::*;

    /// Read bytes from a reader with error handling
    pub fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            match reader.read(&mut buf[offset..]) {
                Ok(0) => return Err(DictError::IoError("Unexpected EOF".to_string())),
                Ok(n) => {
                    offset += n;
                }
                Err(e) => return Err(DictError::IoError(e.to_string())),
            }
        }
        Ok(())
    }

    /// Read 32-bit unsigned integer (little-endian)
    pub fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
        let mut buf = [0u8; 4];
        read_exact(reader, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read 32-bit unsigned integer (big-endian)
    pub fn read_u32_be<R: Read>(reader: &mut R) -> Result<u32> {
        let mut buf = [0u8; 4];
        read_exact(reader, &mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    /// Read 64-bit unsigned integer (little-endian)
    pub fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64> {
        let mut buf = [0u8; 8];
        read_exact(reader, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read 64-bit unsigned integer (big-endian)
    pub fn read_u64_be<R: Read>(reader: &mut R) -> Result<u64> {
        let mut buf = [0u8; 8];
        read_exact(reader, &mut buf)?;
        Ok(u64::from_be_bytes(buf))
    }

    /// Read variable-length integer (VARINT)
    pub fn read_varint<R: Read>(reader: &mut R) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            let mut byte = [0u8; 1];
            read_exact(reader, &mut byte)?;
            let b = byte[0];

            result |= ((b & 0x7F) as u64) << shift;
            shift += 7;

            if (b & 0x80) == 0 {
                break;
            }

            if shift > 63 {
                return Err(DictError::Internal("VARINT too large".to_string()));
            }
        }
        Ok(result)
    }

    /// Read length-prefixed string
    pub fn read_string<R: Read, F: FnMut(String) -> Result<()>>(
        reader: &mut R,
        mut callback: F,
    ) -> Result<()> {
        let length = read_varint(reader)? as usize;
        let mut buffer = vec![0u8; length];
        read_exact(reader, &mut buffer)?;

        let s = String::from_utf8(buffer)
            .map_err(|e| DictError::Internal(format!("Invalid UTF-8: {}", e)))?;

        callback(s)
    }

    /// Write bytes to a writer with error handling
    pub fn write_all<W: Write>(writer: &mut W, buf: &[u8]) -> Result<()> {
        let mut remaining = buf;
        while !remaining.is_empty() {
            match writer.write(remaining) {
                Ok(0) => return Err(DictError::IoError("Write returned 0 bytes".to_string())),
                Ok(n) => {
                    remaining = &remaining[n..];
                }
                Err(e) => return Err(DictError::IoError(e.to_string())),
            }
        }
        Ok(())
    }

    /// Write 32-bit unsigned integer (little-endian)
    pub fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> Result<()> {
        let bytes = value.to_le_bytes();
        write_all(writer, &bytes)
    }

    /// Write 32-bit unsigned integer (big-endian)
    pub fn write_u32_be<W: Write>(writer: &mut W, value: u32) -> Result<()> {
        let bytes = value.to_be_bytes();
        write_all(writer, &bytes)
    }

    /// Write 64-bit unsigned integer (little-endian)
    pub fn write_u64_le<W: Write>(writer: &mut W, value: u64) -> Result<()> {
        let bytes = value.to_le_bytes();
        write_all(writer, &bytes)
    }

    /// Write 64-bit unsigned integer (big-endian)
    pub fn write_u64_be<W: Write>(writer: &mut W, value: u64) -> Result<()> {
        let bytes = value.to_be_bytes();
        write_all(writer, &bytes)
    }

    /// Write variable-length integer (VARINT)
    pub fn write_varint<W: Write>(writer: &mut W, mut value: u64) -> Result<()> {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value > 0 {
                byte |= 0x80;
            }

            write_all(writer, &[byte])?;

            if value == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Write length-prefixed string
    pub fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        write_varint(writer, bytes.len() as u64)?;
        write_all(writer, bytes)
    }

    /// Read 8-bit unsigned integer
    pub fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
        let mut buf = [0u8; 1];
        read_exact(reader, &mut buf)?;
        Ok(buf[0])
    }

    /// Read 16-bit unsigned integer (little-endian)
    pub fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16> {
        let mut buf = [0u8; 2];
        read_exact(reader, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read 16-bit unsigned integer (big-endian)
    pub fn read_u16_be<R: Read>(reader: &mut R) -> Result<u16> {
        let mut buf = [0u8; 2];
        read_exact(reader, &mut buf)?;
        Ok(u16::from_be_bytes(buf))
    }
}

/// Binary search utilities
pub mod binary_search {
    use super::*;

    /// Binary search for a key in a sorted array of keys with values
    pub fn search_sorted<'a, K, V>(
        keys: &'a [K],
        values: &'a [V],
        target: &K,
        compare: impl Fn(&K, &K) -> std::cmp::Ordering,
    ) -> Option<(usize, &'a V)>
    where
        K: Ord,
    {
        assert_eq!(keys.len(), values.len());

        let mut left = 0;
        let mut right = keys.len();

        while left < right {
            let mid = left + (right - left) / 2;
            match compare(&keys[mid], target) {
                std::cmp::Ordering::Equal => return Some((mid, &values[mid])),
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
            }
        }
        None
    }

    /// Binary search for lower bound
    pub fn lower_bound<K>(
        keys: &[K],
        target: &K,
        compare: impl Fn(&K, &K) -> std::cmp::Ordering,
    ) -> usize
    where
        K: Ord,
    {
        let mut left = 0;
        let mut right = keys.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if compare(&keys[mid], target) == std::cmp::Ordering::Less {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    /// Binary search for upper bound
    pub fn upper_bound<K>(
        keys: &[K],
        target: &K,
        compare: impl Fn(&K, &K) -> std::cmp::Ordering,
    ) -> usize
    where
        K: Ord,
    {
        let mut left = 0;
        let mut right = keys.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if compare(&keys[mid], target) == std::cmp::Ordering::Greater {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        left
    }
}

/// Memory management utilities
pub mod memory {
    use super::*;

    /// Calculate optimal cache size based on available memory
    pub fn optimal_cache_size(entries: usize, avg_entry_size: usize) -> usize {
        let total_size = entries as u64 * avg_entry_size as u64;
        let available = total_memory() / 2; // Use half of available memory

        if total_size <= available {
            entries // Cache everything
        } else {
            // Calculate cache size proportional to memory usage
            (available / avg_entry_size as u64) as usize
        }
    }

    /// Get total available system memory (bytes)
    pub fn total_memory() -> u64 {
        sysinfo::System::new_all().total_memory()
    }

    /// Get currently used memory by the process (bytes)
    pub fn used_memory() -> u64 {
        sysinfo::System::new_all().used_memory()
    }

    /// Check if we have enough memory for an operation
    pub fn has_sufficient_memory(required: u64) -> bool {
        total_memory() - used_memory() >= required
    }

    /// Clear memory buffer to prevent data leakage
    pub fn clear_buffer(buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            *byte = 0;
        }
    }

    /// Securely zero sensitive data
    pub fn zero_sensitive<T: Default>(data: &mut T) {
        unsafe {
            std::ptr::write_volatile(
                data as *mut T,
                std::mem::MaybeUninit::zeroed().assume_init(),
            );
        }
    }
}

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Simple performance profiler
    #[derive(Debug)]
    pub struct Profiler {
        start_time: Instant,
        operations: std::collections::HashMap<String, u64>,
    }

    impl Profiler {
        pub fn new() -> Self {
            Self {
                start_time: Instant::now(),
                operations: std::collections::HashMap::new(),
            }
        }

        /// Record an operation count
        pub fn record(&mut self, operation: &str, count: u64) {
            *self.operations.entry(operation.to_string()).or_insert(0) += count;
        }

        /// Get elapsed time since profiler creation
        pub fn elapsed(&self) -> std::time::Duration {
            self.start_time.elapsed()
        }

        /// Print statistics
        pub fn print_stats(&self) {
            println!("Total time: {:?}", self.elapsed());
            for (op, count) in &self.operations {
                println!("{}: {} operations", op, count);
            }
        }

        /// Get operations per second
        pub fn operations_per_second(&self, operation: &str) -> f64 {
            let elapsed = self.elapsed().as_secs_f64();
            let count = self.operations.get(operation).unwrap_or(&0);
            if elapsed > 0.0 {
                *count as f64 / elapsed
            } else {
                0.0
            }
        }
    }

    /// Measure function execution time
    pub fn measure_time<T>(f: impl FnOnce() -> T) -> (T, std::time::Duration) {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        (result, elapsed)
    }

    /// Benchmark a function
    pub fn benchmark<T>(
        iterations: usize,
        mut f: impl FnMut() -> T,
    ) -> (T, std::time::Duration, std::time::Duration) {
        // Warmup phase
        let warmup_time = measure_time(|| {
            for _ in 0..std::cmp::min(10, iterations) {
                let _ = black_box(f());
            }
        });

        // Measurement phase
        let (last_result, result_time) = measure_time(|| {
            let mut last = None;
            for _ in 0..iterations {
                last = Some(black_box(f()));
            }
            // Return last result to the caller
            last.expect("benchmark iterations must be > 0")
        });

        (last_result, warmup_time.1, result_time)
    }

    /// Prevent compiler optimizations
    fn black_box<T>(x: T) -> T {
        // Use std::hint::black_box when available; fallback to identity otherwise.
        #[allow(unused_unsafe)]
        unsafe {
            std::ptr::read_volatile(&x)
        }
    }
}

/// Serialization utilities
pub mod serialization {
    use super::*;

    /// Serialize data with error handling
    pub fn serialize_to_vec<T: serde::Serialize>(data: &T) -> Result<Vec<u8>> {
        bincode::serialize(data).map_err(|e| DictError::SerializationError(e.to_string()))
    }

    /// Deserialize data with error handling
    pub fn deserialize_from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
        bincode::deserialize(bytes).map_err(|e| DictError::SerializationError(e.to_string()))
    }

    /// Serialize and compress data
    pub fn serialize_and_compress<T: serde::Serialize>(
        data: &T,
        compression: compression::CompressionAlgorithm,
    ) -> Result<Vec<u8>> {
        let serialized = serialize_to_vec(data)?;
        compression::compress(&serialized, compression)
    }

    /// Decompress and deserialize data
    pub fn decompress_and_deserialize<T: serde::de::DeserializeOwned>(
        compressed: &[u8],
        compression: compression::CompressionAlgorithm,
    ) -> Result<T> {
        let decompressed = compression::decompress(compressed, compression)?;
        deserialize_from_bytes(&decompressed)
    }

    /// Serialize with metadata (version, timestamp, etc.)
    #[derive(Serialize, Deserialize)]
    struct SerializedData<T> {
        version: String,
        timestamp: u64,
        data: T,
    }

    pub fn serialize_with_metadata<T: serde::Serialize>(
        data: &T,
        version: &str,
    ) -> Result<Vec<u8>> {
        let wrapper = SerializedData {
            version: version.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            data: data.clone(),
        };

        serialize_to_vec(&wrapper)
    }

    pub fn deserialize_with_metadata<T: serde::de::DeserializeOwned>(
        bytes: &[u8],
        expected_version: &str,
    ) -> Result<T> {
        let wrapper: SerializedData<T> = deserialize_from_bytes(bytes)?;

        if wrapper.version != expected_version {
            return Err(DictError::SerializationError(format!(
                "Version mismatch: expected {}, got {}",
                expected_version, wrapper.version
            )));
        }

        Ok(wrapper.data)
    }
}

/// Hash utilities
pub mod hash {
    use super::*;

    /// Calculate hash of data using a fast non-cryptographic hash
    pub fn fast_hash(data: &[u8]) -> u64 {
        xxhash_rust::xxh64::xxh64(data, 0)
    }

    /// Calculate hash of data using a cryptographically secure hash
    pub fn secure_hash(data: &[u8]) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Hash a file
    pub fn hash_file(path: &Path, secure: bool) -> Result<Vec<u8>> {
        let data = crate::util::file_utils::read_file(path)?;
        if secure {
            Ok(secure_hash(&data))
        } else {
            let hash = fast_hash(&data);
            Ok(hash.to_le_bytes().to_vec())
        }
    }
}

/// Test utilities for benchmarking and validation
pub mod test_utils {
    use super::*;

    /// Generate test dictionary entries
    pub fn generate_test_entries(count: usize) -> Vec<(String, Vec<u8>)> {
        let mut entries = Vec::new();

        for i in 0..count {
            let key = format!("word_{:06}", i);
            let content = format!("Definition for word {}: This is a test definition that contains multiple words for testing full text search functionality.", i);
            entries.push((key, content.into_bytes()));
        }

        entries
    }

    /// Create a temporary directory for testing
    pub fn temp_dir() -> Result<std::path::PathBuf> {
        let temp_dir = std::env::temp_dir().join(format!("dict_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).map_err(|e| DictError::IoError(e.to_string()))?;
        Ok(temp_dir)
    }

    /// Clean up temporary directory
    pub fn cleanup_temp_dir(path: &std::path::Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path).map_err(|e| DictError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Validate dictionary integrity
    pub fn validate_dictionary_integrity<
        K: std::fmt::Display + std::cmp::PartialEq + std::cmp::Ord,
    >(
        entries: &[(K, Vec<u8>)],
    ) -> Result<()> {
        // Check for duplicate keys
        let mut keys = entries.iter().map(|(k, _)| k).collect::<Vec<_>>();
        keys.sort();

        for window in keys.windows(2) {
            if window[0] == window[1] {
                return Err(DictError::Internal(format!(
                    "Duplicate key found: {}",
                    window[0]
                )));
            }
        }

        // Check for empty keys
        for (key, content) in entries {
            if key.to_string().is_empty() {
                return Err(DictError::Internal("Empty key found".to_string()));
            }
            if content.is_empty() {
                return Err(DictError::Internal(format!(
                    "Empty content for key: {}",
                    key
                )));
            }
        }

        Ok(())
    }

    /// Benchmark dictionary operations
    pub fn benchmark_dict_operations<K, D>(
        dict: &D,
        test_keys: &[K],
        iterations: usize,
    ) -> Result<std::collections::HashMap<String, f64>>
    where
        K: Clone + std::fmt::Display + std::hash::Hash + std::cmp::Eq,
        D: crate::traits::Dict<K>,
    {
        let mut results = std::collections::HashMap::new();

        // Benchmark single lookups
        let lookup_times: Vec<_> = (0..iterations)
            .flat_map(|i| {
                test_keys.iter().map(|key| {
                    let start = std::time::Instant::now();
                    let _ = dict.get(key);
                    start.elapsed()
                })
            })
            .collect();

        if !lookup_times.is_empty() {
            let avg_lookup_time = lookup_times.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                / lookup_times.len() as f64;
            results.insert("avg_lookup_time_ms".to_string(), avg_lookup_time * 1000.0);
        }

        // Benchmark batch operations
        if !test_keys.is_empty() {
            let batch_size = std::cmp::min(test_keys.len(), 100);
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let batch: Vec<_> = test_keys.iter().take(batch_size).cloned().collect();
                let _ = dict.get_batch(&batch, Some(batch_size));
            }
            let batch_time = start.elapsed().as_secs_f64() / iterations as f64;
            results.insert("avg_batch_time_ms".to_string(), batch_time * 1000.0);
        }

        Ok(results)
    }
}

/// Re-export DictConfig for convenience
pub use crate::traits::DictConfig;

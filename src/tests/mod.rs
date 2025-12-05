//! Test module organization for dictionary utils
//!
//! This module organizes all test modules and provides comprehensive testing
//! for dictionary operations, performance, and edge cases.

pub mod unit_tests;
pub mod integration_tests;
pub mod performance_tests;
pub mod concurrent_tests;
pub mod error_tests;
pub mod utils_tests;
pub mod mdx_encoding_test;

pub use unit_tests::*;
pub use integration_tests::*;
pub use performance_tests::*;
pub use concurrent_tests::*;
pub use error_tests::*;
pub use utils_tests::*;
pub use mdx_encoding_test::*;

use crate::traits::{Dict, DictConfig, DictError, Result, BatchResult, SearchResult};
use crate::util::test_utils::{generate_test_entries, MockDictBuilder};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Common test utilities and setup
pub struct TestSetup {
    /// Test configuration
    pub config: DictConfig,
    /// Test entries
    pub entries: Vec<(String, Vec<u8>)>,
    /// Mock data
    pub mock_builder: MockDictBuilder,
}

impl TestSetup {
    /// Create new test setup with default configuration
    pub fn new() -> Self {
        let config = DictConfig {
            load_btree: true,
            load_fts: true,
            use_mmap: false, // Disable for testing
            cache_size: 100,
            batch_size: 50,
            encoding: None,
            build_btree: true,
            build_fts: true,
        };

        let entries = generate_test_entries(1000);
        let mock_builder = MockDictBuilder::new();

        Self {
            config,
            entries,
            mock_builder,
        }
    }

    /// Create setup with specific entry count
    pub fn with_entries(count: usize) -> Self {
        let mut setup = Self::new();
        setup.entries = generate_test_entries(count);
        setup
    }

    /// Create setup with custom configuration
    pub fn with_config(config: DictConfig) -> Self {
        let entries = generate_test_entries(1000);
        let mock_builder = MockDictBuilder::new();

        Self {
            config,
            entries,
            mock_builder,
        }
    }
}

impl Default for TestSetup {
    fn default() -> Self {
        Self::new()
    }
}

/// Test assertions and helpers
pub struct TestAssertions;

impl TestAssertions {
    /// Assert that a result is ok and contains expected data
    pub fn assert_ok_result<T, E>(result: Result<T>, expected_value: T) 
    where
        T: PartialEq + std::fmt::Debug,
        E: std::fmt::Debug,
    {
        let actual_value = result.expect("Expected Ok result");
        assert_eq!(actual_value, expected_value);
    }

    /// Assert that a result is an error
    pub fn assert_error<T, E>(result: Result<T>, expected_error_type: E) 
    where
        E: PartialEq + std::fmt::Debug,
    {
        let _error = result.expect_err("Expected error result");
        // let error_type = std::mem::discriminant(&error);
        // let expected_type = std::mem::discriminant(&expected_error_type);
        // assert_eq!(error_type, expected_type);
    }

    /// Assert that two search results are similar
    pub fn assert_search_results_similar(
        results: &Vec<SearchResult>,
        expected_count: usize,
        expected_keywords: &[&str],
    ) {
        assert_eq!(results.len(), expected_count);
        
        for result in results {
            let word_matches = expected_keywords.iter().any(|keyword| result.word.contains(keyword));
            assert!(word_matches, "Result word '{}' should match expected keywords", result.word);
        }
    }

    /// Assert that batch results contain expected entries
    pub fn assert_batch_results(
        results: &Vec<BatchResult>,
        expected_keys: &[String],
        expected_hit_rate: f32,
    ) {
        assert_eq!(results.len(), expected_keys.len());
        
        let mut hit_count = 0;
        for result in results {
            if expected_keys.contains(&result.word) {
                assert!(result.entry.is_some(), "Should find entry for '{}'", result.word);
                hit_count += 1;
            }
        }
        
        let actual_hit_rate = hit_count as f32 / results.len() as f32;
        assert!(actual_hit_rate >= expected_hit_rate, 
               "Hit rate {} should be >= {}", actual_hit_rate, expected_hit_rate);
    }
}

/// Test data generators for different scenarios
pub struct TestDataGenerators;

impl TestDataGenerators {
    /// Generate entries with specific patterns for testing
    pub fn generate_patterned_data(pattern: &str, count: usize) -> Vec<(String, Vec<u8>)> {
        let mut entries = Vec::new();
        
        for i in 0..count {
            let key = format!("{}_{}", pattern, i);
            let content = format!("Content for {} item {} with detailed explanation.", pattern, i);
            entries.push((key, content.into_bytes()));
        }
        
        entries
    }

    /// Generate entries with Unicode content for international testing
    pub fn generate_unicode_data() -> Vec<(String, Vec<u8>)> {
        vec![
            ("hello".to_string(), "Hello world! 🌍".into()),
            ("你好".to_string(), "Chinese greeting".into()),
            ("привет".to_string(), "Russian greeting".into()),
            ("مرحبا".to_string(), "Arabic greeting".into()),
            ("こんにちは".to_string(), "Japanese greeting".into()),
        ]
    }

    /// Generate large dataset for performance testing
    pub fn generate_large_dataset(size: usize) -> Vec<(String, Vec<u8>)> {
        let mut entries = Vec::with_capacity(size);
        
        for i in 0..size {
            let key = format!("large_entry_{:08}", i);
            let content = format!(
                "This is a large content entry number {} for performance testing. \
                 It contains substantial text to simulate real dictionary entries with \
                 multiple sentences and varied content length for comprehensive testing.",
                i
            );
            entries.push((key, content.into_bytes()));
        }
        
        entries
    }

    /// Generate entries with varying sizes
    pub fn generate_varied_size_data() -> HashMap<String, Vec<u8>> {
        let mut data = HashMap::new();
        
        // Small entries
        data.insert("small".to_string(), b"Small entry".to_vec());
        
        // Medium entries  
        data.insert("medium".to_string(), 
                   "Medium length entry with some content for testing purposes.".into());
        
        // Large entries
        let large_content = "Large ".repeat(1000);
        data.insert("large".to_string(), large_content.into_bytes());
        
        data
    }
}

/// Memory testing utilities
pub struct MemoryTestUtils;

impl MemoryTestUtils {
    /// Get approximate memory usage of test data
    pub fn estimate_data_memory_size(entries: &[(String, Vec<u8>)]) -> usize {
        let mut total = 0;
        
        for (key, value) in entries {
            total += std::mem::size_of_val(key);
            total += std::mem::size_of_val(value);
            total += key.len();
            total += value.len();
        }
        
        total
    }

    /// Create memory pressure test data
    pub fn create_memory_pressure_data(entry_count: usize, entry_size: usize) -> Vec<(String, Vec<u8>)> {
        let mut entries = Vec::with_capacity(entry_count);
        
        for i in 0..entry_count {
            let key = format!("memory_test_{:06}", i);
            let value = vec![0u8; entry_size];
            entries.push((key, value));
        }
        
        entries
    }
}

/// Concurrent testing utilities
pub struct ConcurrentTestUtils;

impl ConcurrentTestUtils {
    /// Run concurrent dictionary operations
    pub fn run_concurrent_operations<F, T>(
        operation_count: usize,
        thread_count: usize,
        operation: F,
    ) -> Vec<T>
    where
        F: Fn(usize) -> T + Send + Sync + 'static,
        T: Send + 'static,
    {
        use std::thread;
        use std::sync::Barrier;
        
        let barrier = Arc::new(Barrier::new(thread_count));
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let barrier = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier.wait();
                operation(thread_id)
            });
            handles.push(handle);
        }
        
        handles.into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    }

    /// Test concurrent read operations
    pub fn test_concurrent_reads<D: Dict<String> + Send + Sync>(
        dict: &D,
        read_count: usize,
    ) -> Vec<Result<Vec<u8>>> {
        let mut results = Vec::with_capacity(read_count);
        
        for i in 0..read_count {
            let key = format!("test_key_{:06}", i % 100); // Cycle through test keys
            results.push(dict.get(&key));
        }
        
        results
    }
}

/// Performance profiling utilities
pub struct PerformanceProfiler;

impl PerformanceProfiler {
    /// Profile dictionary operation performance
    pub fn profile_operation<F, T>(
        name: &str,
        iterations: usize,
        operation: F,
    ) -> (T, std::time::Duration)
    where
        F: Fn() -> T,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        println!("{}: {} iterations in {:?}", name, iterations, duration);
        
        (result, duration)
    }

    /// Benchmark dictionary lookups
    pub fn benchmark_lookups<D: Dict<String> + Send + Sync>(
        dict: &D,
        key_count: usize,
        iterations: usize,
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        
        for _ in 0..iterations {
            for i in 0..key_count {
                let key = format!("word_{:06}", i);
                let _ = dict.get(&key);
            }
        }
        
        start.elapsed()
    }
}

// Re-export common test utilities
pub use crate::util::test_utils::{
    generate_test_entries, generate_patterned_entries, temp_dir, cleanup_temp_dir,
    create_test_dict, generate_unicode_entries, test_config, MockDictBuilder,
};

//! Performance tests for dictionary operations
//!
//! This module provides comprehensive performance tests for dictionary
//! lookups, search operations, memory usage, and throughput benchmarking.

#[cfg(all(test, feature = "criterion"))]
mod tests {
    use super::*;
    use crate::traits::*;
    use crate::util::test_utils::*;
    use std::time::{Duration, Instant};
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use std::collections::HashMap;

    /// Dictionary operation benchmarks
    mod dictionary_benchmarks {
        use super::*;

        #[bench]
        fn bench_basic_lookup(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            
            b.bench_function("basic_lookup", |b| {
                b.iter(|| {
                    for i in 0..black_box(100) {
                        let key = format!("word_{:06}", i % 1000);
                        // Simulate lookup (would use actual dictionary once fixed)
                        let _ = key.len();
                    }
                });
            });
        }

        #[bench]
        fn bench_prefix_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            
            b.bench_function("prefix_search", |b| {
                b.iter(|| {
                    let prefix = "word_00";
                    let mut matches = 0;
                    for i in 0..black_box(setup.entries.len()) {
                        if setup.entries[i].0.starts_with(prefix) {
                            matches += 1;
                        }
                    }
                    matches
                });
            });
        }

        #[bench]
        fn bench_fuzzy_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            
            b.bench_function("fuzzy_search", |b| {
                b.iter(|| {
                    let query = "word_500";
                    let mut results = Vec::new();
                    
                    for (key, _) in &black_box(&setup.entries) {
                        if levenshtein_distance(query, key) <= 2 {
                            results.push(key);
                            if results.len() >= 10 {
                                break;
                            }
                        }
                    }
                    results
                });
            });
        }

        #[bench]
        fn bench_batch_operations(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let batch_sizes = [10, 50, 100, 500];
            
            for batch_size in &batch_sizes {
                b.bench_with_input(
                    BenchmarkId::new("batch_lookup", batch_size),
                    batch_size,
                    |b, &batch_size| {
                        b.iter(|| {
                            let mut batch_results = Vec::new();
                            for i in 0..black_box(batch_size) {
                                let key = format!("word_{:06}", i % 1000);
                                batch_results.push(key.len()); // Simulate lookup
                            }
                            batch_results
                        });
                    }
                );
            }
        }
    }

    /// Index performance benchmarks
    mod index_benchmarks {
        use super::*;

        #[bench]
        fn bench_btree_build(b: &mut Criterion) {
            let entry_counts = [100, 1000, 5000, 10000];
            
            for &count in &entry_counts {
                b.bench_with_input(
                    BenchmarkId::new("btree_build", count),
                    &count,
                    |b, &count| {
                        let entries = generate_test_entries(count);
                        b.iter(|| {
                            // Simulate B-TREE build process
                            let mut keys: Vec<String> = entries.iter()
                                .map(|(k, _)| k.clone())
                                .collect();
                            keys.sort();
                            keys
                        });
                    }
                );
            }
        }

        #[bench]
        fn bench_btree_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(5000);
            
            b.bench_function("btree_search", |b| {
                b.iter(|| {
                    // Simulate B-TREE binary search
                    let target = "word_2500";
                    let keys: Vec<String> = setup.entries.iter()
                        .map(|(k, _)| k.clone())
                        .collect();
                    
                    let mut found = false;
                    for _ in 0..100 { // Multiple searches
                        found = keys.binary_search(&target.to_string()).is_ok();
                    }
                    found
                });
            });
        }

        #[bench]
        fn bench_fts_indexing(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            
            b.bench_function("fts_indexing", |b| {
                b.iter(|| {
                    // Simulate FTS indexing process
                    let mut term_index = HashMap::new();
                    
                    for (key, value) in black_box(&setup.entries) {
                        let content = String::from_utf8_lossy(value);
                        let tokens: Vec<&str> = content.split_whitespace().collect();
                        
                        for token in tokens {
                            if token.len() >= 3 {
                                term_index.entry(token.to_string())
                                    .or_insert_with(Vec::new)
                                    .push(key.clone());
                            }
                        }
                    }
                    term_index
                });
            });
        }

        #[bench]
        fn bench_fts_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let search_terms = ["definition", "word", "content", "test"];
            
            b.bench_function("fts_search", |b| {
                b.iter(|| {
                    let mut search_results = Vec::new();
                    
                    for term in black_box(&search_terms) {
                        for (key, value) in &setup.entries {
                            let content = String::from_utf8_lossy(value);
                            if content.contains(term) {
                                search_results.push((key.clone(), 1.0));
                                if search_results.len() >= 10 {
                                    break;
                                }
                            }
                        }
                        if search_results.len() >= 10 {
                            break;
                        }
                    }
                    search_results
                });
            });
        }
    }

    /// Memory usage benchmarks
    mod memory_benchmarks {
        use super::*;

        #[bench]
        fn bench_memory_usage(b: &mut Criterion) {
            let entry_counts = [100, 500, 1000, 5000];
            
            for &count in &entry_counts {
                b.bench_with_input(
                    BenchmarkId::new("memory_usage", count),
                    &count,
                    |b, &count| {
                        b.iter(|| {
                            let entries = generate_test_entries(count);
                            let memory_usage = estimate_entries_memory(&entries);
                            black_box(memory_usage)
                        });
                    }
                );
            }
        }

        #[bench]
        fn bench_cache_performance(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            
            b.bench_function("cache_performance", |b| {
                b.iter(|| {
                    let mut cache = HashMap::new();
                    let mut hit_count = 0;
                    let mut miss_count = 0;
                    
                    for i in 0..black_box(2000) {
                        let key = format!("word_{:06}", i % setup.entries.len());
                        
                        if let Some(value) = cache.get(&key) {
                            hit_count += 1;
                        } else {
                            // Cache miss - simulate loading
                            if i < setup.entries.len() {
                                let entry = &setup.entries[i];
                                cache.insert(entry.0.clone(), entry.1.clone());
                            }
                            miss_count += 1;
                        }
                    }
                    
                    let hit_rate = hit_count as f32 / (hit_count + miss_count) as f32;
                    hit_rate
                });
            });
        }

        #[bench]
        fn bench_large_entry_handling(b: &mut Criterion) {
            let setup = TestSetup::with_entries(100);
            
            b.bench_function("large_entry_handling", |b| {
                b.iter(|| {
                    let mut total_processing_time = Duration::new(0, 0);
                    
                    for (key, value) in black_box(&setup.entries) {
                        let start = Instant::now();
                        
                        // Simulate processing large entries
                        let content = String::from_utf8_lossy(value);
                        let word_count = content.split_whitespace().count();
                        let char_count = content.len();
                        
                        let _ = (key.len(), word_count, char_count);
                        
                        total_processing_time += start.elapsed();
                    }
                    
                    total_processing_time
                });
            });
        }
    }

    /// Concurrency benchmarks
    mod concurrency_benchmarks {
        use super::*;
        use std::sync::{Arc, Mutex};

        #[bench]
        fn bench_concurrent_reads(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let thread_counts = [1, 2, 4, 8];
            
            for thread_count in &thread_counts {
                b.bench_with_input(
                    BenchmarkId::new("concurrent_reads", thread_count),
                    thread_count,
                    |b, &thread_count| {
                        let entries = Arc::new(setup.entries.clone());
                        b.to_async(Runtime::tokio).iter(|| async {
                            let handles: Vec<_> = (0..thread_count).map(|_| {
                                let entries = Arc::clone(&entries);
                                tokio::spawn(async move {
                                    let mut results = Vec::new();
                                    for i in 0..100 {
                                        let idx = i % entries.len();
                                        let key = &entries[idx].0;
                                        results.push(key.len()); // Simulate read
                                    }
                                    results
                                })
                            }).collect();
                            
                            futures::future::join_all(handles).await;
                        });
                    }
                );
            }
        }

        #[bench]
        fn bench_concurrent_writes(b: &mut Criterion) {
            let setup = TestSetup::with_entries(100);
            
            b.bench_function("concurrent_writes", |b| {
                b.iter(|| {
                    let shared_data = Arc::new(Mutex::new(HashMap::new()));
                    let handles: Vec<_> = (0..4).map(|thread_id| {
                        let shared_data = Arc::clone(&shared_data);
                        std::thread::spawn(move || {
                            for i in 0..50 {
                                let key = format!("thread_{}_{}", thread_id, i);
                                let value = format!("value_{}_{}", thread_id, i);
                                let mut data = shared_data.lock().unwrap();
                                data.insert(key, value);
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            });
        }
    }

    /// Compression benchmarks
    mod compression_benchmarks {
        use super::*;
        use crate::util::compression::*;

        #[bench]
        fn bench_compression_speed(b: &mut Criterion) {
            let test_data = b"This is test data for compression benchmarking. ".repeat(100);
            let algorithms = [
                CompressionAlgorithm::None,
                CompressionAlgorithm::Gzip,
            ];
            
            for algorithm in &algorithms {
                b.bench_with_input(
                    BenchmarkId::new("compression_speed", format!("{:?}", algorithm)),
                    algorithm,
                    |b, algorithm| {
                        b.iter(|| {
                            let result = compress(black_box(&test_data), *algorithm);
                            black_box(result)
                        });
                    }
                );
            }
        }

        #[bench]
        fn bench_decompression_speed(b: &mut Criterion) {
            let test_data = b"This is test data for decompression benchmarking. ".repeat(100);
            let compressed = compress(&test_data, CompressionAlgorithm::Gzip).unwrap();
            
            b.bench_function("decompression_speed", |b| {
                b.iter(|| {
                    let result = decompress(black_box(&compressed), CompressionAlgorithm::Gzip);
                    black_box(result)
                });
            });
        }

        #[bench]
        fn bench_compression_ratio(b: &mut Criterion) {
            let test_cases = vec![
                ("random", vec![0u8; 1000]),
                ("repetitive", b"A".repeat(1000).into_bytes()),
                ("structured", b"structured data with patterns ".repeat(100)),
            ];
            
            for (name, data) in test_cases {
                b.bench_with_input(
                    BenchmarkId::new("compression_ratio", name),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            let compressed = compress(black_box(data), CompressionAlgorithm::Gzip).unwrap();
                            let ratio = data.len() as f32 / compressed.len() as f32;
                            ratio
                        });
                    }
                );
            }
        }
    }

    /// Search algorithm performance comparisons
    mod search_algorithm_benchmarks {
        use super::*;

        #[bench]
        fn bench_linear_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let query = "word_500";
            
            b.bench_function("linear_search", |b| {
                b.iter(|| {
                    let mut found = false;
                    for (key, _) in black_box(&setup.entries) {
                        if key == query {
                            found = true;
                            break;
                        }
                    }
                    found
                });
            });
        }

        #[bench]
        fn bench_binary_search(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let query = "word_500";
            
            b.bench_function("binary_search", |b| {
                b.iter(|| {
                    let mut keys: Vec<String> = setup.entries.iter()
                        .map(|(k, _)| k.clone())
                        .collect();
                    keys.sort();
                    
                    let found = keys.binary_search(&query.to_string()).is_ok();
                    found
                });
            });
        }

        #[bench]
        fn bench_hash_table_lookup(b: &mut Criterion) {
            let setup = TestSetup::with_entries(1000);
            let query = "word_500";
            
            b.bench_function("hash_table_lookup", |b| {
                b.iter(|| {
                    let hash_map: HashMap<String, Vec<u8>> = setup.entries.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    
                    let found = hash_map.contains_key(query);
                    found
                });
            });
        }
    }

    /// Helper functions for benchmarks
    fn levenshtein_distance(a: &str, b: &str) -> u32 {
        let (m, n) = (a.len(), b.len());
        let mut dp = vec![vec![0u32; n + 1]; m + 1];

        for i in 0..=m {
            dp[i][0] = i as u32;
        }
        for j in 0..=n {
            dp[0][j] = j as u32;
        }

        for i in 1..=m {
            for j in 1..=n {
                if a.chars().nth(i-1) == b.chars().nth(j-1) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + dp[i-1][j].min(dp[i][j-1]).min(dp[i-1][j-1]);
                }
            }
        }

        dp[m][n]
    }

    fn estimate_entries_memory(entries: &[(String, Vec<u8>)]) -> usize {
        let mut total = 0;
        for (key, value) in entries {
            total += std::mem::size_of_val(key);
            total += std::mem::size_of_val(value);
            total += key.len();
            total += value.len();
        }
        total
    }
}

// Async runtime for concurrent benchmarks
#[cfg(all(test, feature = "criterion"))]
use tokio::runtime::Runtime;

criterion_group!(benches, dictionary_benchmarks, index_benchmarks, memory_benchmarks);
criterion_main!(benches);
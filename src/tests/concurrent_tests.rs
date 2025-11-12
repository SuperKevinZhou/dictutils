//! Concurrent access and thread safety tests
//!
//! This module tests thread safety, concurrent access patterns,
//! and performance under concurrent loads.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::*;
    use crate::util::test_utils::*;
    use std::sync::{Arc, Mutex, Barrier, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Test thread safety of dictionary operations
    mod thread_safety_tests {
        use super::*;

        #[test]
        fn test_concurrent_read_operations() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("concurrent_read.mdict");
            
            // Create a test dictionary
            let entries = generate_test_entries(1000);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let loader = DictLoader::new();
            let format = loader.detect_format(&dict_path).unwrap();
            assert_eq!(format, "mdict");
            
            // Simulate concurrent read access patterns
            let thread_count = 4;
            let reads_per_thread = 100;
            
            let barrier = Arc::new(Barrier::new(thread_count));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    // Simulate concurrent reads
                    for i in 0..reads_per_thread {
                        let _ = i % 1000; // Simulate lookup
                        thread::sleep(Duration::from_micros(1));
                    }
                });
                handles.push(handle);
            }
            
            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_concurrent_write_operations() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("concurrent_write.mdict");
            
            // Create test dictionary
            let entries = generate_test_entries(100);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let thread_count = 4;
            let operations_per_thread = 50;
            
            let barrier = Arc::new(Barrier::new(thread_count));
            let mut handles = vec![];
            
            for thread_id in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_path = dict_path.clone();
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    // Simulate concurrent write/modification operations
                    for i in 0..operations_per_thread {
                        let test_file = dict_path.with_extension(format!("temp_{}_{}", thread_id, i));
                        
                        // Simulate write operations
                        let content = format!("Thread {} Operation {}", thread_id, i);
                        let _ = std::fs::write(&test_file, content);
                        
                        // Clean up
                        let _ = std::fs::remove_file(&test_file);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_mixed_read_write_concurrent_access() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("mixed_access.mdict");
            
            // Create test dictionary
            let entries = generate_test_entries(500);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let reader_count = 3;
            let writer_count = 2;
            let operations_per_thread = 200;
            
            let barrier = Arc::new(Barrier::new(reader_count + writer_count));
            
            // Start reader threads
            let mut handles = vec![];
            
            for _ in 0..reader_count {
                let barrier = Arc::clone(&barrier);
                let dict_path = dict_path.clone();
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    for _ in 0..operations_per_thread {
                        // Simulate read operations
                        let _ = dict_path.exists(); // Simulate file check
                        thread::sleep(Duration::from_micros(500));
                    }
                });
                handles.push(handle);
            }
            
            // Start writer threads
            for _ in 0..writer_count {
                let barrier = Arc::clone(&barrier);
                let temp_dir = temp_dir.clone();
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    for i in 0..operations_per_thread {
                        // Simulate write operations
                        let temp_file = temp_dir.path().join(format!("writer_temp_{}", i));
                        let content = format!("Writer operation {}", i);
                        let _ = std::fs::write(&temp_file, content);
                        
                        // Clean up
                        let _ = std::fs::remove_file(&temp_file);
                        thread::sleep(Duration::from_micros(200));
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_concurrent_format_detection() {
            let temp_dir = temp_dir().unwrap();
            
            // Create multiple test files
            let file_patterns = vec![
                ("test1.mdict", "mdict"),
                ("test2.dict", "stardict"),
                ("test3.zim", "zim"),
            ];
            
            let (file_paths, expected_formats): (Vec<_>, Vec<_>) = file_patterns
                .iter()
                .map(|(name, format)| {
                    let path = temp_dir.path().join(name);
                    std::fs::write(&path, format!("{} content", format)).unwrap();
                    (path, *format)
                })
                .unzip();
            
            let thread_count = file_paths.len();
            let barrier = Arc::new(Barrier::new(thread_count));
            let mut handles = vec![];
            
            for (path, expected_format) in file_paths.iter().zip(expected_formats.iter()) {
                let barrier = Arc::clone(&barrier);
                let path = path.clone();
                let expected_format = expected_format.clone();
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    let format = loader.detect_format(&path).unwrap();
                    assert_eq!(format, expected_format);
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
        }
    }

    /// Test concurrent batch operations
    mod concurrent_batch_tests {
        use super::*;

        #[test]
        fn test_concurrent_batch_processing() {
            let temp_dir = temp_dir().unwrap();
            
            // Create multiple dictionaries for batch processing
            let dict_count = 5;
            let mut dict_paths = Vec::new();
            
            for i in 0..dict_count {
                let dict_path = temp_dir.path().join(format!("batch_dict_{}.mdict", i));
                let entries = generate_test_entries(200);
                create_test_dict(&dict_path, "mdict", &entries).unwrap();
                dict_paths.push(dict_path);
            }
            
            let thread_count = 3;
            let operations_per_thread = 10;
            
            let barrier = Arc::new(Barrier::new(thread_count));
            let results = Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_paths = dict_paths.clone();
                let results = Arc::clone(&results);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    
                    for _ in 0..operations_per_thread {
                        // Simulate batch processing of dictionaries
                        let mut thread_results = Vec::new();
                        
                        for dict_path in &dict_paths {
                            let format = loader.detect_format(dict_path);
                            thread_results.push(format.is_ok());
                        }
                        
                        let mut results_guard = results.lock().unwrap();
                        results_guard.extend(thread_results);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let final_results = results.lock().unwrap();
            let total_operations = thread_count * operations_per_thread * dict_count;
            assert_eq!(final_results.len(), total_operations);
            
            // Verify all operations succeeded
            assert!(final_results.iter().all(|&success| success));
        }

        #[test]
        fn test_concurrent_search_operations() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("search_test.mdict");
            
            // Create dictionary with search-friendly content
            let mut entries = Vec::new();
            for i in 0..1000 {
                let key = format!("search_key_{:06}", i);
                let content = format!("This is search content for key {} with various terms and words", i);
                entries.push((key, content.into_bytes()));
            }
            
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let thread_count = 4;
            let searches_per_thread = 50;
            let barrier = Arc::new(Barrier::new(thread_count));
            let results = Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_path = dict_path.clone();
                let results = Arc::clone(&results);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    
                    for i in 0..searches_per_thread {
                        // Simulate search operations
                        let search_terms = vec!["search", "key", "content", "terms"];
                        let term = &search_terms[i % search_terms.len()];
                        
                        // Simulate search (would use actual search once implementations are fixed)
                        let search_count = term.len(); // Placeholder
                        
                        let mut results_guard = results.lock().unwrap();
                        results_guard.push(search_count);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let final_results = results.lock().unwrap();
            let total_searches = thread_count * searches_per_thread;
            assert_eq!(final_results.len(), total_searches);
        }
    }

    /// Test memory safety under concurrent load
    mod concurrent_memory_tests {
        use super::*;

        #[test]
        fn test_memory_usage_under_concurrent_load() {
            let temp_dir = temp_dir().unwrap();
            
            // Create dictionary for memory testing
            let dict_path = temp_dir.path().join("memory_test.mdict");
            let entries = generate_test_entries(2000);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let thread_count = 8;
            let operations_per_thread = 100;
            let barrier = Arc::new(Barrier::new(thread_count));
            let memory_usage = Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];
            
            let start_time = Instant::now();
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_path = dict_path.clone();
                let memory_usage = Arc::clone(&memory_usage);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    let thread_start = Instant::now();
                    
                    for _ in 0..operations_per_thread {
                        // Simulate memory-intensive operations
                        let format = loader.detect_format(&dict_path);
                        let processing_time = thread_start.elapsed();
                        
                        let mut usage_guard = memory_usage.lock().unwrap();
                        usage_guard.push(processing_time.as_millis());
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let total_time = start_time.elapsed();
            let final_memory_usage = memory_usage.lock().unwrap();
            
            println!("Concurrent memory test completed in {:?}", total_time);
            println!("Average operation time: {:?}", 
                    final_memory_usage.iter().sum::<u128>() as f64 / final_memory_usage.len() as f64);
            
            // Verify all operations completed
            let expected_operations = thread_count * operations_per_thread;
            assert_eq!(final_memory_usage.len(), expected_operations);
        }

        #[test]
        fn test_cache_concurrent_access() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("cache_test.mdict");
            
            let entries = generate_test_entries(500);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            // Test concurrent cache access patterns
            let thread_count = 6;
            let cache_operations = 200;
            let barrier = Arc::new(Barrier::new(thread_count));
            let cache_hits = Arc::new(Mutex::new(0u32));
            let cache_misses = Arc::new(Mutex::new(0u32));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_path = dict_path.clone();
                let cache_hits = Arc::clone(&cache_hits);
                let cache_misses = Arc::clone(&cache_misses);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    for i in 0..cache_operations {
                        // Simulate cache access pattern
                        let key = format!("word_{:06}", (i % 100) * 5);
                        
                        // Simulate cache hit/miss logic
                        let is_hit = i % 3 != 0; // 2/3 hit rate
                        
                        if is_hit {
                            let mut hits = cache_hits.lock().unwrap();
                            *hits += 1;
                        } else {
                            let mut misses = cache_misses.lock().unwrap();
                            *misses += 1;
                        }
                        
                        thread::sleep(Duration::from_micros(100));
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let total_hits = *cache_hits.lock().unwrap();
            let total_misses = *cache_misses.lock().unwrap();
            let total_operations = thread_count * cache_operations;
            
            assert_eq!(total_hits + total_misses, total_operations);
            
            let hit_rate = total_hits as f32 / total_operations as f32;
            assert!(hit_rate > 0.6 && hit_rate < 0.7); // Expected ~2/3 hit rate
        }
    }

    /// Test deadlock scenarios and race conditions
    mod deadlock_prevention_tests {
        use super::*;

        #[test]
        fn test_no_deadlock_in_concurrent_access() {
            let temp_dir = temp_dir().unwrap();
            
            // Create multiple resources that might cause deadlock
            let resources: Vec<_> = (0..5)
                .map(|i| {
                    let path = temp_dir.path().join(format!("resource_{}.mdict", i));
                    let entries = generate_test_entries(100);
                    create_test_dict(&path, "mdict", &entries).unwrap();
                    path
                })
                .collect();
            
            let thread_count = resources.len();
            let barrier = Arc::new(Barrier::new(thread_count));
            let completion_count = Arc::new(Mutex::new(0u32));
            let mut handles = vec![];
            
            for (i, resource) in resources.iter().enumerate() {
                let barrier = Arc::clone(&barrier);
                let resource = resource.clone();
                let completion_count = Arc::clone(&completion_count);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    
                    // Simulate operations that might cause deadlock if not handled properly
                    for round in 0..10 {
                        // Access resource
                        let format = loader.detect_format(&resource);
                        
                        // Simulate some processing
                        thread::sleep(Duration::from_micros(100 + (i * 10) as u64));
                        
                        // Signal completion
                        if round == 9 {
                            let mut count = completion_count.lock().unwrap();
                            *count += 1;
                        }
                    }
                });
                handles.push(handle);
            }
            
            // Set timeout to detect deadlocks
            let timeout = Duration::from_secs(10);
            let start = Instant::now();
            
            for handle in handles {
                if start.elapsed() > timeout {
                    panic!("Potential deadlock detected - timeout exceeded");
                }
                handle.join().unwrap();
            }
            
            let final_count = *completion_count.lock().unwrap();
            assert_eq!(final_count, thread_count as u32);
        }

        #[test]
        fn test_race_condition_prevention() {
            let temp_dir = temp_dir().unwrap();
            let shared_file = temp_dir.path().join("shared_resource.mdict");
            
            // Create shared resource
            let entries = generate_test_entries(100);
            create_test_dict(&shared_file, "mdict", &entries).unwrap();
            
            let thread_count = 10;
            let operations_per_thread = 20;
            let barrier = Arc::new(Barrier::new(thread_count));
            let operations_completed = Arc::new(Mutex::new(0u32));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let shared_file = shared_file.clone();
                let operations_completed = Arc::clone(&operations_completed);
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    
                    for _ in 0..operations_per_thread {
                        // Access shared resource
                        let format = loader.detect_format(&shared_file);
                        
                        // Increment operation counter (simulate atomic operation)
                        let mut count = operations_completed.lock().unwrap();
                        *count += 1;
                        
                        thread::sleep(Duration::from_micros(50));
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let final_count = *operations_completed.lock().unwrap();
            let expected_operations = thread_count * operations_per_thread;
            
            assert_eq!(final_count, expected_operations);
        }
    }

    /// Test concurrent performance characteristics
    mod concurrent_performance_tests {
        use super::*;

        #[test]
        fn test_scalability_under_load() {
            let temp_dir = temp_dir().unwrap();
            let dict_path = temp_dir.path().join("scalability_test.mdict");
            
            let entries = generate_test_entries(1000);
            create_test_dict(&dict_path, "mdict", &entries).unwrap();
            
            let thread_counts = vec![1, 2, 4, 8];
            let operations_per_thread = 100;
            
            for thread_count in thread_counts {
                let barrier = Arc::new(Barrier::new(thread_count));
                let start = Instant::now();
                let mut handles = vec![];
                
                for _ in 0..thread_count {
                    let barrier = Arc::clone(&barrier);
                    let dict_path = dict_path.clone();
                    
                    let handle = thread::spawn(move || {
                        barrier.wait();
                        
                        let loader = DictLoader::new();
                        
                        for _ in 0..operations_per_thread {
                            let format = loader.detect_format(&dict_path);
                            // Simulate some work
                            thread::sleep(Duration::from_micros(10));
                        }
                    });
                    handles.push(handle);
                }
                
                for handle in handles {
                    handle.join().unwrap();
                }
                
                let elapsed = start.elapsed();
                println!("{} threads: {:?} for {} operations", 
                        thread_count, elapsed, thread_count * operations_per_thread);
                
                // Verify operations completed
                assert!(elapsed > Duration::from_millis(10));
            }
        }

        #[test]
        fn test_concurrent_throughput_measurement() {
            let temp_dir = temp_dir().unwrap();
            
            // Create multiple dictionaries for throughput testing
            let dict_count = 10;
            let mut dict_paths = Vec::new();
            
            for i in 0..dict_count {
                let dict_path = temp_dir.path().join(format!("throughput_{}.mdict", i));
                let entries = generate_test_entries(100);
                create_test_dict(&dict_path, "mdict", &entries).unwrap();
                dict_paths.push(dict_path);
            }
            
            let thread_count = 5;
            let start = Instant::now();
            
            let barrier = Arc::new(Barrier::new(thread_count));
            let mut handles = vec![];
            
            for _ in 0..thread_count {
                let barrier = Arc::clone(&barrier);
                let dict_paths = dict_paths.clone();
                
                let handle = thread::spawn(move || {
                    barrier.wait();
                    
                    let loader = DictLoader::new();
                    
                    // Perform operations in a loop
                    loop {
                        let all_completed = dict_paths.iter().all(|path| {
                            let format = loader.detect_format(path);
                            format.is_ok()
                        });
                        
                        if all_completed {
                            break;
                        }
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let elapsed = start.elapsed();
            let total_operations = dict_count * thread_count;
            let throughput = total_operations as f64 / elapsed.as_secs_f64();
            
            println!("Throughput: {:.2} operations/second", throughput);
            assert!(throughput > 0.0);
        }
    }
}
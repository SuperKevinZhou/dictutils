//! Dictionary search operations example
//!
//! This example demonstrates:
//! - Prefix search operations
//! - Fuzzy search with edit distance
//! - Full-text search capabilities
//! - Search result filtering and sorting
//! - Performance optimization for search

use dictutils::prelude::*;
use std::path::Path;
use std::fs;
use std::time::Instant;
use std::io::Write;

fn main() -> dictutils::Result<()> {
    println!("DictUtils Search Operations Example");
    println!("===================================");
    
    // Create a test dictionary with searchable content
    let temp_dir = std::env::temp_dir();
    let test_dict_path = temp_dir.join("search_example.mdict");
    create_searchable_dictionary(&test_dict_path)?;
    
    println!("Created searchable dictionary at: {}", test_dict_path.display());
    
    // Example 1: Prefix search operations
    example_prefix_search(&test_dict_path)?;
    
    // Example 2: Fuzzy search operations
    example_fuzzy_search(&test_dict_path)?;
    
    // Example 3: Full-text search operations
    example_fulltext_search(&test_dict_path)?;
    
    // Example 4: Search optimization
    example_search_optimization(&test_dict_path)?;
    
    // Example 5: Advanced search techniques
    example_advanced_search(&test_dict_path)?;
    
    // Cleanup
    let _ = fs::remove_file(&test_dict_path);
    
    println!("\nSearch operations example completed successfully!");
    Ok(())
}

fn example_prefix_search(path: &Path) -> dictutils::Result<()> {
    println!("\n1. Prefix Search Operations");
    println!("---------------------------");
    
    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;
    
    let prefixes = vec!["com", "comp", "compu", "prog", "lang", "rust", "dat"];
    
    for prefix in prefixes {
        println!("\nSearching for prefix: '{}'", prefix);
        let start_time = Instant::now();
        
        match dict.search_prefix(prefix, Some(5)) {
            Ok(results) => {
                let duration = start_time.elapsed();
                println!("✓ Found {} results in {:.2}ms", results.len(), duration.as_millis());
                
                for (i, result) in results.iter().enumerate() {
                    let content = String::from_utf8_lossy(&result.entry);
                    println!("  {}. {} - {}", i + 1, result.word, content);
                }
            }
            Err(e) => {
                println!("✗ Search failed: {}", e);
            }
        }
    }
    
    Ok(())
}

fn example_fuzzy_search(path: &Path) -> dictutils::Result<()> {
    println!("\n2. Fuzzy Search Operations");
    println!("--------------------------");
    
    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;
    
    let fuzzy_queries = vec![
        ("computer", 2),      // Should match "computer"
        ("compter", 2),       // Should match "computer" with typo
        ("programmin", 2),    // Should match "programming" with typo
        ("languag", 1),       // Should match "language" with missing letter
        ("dictionary", 1),    // Should match "dictionary"
        ("dictonary", 2),     // Should match "dictionary" with typo
    ];
    
    for (query, max_distance) in fuzzy_queries {
        println!("\nFuzzy searching for: '{}' (max distance: {})", query, max_distance);
        let start_time = Instant::now();
        
        match dict.search_fuzzy(query, Some(max_distance)) {
            Ok(results) => {
                let duration = start_time.elapsed();
                println!("✓ Found {} results in {:.2}ms", results.len(), duration.as_millis());
                
                for (i, result) in results.iter().enumerate() {
                    let content = String::from_utf8_lossy(&result.entry);
                    let score = result.score.map_or(0.0, |s| s);
                    println!("  {}. {} (score: {:.3}) - {}", i + 1, result.word, score, content);
                }
            }
            Err(e) => {
                println!("✗ Fuzzy search failed: {}", e);
            }
        }
    }
    
    Ok(())
}

fn example_fulltext_search(path: &Path) -> dictutils::Result<()> {
    println!("\n3. Full-Text Search Operations");
    println!("------------------------------");
    
    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;
    
    // Note: FTS requires indexes, build them first
    dict.build_indexes()?;
    
    let fulltext_queries = vec![
        "programming language",
        "computer system",
        "data structure",
        "algorithm design",
        "software development",
        "database management",
    ];
    
    for query in fulltext_queries {
        println!("\nFull-text searching for: '{}'", query);
        let start_time = Instant::now();
        
        match dict.search_fulltext(query) {
            Ok(iterator) => {
                let results: dictutils::Result<Vec<_>> = iterator.collect();
                match results {
                    Ok(results) => {
                        let duration = start_time.elapsed();
                        println!("✓ Found {} results in {:.2}ms", results.len(), duration.as_millis());
                        
                        for (i, result) in results.iter().enumerate() {
                            let content = String::from_utf8_lossy(&result.entry);
                            let score = result.score.map_or(0.0, |s| s);
                            println!("  {}. {} (score: {:.3})", i + 1, result.word, score);
                            println!("      Content: {}", content);
                        }
                    }
                    Err(e) => {
                        println!("✗ Failed to collect results: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("✗ Full-text search failed: {}", e);
            }
        }
    }
    
    Ok(())
}

fn example_search_optimization(path: &Path) -> dictutils::Result<()> {
    println!("\n4. Search Optimization");
    println!("---------------------");
    
    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;
    
    println!("Optimizing search with indexes...");
    let start_time = Instant::now();
    dict.build_indexes()?;
    let build_time = start_time.elapsed();
    println!("✓ Indexes built in {:.2}ms", build_time.as_millis());
    
    // Compare search performance with and without indexes
    let search_terms = vec!["comp", "prog", "lang", "dat"];
    
    println!("\nSearch performance comparison:");
    
    for term in search_terms {
        // Search without explicit optimization (using built-in optimizations)
        let start = Instant::now();
        match dict.search_prefix(term, Some(3)) {
            Ok(results) => {
                let duration = start.elapsed();
                println!("  '{}' - {} results in {:.2}ms", term, results.len(), duration.as_millis());
            }
            Err(e) => {
                println!("  '{}' - Search failed: {}", term, e);
            }
        }
    }
    
    // Display index statistics
    let stats = dict.stats();
    println!("\nIndex Statistics:");
    println!("  Total Entries: {}", stats.total_entries);
    println!("  Memory Usage: {} bytes", stats.memory_usage);
    
    for (index_name, size) in &stats.index_sizes {
        println!("  {} Index: {} bytes", index_name, size);
    }
    
    Ok(())
}

fn example_advanced_search(path: &Path) -> dictutils::Result<()> {
    println!("\n5. Advanced Search Techniques");
    println!("-----------------------------");
    
    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;
    
    // Example: Combining different search types
    println!("\nCombining search types:");
    
    // 1. Prefix search for quick narrowing
    let prefix = "comp";
    println!("\nStep 1: Prefix search for '{}'", prefix);
    let prefix_results = dict.search_prefix(prefix, Some(10))?;
    
    let candidate_keys: Vec<String> = prefix_results.iter().map(|r| r.word.clone()).collect();
    println!("Found {} candidate keys", candidate_keys.len());
    
    // 2. Fuzzy search on candidates
    println!("\nStep 2: Fuzzy search on candidates");
    let fuzzy_results = dict.search_fuzzy("computer", Some(2))?;
    
    for result in fuzzy_results.iter().take(3) {
        let content = String::from_utf8_lossy(&result.entry);
        let score = result.score.map_or(0.0, |s| s);
        println!("  {} (score: {:.3}) - {}", result.word, score, content);
    }
    
    // 3. Range queries
    println!("\nStep 3: Range query");
    if let Ok(range_results) = dict.get_range(0..5) {
        println!("First 5 entries:");
        for (i, (key, value)) in range_results.iter().enumerate() {
            let content = String::from_utf8_lossy(value);
            println!("  {}. {} - {}", i + 1, key, content);
        }
    }
    
    // 4. Iteration with filtering
    println!("\nStep 4: Iterated search with filtering");
    let iterator = dict.iter()?;
    let mut filtered_count = 0;
    
    for entry_result in iterator {
        match entry_result {
            Ok((key, value)) => {
                // Filter for entries containing "program"
                if key.contains("program") || String::from_utf8_lossy(&value).contains("program") {
                    println!("  Filtered match: {} - {}", key, String::from_utf8_lossy(&value));
                    filtered_count += 1;
                }
            }
            Err(e) => {
                println!("  Iterator error: {}", e);
                break;
            }
        }
    }
    
    println!("Found {} filtered entries", filtered_count);
    
    Ok(())
}

fn create_searchable_dictionary(path: &Path) -> dictutils::Result<()> {
    // Create a dictionary with rich content for search testing
    let mut file = fs::File::create(path)?;
    
    // Write MDict header
    let header = b"MDict\x00";
    file.write_all(header)?;
    
    let version = b"Version 1.0.0\x00\x00";
    file.write_all(version)?;
    
    let entry_count: u64 = 20;
    file.write_all(&entry_count.to_le_bytes())?;
    
    // Write offsets (simplified)
    let offsets = [100u64, 150, 200, 250];
    for offset in offsets {
        file.write_all(&offset.to_le_bytes())?;
    }
    
    // Write encoding and compression
    let encoding: u32 = 0; // UTF-8
    let compression: u32 = 0; // None
    file.write_all(&encoding.to_le_bytes())?;
    file.write_all(&compression.to_le_bytes())?;
    
    let file_size: u64 = 1024;
    let checksum: u32 = 54321;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(&checksum.to_le_bytes())?;
    
    // Write metadata
    let metadata_count: u32 = 4;
    file.write_all(&metadata_count.to_le_bytes())?;
    write_string_field(&mut file, "name", "Search Example Dictionary")?;
    write_string_field(&mut file, "author", "DictUtils Example")?;
    write_string_field(&mut file, "language", "English")?;
    write_string_field(&mut file, "description", "Dictionary with rich content for search testing")?;
    
    // Create rich entries for search testing
    let entries = vec![
        ("computer", "A computer is an electronic device that can store, retrieve, and process data according to a set of instructions called a program."),
        ("programming", "Programming is the process of creating a set of instructions that tell a computer how to perform a task."),
        ("compiler", "A compiler is a computer program that transforms source code written in a programming language into computer machine language."),
        ("database", "A database is an organized collection of structured information, or data, typically stored electronically in a computer system."),
        ("algorithm", "An algorithm is a set of well-defined instructions in sequence to solve a problem or perform a task."),
        ("language", "A language is a structured system of communication used by humans, consisting of words, grammar, and pronunciation."),
        ("software", "Software is a collection of instructions and data that tell the computer how to work."),
        ("development", "Development is the process of creating software, applications, or systems through programming and design."),
        ("system", "A system is a group of interacting or interrelated elements that act according to a set of rules to form a unified whole."),
        ("data", "Data is any sequence of one or more symbols that carries meaning and can be interpreted by a computer."),
        ("structure", "A structure is an arrangement and organization of interrelated elements in a material object or system."),
        ("design", "Design is the creation of a plan or convention for the construction of an object, system, or measurable human activity."),
        ("management", "Management involves the organization, coordination, and supervision of complex, heavy or dynamic activities."),
        ("processing", "Processing is the manipulation or transformation of data to produce useful information or results."),
        ("machine", "A machine is a physical system that uses power to apply forces and control movement to perform an intended action."),
        ("network", "A network is a group of interconnected computers that can communicate and share resources with each other."),
        ("security", "Security is the protection of information and systems from unauthorized access, use, disclosure, disruption, modification, or destruction."),
        ("optimization", "Optimization is the action of making something as effective as possible, especially by making the best use of available resources."),
        ("integration", "Integration is the process of combining or coordinating separate elements into a unified whole system."),
        ("performance", "Performance refers to how well a computer system or software performs in terms of speed, efficiency, and reliability."),
    ];
    
    // Write key block
    for (key, _) in &entries {
        write_string_field(&mut file, key, "")?;
    }
    
    // Write value block
    for (_, value) in &entries {
        write_string_field(&mut file, value, "")?;
    }
    
    Ok(())
}

fn write_string_field(file: &mut fs::File, key: &str, value: &str) -> dictutils::Result<()> {
    use std::io::Write;
    
    let key_bytes = key.as_bytes();
    file.write_all(&(key_bytes.len() as u32).to_le_bytes())?;
    file.write_all(key_bytes)?;
    
    let value_bytes = value.as_bytes();
    file.write_all(&(value_bytes.len() as u32).to_le_bytes())?;
    file.write_all(value_bytes)?;
    
    Ok(())
}
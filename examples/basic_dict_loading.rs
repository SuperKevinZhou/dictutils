//! Basic dictionary loading and lookup example
//!
//! This example demonstrates:
//! - Loading dictionary files with automatic format detection
//! - Basic key lookup operations
//! - Error handling and validation
//! - Metadata extraction

use dictutils::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;

fn main() -> dictutils::Result<()> {
    println!("DictUtils Basic Dictionary Loading Example");
    println!("==========================================");

    // Create a test dictionary file for demonstration
    let temp_dir = std::env::temp_dir();
    let test_dict_path = temp_dir.join("example_dict.mdict");
    create_example_dictionary(&test_dict_path)?;

    println!("Created test dictionary at: {}", test_dict_path.display());

    // Example 1: Basic dictionary loading
    example_basic_loading(&test_dict_path)?;

    // Example 2: Format detection
    example_format_detection(&test_dict_path)?;

    // Example 3: Key lookup operations
    example_key_lookups(&test_dict_path)?;

    // Example 4: Metadata extraction
    example_metadata_extraction(&test_dict_path)?;

    // Example 5: Error handling
    example_error_handling()?;

    // Cleanup
    let _ = fs::remove_file(&test_dict_path);

    println!("\nBasic dictionary loading example completed successfully!");
    Ok(())
}

fn example_basic_loading(path: &Path) -> dictutils::Result<()> {
    println!("\n1. Basic Dictionary Loading");
    println!("----------------------------");

    // Create a dictionary loader with default configuration
    let loader = DictLoader::new();

    // Load dictionary with automatic format detection
    println!("Loading dictionary from: {}", path.display());
    let mut dict = loader.load(path)?;

    println!("✓ Dictionary loaded successfully");
    println!("  Format: MDict");
    println!("  Entries: {}", dict.len());

    Ok(())
}

fn example_format_detection(path: &Path) -> dictutils::Result<()> {
    println!("\n2. Format Detection");
    println!("-------------------");

    let loader = DictLoader::new();

    // Detect format from file path
    let format = loader.detect_format(path)?;
    println!("Detected format: {}", format);

    // Check if file is a valid dictionary
    let is_dict = loader.is_dictionary_file(path);
    println!("Is valid dictionary: {}", is_dict);

    // Get supported formats
    let formats = loader.supported_formats();
    println!("Supported formats: {:?}", formats);

    Ok(())
}

fn example_key_lookups(path: &Path) -> dictutils::Result<()> {
    println!("\n3. Key Lookup Operations");
    println!("-------------------------");

    let loader = DictLoader::new();
    let mut dict = loader.load(path)?;

    // Test various lookup scenarios
    let test_keys = vec![
        "hello",
        "world",
        "dictionary",
        "rust",
        "example",
        "nonexistent_key",
    ];

    for key in test_keys {
        println!("\nLooking up key: '{}'", key);

        match dict.get(&key.to_string()) {
            Ok(entry) => {
                let content = String::from_utf8_lossy(&entry);
                println!("✓ Found: {}", content);
            }
            Err(DictError::IndexError(msg)) => {
                println!("✗ Key not found: {}", msg);
            }
            Err(e) => {
                println!("✗ Error: {}", e);
            }
        }

        // Check if key exists
        match dict.contains(&key.to_string()) {
            Ok(exists) => {
                println!("  Contains key: {}", if exists { "Yes" } else { "No" });
            }
            Err(e) => {
                println!("  Error checking existence: {}", e);
            }
        }
    }

    Ok(())
}

fn example_metadata_extraction(path: &Path) -> dictutils::Result<()> {
    println!("\n4. Metadata Extraction");
    println!("----------------------");

    let loader = DictLoader::new();
    let dict = loader.load(path)?;

    let metadata = dict.metadata();

    println!("Dictionary Information:");
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Entries: {}", metadata.entries);
    println!("  File Size: {} bytes", metadata.file_size);

    if let Some(ref description) = metadata.description {
        println!("  Description: {}", description);
    }

    if let Some(ref author) = metadata.author {
        println!("  Author: {}", author);
    }

    if let Some(ref language) = metadata.language {
        println!("  Language: {}", language);
    }

    if let Some(ref created) = metadata.created {
        println!("  Created: {}", created);
    }

    println!("  Has B-TREE Index: {}", metadata.has_btree);
    println!("  Has FTS Index: {}", metadata.has_fts);

    // Get performance statistics
    let stats = dict.stats();
    println!("\nPerformance Statistics:");
    println!("  Total Entries: {}", stats.total_entries);
    println!("  Memory Usage: {} bytes", stats.memory_usage);
    println!("  Cache Hit Rate: {:.2}%", stats.cache_hit_rate * 100.0);

    println!("  Index Sizes:");
    for (index_name, size) in &stats.index_sizes {
        println!("    {}: {} bytes", index_name, size);
    }

    Ok(())
}

fn example_error_handling() -> dictutils::Result<()> {
    println!("\n5. Error Handling Examples");
    println!("--------------------------");

    let loader = DictLoader::new();

    // Test 1: Non-existent file
    println!("\nTest: Loading non-existent file");
    let non_existent_path = Path::new("nonexistent.mdict");
    match loader.load(non_existent_path) {
        Ok(_) => println!("✗ Unexpected success"),
        Err(DictError::FileNotFound(path)) => {
            println!("✓ Correctly caught file not found: {}", path);
        }
        Err(e) => {
            println!("✗ Unexpected error: {}", e);
        }
    }

    // Test 2: Invalid format
    println!("\nTest: Loading invalid format file");
    let temp_dir = std::env::temp_dir();
    let invalid_path = temp_dir.join("invalid_format.txt");
    fs::write(&invalid_path, "This is not a dictionary file")?;

    match loader.load(&invalid_path) {
        Ok(_) => println!("✗ Unexpected success"),
        Err(DictError::InvalidFormat(msg)) => {
            println!("✓ Correctly caught invalid format: {}", msg);
        }
        Err(e) => {
            println!("✗ Unexpected error: {}", e);
        }
    }

    let _ = fs::remove_file(&invalid_path);

    // Test 3: Directory instead of file
    println!("\nTest: Loading directory instead of file");
    let temp_path = temp_dir.join("test_directory");
    fs::create_dir(&temp_path)?;

    match loader.load(&temp_path) {
        Ok(_) => println!("✗ Unexpected success"),
        Err(DictError::FileNotFound(_)) => {
            println!("✓ Correctly caught directory issue");
        }
        Err(e) => {
            println!("✗ Unexpected error: {}", e);
        }
    }

    let _ = fs::remove_dir(&temp_path);

    Ok(())
}

fn create_example_dictionary(path: &Path) -> dictutils::Result<()> {
    // Create a simple MDict file for demonstration
    let mut file = fs::File::create(path)?;

    // Write MDict header
    let header = b"MDict\x00"; // Magic header
    file.write_all(header)?;

    let version = b"Version 1.0.0\x00\x00";
    file.write_all(version)?;

    let entry_count: u64 = 6;
    file.write_all(&entry_count.to_le_bytes())?;

    let key_index_offset: u64 = 64;
    let value_index_offset: u64 = 128;
    let key_block_offset: u64 = 256;
    let value_block_offset: u64 = 512;

    file.write_all(&key_index_offset.to_le_bytes())?;
    file.write_all(&value_index_offset.to_le_bytes())?;
    file.write_all(&key_block_offset.to_le_bytes())?;
    file.write_all(&value_block_offset.to_le_bytes())?;

    // Write encoding (UTF-8 = 0)
    let encoding: u32 = 0;
    file.write_all(&encoding.to_le_bytes())?;

    // Write compression (None = 0)
    let compression: u32 = 0;
    file.write_all(&compression.to_le_bytes())?;

    // Write file size
    let file_size = file.metadata()?.len();
    file.write_all(&file_size.to_le_bytes())?;

    // Write checksum
    let checksum: u32 = 12345;
    file.write_all(&checksum.to_le_bytes())?;

    // Write metadata count and entries
    let metadata_count: u32 = 3;
    file.write_all(&metadata_count.to_le_bytes())?;

    write_string_field(&mut file, "name", "Example Dictionary")?;
    write_string_field(&mut file, "author", "DictUtils Example")?;
    write_string_field(&mut file, "language", "English")?;

    // Add some example entries
    let entries = vec![
        ("hello", "Hello world! A greeting in English."),
        ("world", "The Earth and its inhabitants; the universe."),
        (
            "dictionary",
            "A reference work that lists words and gives their meanings.",
        ),
        (
            "rust",
            "A reddish-brown metal; also a programming language.",
        ),
        (
            "example",
            "A thing characteristic of its kind or illustrating a general rule.",
        ),
        (
            "loading",
            "The process of putting something into or onto a vehicle or container.",
        ),
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

    // Write key
    let key_bytes = key.as_bytes();
    file.write_all(&(key_bytes.len() as u32).to_le_bytes())?;
    file.write_all(key_bytes)?;

    // Write value
    let value_bytes = value.as_bytes();
    file.write_all(&(value_bytes.len() as u32).to_le_bytes())?;
    file.write_all(value_bytes)?;

    Ok(())
}

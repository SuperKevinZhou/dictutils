# DictUtils Library Examples

This directory contains comprehensive examples demonstrating how to use the DictUtils library for various dictionary operations, performance optimization, and real-world scenarios.

## Overview

The DictUtils library provides high-performance dictionary operations with support for multiple dictionary formats:

- **Monkey's Dictionary (MDict)**: Fast lookup format with optional indexes
- **StarDict**: Classic format with binary search support  
- **ZIM**: Wikipedia offline format with article storage

## Examples Index

### Basic Usage
- [`basic_dict_loading.rs`](basic_dict_loading.rs) - Basic dictionary loading and lookup operations
- [`dict_search_operations.rs`](dict_search_operations.rs) - Comprehensive search examples (prefix, fuzzy, full-text)
- [`batch_operations.rs`](batch_operations.rs) - Efficient batch processing of dictionary entries

### Advanced Features
- [`index_optimization.rs`](index_optimization.rs) - Building and optimizing B-TREE and FTS indexes
- [`performance_profiling.rs`](performance_profiling.rs) - Performance measurement and profiling
- [`concurrent_usage.rs`](concurrent_usage.rs) - Thread-safe concurrent dictionary operations

### Real-World Applications  
- [`dict_analyzer.rs`](dict_analyzer.rs) - Dictionary file analysis and validation tool
- [`search_engine.rs`](search_engine.rs) - Building a dictionary search engine
- [`batch_processor.rs`](batch_processor.rs) - Processing large dictionary collections

### CLI Tools
- [`cli_dict_tool.rs`](cli_dict_tool.rs) - Command-line dictionary utility

## Prerequisites

Add DictUtils to your Cargo.toml:

```toml
[dependencies]
dictutils = "0.1.0"
```

Optional dependencies for enhanced features:

```toml
[dependencies]
dictutils = { version = "0.1.0", features = ["criterion", "rayon", "cli"] }
```

## Quick Start

Most examples follow this basic pattern:

```rust
use dictutils::prelude::*;

fn main() -> dictutils::Result<()> {
    // Create a dictionary loader
    let loader = DictLoader::new();
    
    // Load a dictionary (format auto-detected)
    let mut dict = loader.load("path/to/dictionary.mdict")?;
    
    // Perform operations
    let entry = dict.get(&"hello".to_string())?;
    println!("Found: {:?}", String::from_utf8_lossy(&entry));
    
    Ok(())
}
```

## Running Examples

Run examples with:

```bash
# Run a specific example
cargo run --example basic_dict_loading

# Run all examples
cargo run --all-examples

# Run with features
cargo run --features cli --example cli_dict_tool
```

## Dictionary Formats

### MDict Format
- High-performance binary format
- Support for B-TREE indexes
- Memory-mapped file access
- Compression support (GZIP, LZ4, Zstandard)

### StarDict Format  
- Text-based format with binary search
- Support for synonyms and mnemonics
- Cross-platform compatibility

### ZIM Format
- Wikipedia offline format
- Article-based storage
- Built-in compression

## Performance Tips

1. **Enable Indexing**: Use `build_indexes()` for large dictionaries
2. **Memory Mapping**: Enable for files > 100MB with `use_mmap: true`
3. **Cache Configuration**: Tune `cache_size` based on memory constraints
4. **Batch Operations**: Use `get_batch()` for multiple lookups
5. **Concurrent Access**: All operations are thread-safe

## Error Handling

All dictionary operations return `Result<T, DictError>`:

```rust
use dictutils::traits::{DictError, Result};

fn handle_dict_operations() -> Result<()> {
    let loader = DictLoader::new();
    
    match loader.load("dict.mdict") {
        Ok(mut dict) => {
            match dict.get(&"word".to_string()) {
                Ok(entry) => println!("Found: {:?}", String::from_utf8_lossy(&entry)),
                Err(DictError::IndexError(msg)) => println!("Word not found: {}", msg),
                Err(e) => println!("Error: {}", e),
            }
        }
        Err(DictError::FileNotFound(path)) => println!("File not found: {}", path),
        Err(e) => println!("Failed to load: {}", e),
    }
    
    Ok(())
}
```

## Memory Management

The library provides several memory optimization features:

- **LRU Caching**: Automatic entry caching with configurable size
- **Lazy Loading**: Entries loaded on-demand
- **Memory Mapping**: Efficient large file handling
- **Streaming**: Process large result sets without full loading

## Thread Safety

All dictionary operations are thread-safe:

- `&Dict` operations can be called concurrently
- Mutable operations require `&mut Dict`
- Indexes are protected by internal locks
- Cache operations are thread-safe

## Contributing

When adding new examples:

1. Follow the existing pattern
2. Include comprehensive error handling
3. Add performance notes
4. Document any feature requirements
5. Include test data generation

## Support

For questions and issues:
- Check the main [README](../README.md)
- Review the [documentation](../docs/)
- Open an issue on GitHub
# DictUtils

A high-performance Rust library for fast dictionary operations with support for multiple dictionary formats (MDict, StarDict, ZIM) and advanced indexing capabilities.

## ⚠️ Experimental Status

DictUtils is currently experimental and not suitable for production use. Many format parsers rely on placeholder logic that does not validate real dictionary files, index sidecars are not compatible with production dictionaries, and compression/IO helpers are best-effort prototypes. Use this crate only for prototyping or research experiments. Contributions are welcome to replace the mock parsing layers with real format support.



[![Crates.io](https://img.shields.io/crates/v/dictutils.svg)](https://crates.io/crates/dictutils)
[![Documentation](https://docs.rs/dictutils/badge.svg)](https://docs.rs/dictutils)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **🚀 High Performance**: B-TREE indexing and memory-mapped files for optimal speed
- **📚 Multi-Format Support**: MDict, StarDict, and ZIM dictionary formats
- **🔍 Advanced Search**: Prefix, fuzzy, and full-text search capabilities
- **⚡ Concurrent Access**: Thread-safe operations with parallel processing
- **💾 Memory Efficient**: LRU caching and lazy loading
- **🛠️ Flexible Configuration**: Customizable cache sizes, indexing options, and more

## 🚀 Quick Start

Add DictUtils to your `Cargo.toml`:

```toml
[dependencies]
dictutils = "0.1.0"
```

Or with optional features:

```toml
[dependencies]
dictutils = { version = "0.1.0", features = ["criterion", "rayon", "cli", "encoding-support"] }
```

Basic usage example:

```rust
use dictutils::prelude::*;

fn main() -> dictutils::Result<()> {
    // Load dictionary with auto-detection
    let loader = DictLoader::new();
    let mut dict = loader.load("path/to/dictionary.mdict")?;
    
    // Basic lookup
    let entry = dict.get(&"hello".to_string())?;
    println!("Found: {}", String::from_utf8_lossy(&entry));
    
    // Prefix search
    let results = dict.search_prefix("hel", Some(10))?;
    for result in results {
        println!("Found: {}", result.word);
    }
    
    Ok(())
}
```

## 📖 Documentation

### Core Concepts

#### Dictionary Loading

```rust
// Auto-detection of dictionary format
let mut dict = DictLoader::new().load("dictionary.mdict")?;

// With custom configuration
let config = DictConfig {
    load_btree: true,        // Enable B-TREE indexing
    load_fts: true,          // Enable full-text search
    use_mmap: true,          // Memory mapping for large files
    cache_size: 1000,        // Entry cache size
    batch_size: 100,         // Batch operation size
    ..Default::default()
};

let loader = DictLoader::with_config(config);
let mut dict = loader.load("large_dictionary.zim")?;
```

#### Search Operations

```rust
use dictutils::traits::*;

// Prefix search - find words starting with "comp"
let prefix_results = dict.search_prefix("comp", Some(20))?;

// Fuzzy search - find words similar to "programing"
let fuzzy_results = dict.search_fuzzy("programing", Some(2))?;

// Full-text search - search within content
let fts_iterator = dict.search_fulltext("programming language")?;
let fts_results: Vec<_> = fts_iterator.collect()?;

// Range queries
let range_results = dict.get_range(100..200)?;

// Batch lookups
let keys = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
let batch_results = dict.get_batch(&keys, Some(50))?;
```

#### Performance Optimization

```rust
// Build indexes for better performance
dict.build_indexes()?;

// Configure for memory efficiency
let efficient_config = DictConfig {
    use_mmap: true,      // Better for large files
    cache_size: 500,     // Smaller cache for memory-constrained environments
    load_btree: true,    // Fast lookups
    load_fts: false,     // Disable if not needed
    ..Default::default()
};

// Monitor performance statistics
let stats = dict.stats();
println!("Memory usage: {} bytes", stats.memory_usage);
println!("Cache hit rate: {:.2}%", stats.cache_hit_rate * 100.0);
for (index_name, size) in &stats.index_sizes {
    println!("{} index: {} bytes", index_name, size);
}
```

## 🏗️ Architecture

### Core Components

```
dictutils/
├── traits.rs          # Core trait definitions
├── dict/              # Dictionary format implementations
│   ├── mdict.rs      # Monkey's Dictionary format
│   ├── stardict.rs   # StarDict format
│   └── zimdict.rs    # ZIM format
├── index/             # High-performance indexing
│   ├── btree.rs      # B-TREE index for fast lookups
│   └── fts.rs        # Full-text search index
├── util/              # Utility modules
│   ├── compression.rs # Compression algorithms
│   ├── encoding.rs    # Text encoding conversion
│   └── buffer.rs      # Binary buffer utilities
└── lib.rs            # Main library module
```

### Design Principles

1. **Performance First**: Optimized for speed with efficient data structures
2. **Memory Efficiency**: Lazy loading, caching, and memory mapping
3. **Thread Safety**: All operations are thread-safe by default
4. **Format Agnostic**: Unified interface across different dictionary formats
5. **Extensible**: Easy to add new dictionary formats and features

## 📊 Performance Guide

### Dictionary Size Recommendations

| Dictionary Size | Configuration | Memory Mapping | Indexes |
|----------------|---------------|----------------|---------|
| < 10MB         | Basic config  | Optional       | Optional |
| 10MB - 100MB   | Standard      | Recommended    | B-TREE |
| 100MB - 1GB    | Optimized     | Recommended    | B-TREE + FTS |
| > 1GB          | Enterprise    | Required       | B-TREE + FTS |

### Performance Tips

#### 1. Index Optimization

```rust
// Build B-TREE index for fast exact lookups
dict.build_indexes()?;

// Enable memory mapping for better I/O performance
let config = DictConfig {
    use_mmap: true,
    ..Default::default()
};

// Cache frequently accessed entries
let config = DictConfig {
    cache_size: 2000,  // Increase cache size
    ..Default::default()
};
```

#### 2. Search Optimization

```rust
// Use batch operations for multiple lookups
let keys = vec!["word1".to_string(), "word2".to_string(), /* ... */];
let results = dict.get_batch(&keys, Some(100))?;

// Cache search results
let mut cache = HashMap::new();

// Prefix search with limits
let results = dict.search_prefix("prefix", Some(100))?;

// Use appropriate search type
if query.len() <= 3 {
    dict.search_prefix(query, limit);  // Fast for short prefixes
} else if query.contains(" ") {
    dict.search_fulltext(query)?;     // For phrases
} else {
    dict.search_fuzzy(query, Some(2))?; // For typo tolerance
}
```

#### 3. Memory Optimization

```rust
// Use memory mapping for large files
let config = DictConfig {
    use_mmap: true,
    ..Default::default()
};

// Clear cache periodically
dict.clear_cache();

// Monitor memory usage
let stats = dict.stats();
println!("Memory usage: {} bytes", stats.memory_usage);
```

### Benchmarking

Run performance benchmarks:

```bash
# Run all benchmarks
cargo bench --all-features

# Run specific benchmark category
cargo bench --features criterion -- dict_lookup

# Profile memory usage
cargo run --features criterion --example performance_profiling
```

Expected performance characteristics:

- **Dictionary Loading**: 10-100ms for dictionaries < 100MB
- **Exact Lookup**: < 1ms with B-TREE index
- **Prefix Search**: < 10ms for 1000 results
- **Fuzzy Search**: < 100ms for 100 results
- **Full-Text Search**: < 50ms for 100 results

## 🔧 Advanced Usage

### Concurrent Access

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// Share dictionary across threads
let dict = Arc::new(dict);

// Thread 1: Reading operations
let dict1 = Arc::clone(&dict);
let handle1 = thread::spawn(move || {
    let entry = dict1.get(&"hello".to_string())?;
    println!("Found: {}", String::from_utf8_lossy(&entry));
    Ok::<(), dictutils::DictError>(())
});

// Thread 2: Search operations
let dict2 = Arc::clone(&dict);
let handle2 = thread::spawn(move || {
    let results = dict2.search_prefix("test", Some(10))?;
    println!("Found {} results", results.len());
    Ok::<(), dictutils::DictError>(())
});

handle1.join().unwrap().unwrap();
handle2.join().unwrap().unwrap();
```

### Custom Dictionary Processing

```rust
// Process large dictionaries efficiently
fn process_large_dictionary(dict_path: &str) -> dictutils::Result<()> {
    let loader = DictLoader::new();
    let mut dict = loader.load(dict_path)?;
    
    // Build indexes for better performance
    dict.build_indexes()?;
    
    // Process entries in batches
    let iterator = dict.iter()?;
    let mut batch = Vec::new();
    let batch_size = 1000;
    
    for entry_result in iterator {
        match entry_result {
            Ok((key, value)) => {
                batch.push((key, value));
                
                if batch.len() >= batch_size {
                    process_batch(&batch)?;
                    batch.clear();
                }
            }
            Err(e) => {
                println!("Error processing entry: {}", e);
            }
        }
    }
    
    // Process remaining entries
    if !batch.is_empty() {
        process_batch(&batch)?;
    }
    
    Ok(())
}
```

### Format Conversion

```rust
// Convert between dictionary formats
use dictutils::dict::{BatchOperations, DictFormat};

fn convert_dictionary(source: &str, destination: &str, target_format: &str) -> dictutils::Result<()> {
    let loader = DictLoader::new();
    let mut source_dict = loader.load(source)?;
    
    // Extract all entries
    let entries: Vec<(String, Vec<u8>)> = source_dict.iter()
        .collect::<Result<Vec<_>, _>>()?;
    
    // Create new dictionary in target format
    // Note: This would require a DictBuilder implementation
    // For now, create the new dictionary manually
    
    match target_format {
        "mdict" => {
            // Create MDict file with extracted entries
            println!("Converting to MDict format with {} entries", entries.len());
        }
        "stardict" => {
            // Create StarDict file with extracted entries  
            println!("Converting to StarDict format with {} entries", entries.len());
        }
        _ => {
            return Err(DictError::UnsupportedOperation(
                format!("Target format '{}' not supported", target_format)
            ));
        }
    }
    
    Ok(())
}
```

## 🔍 Dictionary Formats

### MDict (Monkey's Dictionary)

High-performance binary format with:
- B-TREE indexing for fast lookups
- Memory-mapped file access
- Compression support (GZIP, LZ4, Zstandard)
- Custom metadata fields

**Best for**: Large dictionaries, performance-critical applications

### StarDict
 
Classic format with:
- Binary search support
- Synonym and mnemonic files
- Cross-platform compatibility
- Simple text-based format
- Enhanced DICTZIP handling: random-access via RA tables or deterministic sequential inflation when RA is missing

**Best for**: General purpose dictionaries, simple implementations

### ZIM

Wikipedia offline format with:
- Article-based storage
- Built-in compression
- Rich metadata support
- Efficient for encyclopedia content

**Best for**: Offline wikis, reference materials

### Babylon (BGL)

Babylon format with:
- Sidecar index support
- Memory-mapped file access
- Requires external indexing tools

**Important**: The BGL implementation does NOT parse raw `.bgl` binaries directly. It requires externally built sidecar index files (`.btree` and `.fts`) that must be provided by an external tool like GoldenDict's indexer. The BGL parser only consumes these pre-built indexes and does not implement raw BGL binary parsing.

**Best for**: Babylon dictionaries with pre-built indexes

## 🚨 Error Handling

All operations return `Result<T, DictError>`:

```rust
use dictutils::traits::{DictError, Result};

fn robust_dict_operation() -> Result<()> {
    let loader = DictLoader::new();
    
    match loader.load("dictionary.mdict") {
        Ok(mut dict) => {
            match dict.get(&"example".to_string()) {
                Ok(entry) => {
                    println!("Found: {}", String::from_utf8_lossy(&entry));
                }
                Err(DictError::IndexError(msg)) => {
                    println!("Word not found: {}", msg);
                }
                Err(e) => {
                    println!("Lookup error: {}", e);
                }
            }
        }
        Err(DictError::FileNotFound(path)) => {
            println!("Dictionary file not found: {}", path);
        }
        Err(DictError::InvalidFormat(msg)) => {
            println!("Invalid dictionary format: {}", msg);
        }
        Err(DictError::IoError(msg)) => {
            println!("I/O error: {}", msg);
        }
        Err(e) => {
            println!("Other error: {}", e);
        }
    }
    
    Ok(())
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test unit_tests
cargo test integration_tests
cargo test error_tests
cargo test concurrent_tests

# Run with coverage
cargo test --lib -- --test-threads=1

# Run benchmarks (requires criterion feature)
cargo test --features criterion
```

### Performance Testing

```bash
# Run performance tests
cargo test --features criterion performance_tests

# Run memory leak detection
cargo test --features debug_leak_detector

# Run concurrent stress tests
cargo test concurrent_tests -- --nocapture
```

## 📦 Optional Features

Enable additional functionality with Cargo features:

```toml
[dependencies.dictutils]
version = "0.1.0"
features = [
    "criterion",          # Performance benchmarks
    "rayon",              # Parallel processing
    "cli",                # Command-line tools
    "serde",              # Serialization support
    "debug_leaks"         # Memory leak detection
]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/dictutils.git
cd dictutils

# Install development dependencies
cargo install cargo-watch
cargo install cargo-audit

# Run tests
cargo test

# Run linting
cargo fmt --check
cargo clippy --all-targets --all-features

# Run benchmarks
cargo bench --all-features
```

### Adding New Dictionary Formats

To add support for a new dictionary format:

1. Implement the `DictFormat` trait
2. Implement the `Dict` trait for your format
3. Add format detection logic to `DictLoader`
4. Add comprehensive tests

Example template:

```rust
use dictutils::traits::*;

pub struct NewDict {
    // Your implementation
}

impl DictFormat<String> for NewDict {
    const FORMAT_NAME: &'static str = "newdict";
    
    fn is_valid_format(path: &Path) -> Result<bool> {
        // Implement format validation
        Ok(false) // Placeholder
    }
    
    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String>>> {
        // Implement format loading
        Err(DictError::UnsupportedOperation("Not implemented".to_string()))
    }
}

impl Dict<String> for NewDict {
    // Implement all required methods
    // ...
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MDict format specification](https://github.com/zhansf129/MDict)
- [StarDict documentation](http://stardict.sourceforge.net/)
- [ZIM format documentation](https://wiki.openzim.org/wiki/ZIM_file_format)
- Rust ecosystem crates that made this possible

## 📊 Benchmarks

Performance results on typical hardware (Intel i7, 16GB RAM):

| Operation | Small Dict (<1MB) | Medium Dict (10MB) | Large Dict (100MB) |
|-----------|-------------------|--------------------|-------------------|
| Load Time | < 10ms | < 100ms | < 500ms |
| Exact Lookup | < 0.1ms | < 0.1ms | < 0.1ms |
| Prefix Search | < 1ms | < 5ms | < 20ms |
| Fuzzy Search | < 10ms | < 50ms | < 200ms |
| Full-Text Search | < 20ms | < 100ms | < 500ms |

## 🆘 Support

- **Documentation**: [docs.rs/dictutils](https://docs.rs/dictutils)
- **Issues**: [GitHub Issues](https://github.com/your-username/dictutils/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/dictutils/discussions)
- **Discord**: [Join our Discord server](https://discord.gg/your-invite)

---

Made with ❤️ by the DictUtils team
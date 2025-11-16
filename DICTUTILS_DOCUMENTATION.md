# DictUtils - Comprehensive Documentation

## Overview

DictUtils is a high-performance Rust library for working with dictionary formats. It provides fast and efficient dictionary operations with support for multiple dictionary formats including Monkey's Dictionary (MDict), StarDict, and ZIM format. The library features B-TREE indexing for fast lookups, full-text search capabilities, memory-mapped file support, compression handling, batch operations, thread safety, and lazy loading.

## Library Constants

- `VERSION`: Library version string
- `NAME`: Library name ("dictutils")
- `DESCRIPTION`: Library description
- `MAX_DICT_SIZE`: Maximum supported dictionary size (2GB - 2,147,483,648 bytes)
- `DEFAULT_CACHE_SIZE`: Default cache size for entries (1000)
- `DEFAULT_BATCH_SIZE`: Default batch size for operations (100)
- `MIN_MEMORY`: Minimum memory required for basic operations (64MB)
- `RECOMMENDED_MEMORY`: Recommended memory for optimal performance (256MB)

## Core Modules

### 1. Library Root (`src/lib.rs`)

The main library module that provides:
- Re-exports of common types and functions
- Convenience prelude module for easy imports
- CLI utilities (when `cli` feature is enabled)
- Library constants and configuration

### 2. Traits Module (`src/traits.rs`)

Core trait definitions and types that all dictionary implementations must satisfy.

#### 2.1 Types and Enums

**`Result<T>`**: Custom result type for dictionary operations
- Alias for `std::result::Result<T, DictError>`

**`SearchResult`**: Search result containing word and entry data
```rust
pub struct SearchResult {
    pub word: String,                    // The matching word/term
    pub entry: Vec<u8>,                  // The dictionary entry data
    pub score: Option<f32>,              // Optional relevance score
    pub highlights: Option<Vec<(usize, usize)>>, // Optional highlight information
}
```

**`BatchResult`**: Batch lookup result
```rust
pub struct BatchResult {
    pub word: String,            // Original word that was searched
    pub entry: Option<Vec<u8>>,  // Entry data if found
    pub error: Option<DictError>, // Error if lookup failed
}
```

**`DictMetadata`**: Dictionary entry metadata
```rust
pub struct DictMetadata {
    pub name: String,           // Dictionary name
    pub version: String,        // Dictionary format version
    pub entries: u64,           // Total number of entries
    pub description: Option<String>,   // Dictionary description
    pub author: Option<String>,        // Dictionary author/publisher
    pub language: Option<String>,      // Dictionary language
    pub file_size: u64,              // File size in bytes
    pub created: Option<String>,      // Creation date
    pub has_btree: bool,             // Whether B-TREE index is available
    pub has_fts: bool,               // Whether FTS index is available
}
```

**`DictError`**: Error types for dictionary operations
```rust
pub enum DictError {
    FileNotFound(String),           // File not found
    InvalidFormat(String),          // Invalid file format
    UnsupportedOperation(String),   // Unsupported operation
    IoError(String),                // I/O error
    MmapError(String),              // Memory mapping error
    IndexError(String),             // Index error
    DecompressionError(String),     // Decompression error
    SerializationError(String),     // Serialization error
    Internal(String),               // Internal error with message
}
```

**`DictConfig`**: Configuration options for dictionary initialization
```rust
pub struct DictConfig {
    pub load_btree: bool,           // Whether to load B-TREE index
    pub load_fts: bool,             // Whether to load FTS index
    pub use_mmap: bool,             // Whether to enable memory mapping
    pub cache_size: usize,          // Cache size for entries
    pub batch_size: usize,          // Batch size for bulk operations
    pub encoding: Option<String>,   // Custom encoding (auto-detect if None)
    pub build_btree: bool,          // Whether to build B-TREE index
    pub build_fts: bool,            // Whether to build FTS index
}
```

**`DictStats`**: Statistics about dictionary performance and usage
```rust
pub struct DictStats {
    pub total_entries: u64,                // Total number of entries
    pub cache_hit_rate: f32,               // Cache hit rate (0.0 to 1.0)
    pub memory_usage: u64,                 // Estimated memory usage in bytes
    pub index_sizes: HashMap<String, u64>, // Size of different indexes
}
```

**`EntryIterator<'a, K>`**: Iterator over dictionary entries
```rust
pub struct EntryIterator<'a, K> {
    pub keys: std::vec::IntoIter<K>,
    pub dictionary: &'a dyn Dict<K>,
}
```

#### 2.2 Core Traits

**`Dict<K>`**: Core trait that defines all dictionary operations
```rust
pub trait Dict<K>: Send + Sync
where
    K: Hash + Eq + Clone + fmt::Display,
{
    // Metadata and basic operations
    fn metadata(&self) -> &DictMetadata;
    fn contains(&self, key: &K) -> Result<bool>;
    fn get(&self, key: &K) -> Result<Vec<u8>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn file_paths(&self) -> &[std::path::PathBuf];

    // Batch operations
    fn get_multiple(&self, keys: &[K]) -> Result<Vec<BatchResult>>;
    fn get_batch(&self, keys: &[K], batch_size: Option<usize>) -> Result<Vec<BatchResult>>;

    // Collection operations
    fn keys(&self) -> Result<Vec<K>>;
    fn values(&self) -> Result<Vec<Vec<u8>>>;
    fn get_range(&self, range: Range<usize>) -> Result<Vec<(K, Vec<u8>)>>;
    fn iter(&self) -> Result<EntryIterator<K>>;

    // Search operations
    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>>;
    fn search_fuzzy(&self, query: &str, max_distance: Option<u32>) -> Result<Vec<SearchResult>>;
    fn search_fulltext(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>>;
    fn prefix_iter(&self, prefix: &str) -> Result<Box<dyn Iterator<Item = Result<(K, Vec<u8>)>> + Send>>;

    // Maintenance operations
    fn reload_indexes(&mut self) -> Result<()>;
    fn clear_cache(&mut self);
    fn stats(&self) -> DictStats;
    fn build_indexes(&mut self) -> Result<()>;
}
```

**`DictBuilder<K>`**: Trait for building dictionaries (for creating new dictionary files)
```rust
pub trait DictBuilder<K> {
    fn add_entry(&mut self, key: K, entry: &[u8]) -> Result<()>;
    fn build(&mut self, output_path: &Path, config: Option<DictConfig>) -> Result<()>;
    fn set_metadata(&mut self, metadata: DictMetadata);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

**`HighPerformanceDict<K>`**: Trait for high-performance operations
```rust
pub trait HighPerformanceDict<K>: Dict<K>
where
    K: Hash + Eq + Clone + fmt::Display,
{
    fn binary_search_get(&self, key: &K) -> Result<Vec<u8>>;
    fn stream_search(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>>;
}
```

**`DictFormat<K>`**: Trait for dictionary formats
```rust
pub trait DictFormat<K> {
    const FORMAT_NAME: &'static str;
    const FORMAT_VERSION: &'static str;
    fn is_valid_format(path: &Path) -> Result<bool>;
    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<K> + Send + Sync>>;
}
```

#### 2.3 Constants

**Format Constants**:
- `FORMAT_MDICT`: "mdict"
- `FORMAT_STARDICT`: "stardict"
- `FORMAT_ZIM`: "zim"

**File Extension Constants**:
- `EXT_DICT`: ".dict"
- `EXT_IDX`: ".idx"
- `EXT_INFO`: ".info"
- `EXT_BTREE`: ".btree"
- `EXT_FTS`: ".fts"

### 3. Dictionary Module (`src/dict/`)

Implementations for various dictionary formats.

#### 3.1 Dictionary Loader (`src/dict/mod.rs`)

**`DictLoader`**: Dictionary format detection and loading
```rust
pub struct DictLoader {
    default_config: DictConfig,
}

// Methods:
impl DictLoader {
    pub fn new() -> Self;                              // Create new loader
    pub fn with_config(config: DictConfig) -> Self;    // Create with custom config
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Dict<String> + Send + Sync>>;  // Auto-detect format
    pub fn load_format<P: AsRef<Path>>(&self, path: P, format: &str) -> Result<Box<dyn Dict<String> + Send + Sync>>;  // Load specific format
    pub fn detect_format(&self, path: &Path) -> Result<String>;  // Detect format from file
    pub fn scan_directory<P: AsRef<Path>>(&self, dir: P) -> Result<Vec<PathBuf>>;  // Scan for dictionaries
    pub fn is_dictionary_file(&self, path: &Path) -> bool;  // Check if file is dictionary
    pub fn supported_formats(&self) -> Vec<String>;      // Get supported formats
    pub fn default_config(&self) -> &DictConfig;         // Get default config
    pub fn set_default_config(&mut self, config: DictConfig);  // Set default config
}
```

**`BatchOperations`**: Dictionary batch operations utilities
```rust
pub struct BatchOperations;

// Methods:
impl BatchOperations {
    pub fn load_batch<P: AsRef<Path>>(paths: &[P], config: Option<DictConfig>) -> Result<Vec<Box<dyn Dict<String> + Send + Sync>>>;
    pub fn search_multiple(dictionaries: &[Box<dyn Dict<String> + Send + Sync>], query: &str, search_type: SearchType) -> Result<Vec<SearchResult<String>>>;
    pub fn merge<K>(dictionaries: &[Box<dyn Dict<K> + Send + Sync>], output_path: &Path, format: &str) -> Result<()>
    where
        K: Clone + std::fmt::Display + serde::Serialize + serde::de::DeserializeOwned + Eq + std::hash::Hash;
    pub fn validate_batch<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<(PathBuf, bool)>>;
}
```

**`SearchType`**: Search type for batch operations
```rust
pub enum SearchType {
    Prefix(String),     // Prefix search
    Fuzzy(String),      // Fuzzy search
    Fulltext(String),   // Full-text search
}
```

**`SearchResult<K>`**: Search result for multiple dictionaries
```rust
pub struct SearchResult<K> {
    pub key: K,                              // Dictionary key
    pub entry: Vec<u8>,                      // Entry data
    pub score: Option<f32>,                  // Relevance score
    pub source_dict: Option<String>,         // Source dictionary name
    pub highlights: Option<Vec<(usize, usize)>>, // Highlight information
}
```

**Dictionary Utility Functions** (`utils` submodule):
```rust
pub fn get_dict_size<P: AsRef<Path>>(path: P) -> Result<u64>;
pub fn is_readable<P: AsRef<Path>>(path: P) -> bool;
pub fn get_dict_format<P: AsRef<Path>>(path: P) -> Result<String>;
pub fn copy_dict<P: AsRef<Path>>(source: P, destination: P, create_indexes: bool) -> Result<()>;
pub fn remove_dict<P: AsRef<Path>>(path: P) -> Result<()>;
pub fn list_dicts<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>>;
```

#### 3.2 MDict Implementation (`src/dict/mdict.rs`)

**`MDict`**: Monkey's Dictionary implementation
```rust
pub struct MDict {
    file_path: std::path::PathBuf,           // File path
    mmap: Option<Arc<Mmap>>,                 // Memory-mapped file
    file: Option<File>,                      // File for sequential access
    header: MdictHeader,                     // Header information
    btree_index: Option<BTreeIndex>,         // B-TREE index for fast lookups
    fts_index: Option<FtsIndex>,             // FTS index for full-text search
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>, // Cache for frequently accessed entries
    config: DictConfig,                      // Index configuration
    metadata: DictMetadata,                  // Cached metadata
}

// Methods:
impl MDict {
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self>;  // Create new MDict instance
    pub fn build_indexes(&mut self) -> Result<()>;  // Build indexes for this MDict
    pub fn file_paths(&self) -> Vec<std::path::PathBuf>;  // Get file paths for this dictionary
}
```

**`MdictHeader`**: MDict header information
```rust
struct MdictHeader {
    encoding: String,                        // Encoding name as in header (normalized)
    version: f64,                           // Version as parsed from GeneratedByEngineVersion
    encrypted: i32,                         // Encrypted flags (bitmask)
    rtl: bool,                              // Right-to-left flag
    title: String,                          // Title (or filename fallback)
    description: String,                    // Description (plain text)
    attributes: HashMap<String, String>,    // Raw attribute map for extensibility
    number_size: u8,                        // Number size for numeric fields (4 or 8)
    headword_block_info_pos: u64,           // Position of headword block info (absolute in file)
    headword_block_info_size: u64,          // Size of headword block info (compressed or plain)
    num_headword_blocks: u64,               // Number of headword blocks
    word_count: u64,                        // Total word count (entries)
    headword_block_size: u64,               // Size of headword block (compressed/decompressed descriptor)
    record_block_info_pos: u64,             // Position of record block info table
    total_records_size: u64,                // Total decompressed size of all records
    record_blocks: Vec<RecordIndex>,        // Record blocks (compressed/decompressed sizes and shadow offsets)
    file_size: u64,                         // Absolute file size for metadata and safety checks
}
```

**`RecordIndex`**: Record block index entry
```rust
struct RecordIndex {
    compressed_size: u64,                   // Compressed size
    decompressed_size: u64,                 // Decompressed size
    start_pos: u64,                         // Start position (relative to first record block) in compressed space
    shadow_start_pos: u64,                  // Start position in concatenated decompressed space
    shadow_end_pos: u64,                    // End position in decompressed space
}
```

**`MdictKeyEntry`**: One headword entry mapped to a record offset/size
```rust
struct MdictKeyEntry {
    key: String,                            // Key
    record_offset: u64,                     // Absolute record offset in concatenated decompressed record stream
    record_size: u64,                       // Length of the record data
}
```

#### 3.3 StarDict Implementation (`src/dict/stardict.rs`)

**`StarDict`**: StarDict dictionary implementation
```rust
pub struct StarDict {
    ifo_path: PathBuf,                      // .ifo path
    dict_path: PathBuf,                     // Associated .dict or .dict.dz
    dict_is_dz: bool,                       // Whether dict is DICTZIP-compressed (.dict.dz)
    syn_path: Option<PathBuf>,              // Optional .syn path
    header: StarDictHeader,                 // Parsed header
    index: HashMap<String, EntryLoc>,       // In-memory index: word → (offset,size)
    mmap: Option<Arc<Mmap>>,                // Memory-mapped .dict (for uncompressed dict)
    dict_file: File,                        // File handle for .dict/.dz
    btree_index: Option<BTreeIndex>,        // Optional BTree index for fast key lookups
    fts_index: Option<FtsIndex>,            // Optional FTS index for full-text search
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>, // Cache for frequently accessed entries
    config: DictConfig,                     // Configuration
    metadata: DictMetadata,                 // Cached metadata
}

// Methods:
impl StarDict {
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self>;  // Create new StarDict from .ifo file
}
```

**`Ifo`**: Parsed contents of .ifo file
```rust
struct Ifo {
    version: String,                        // Version
    bookname: String,                       // Book name
    wordcount: u64,                         // Word count
    synwordcount: u64,                      // Synonym word count
    idxfilesize: Option<u64>,               // Index file size
    idxoffsetbits: u32,                     // Index offset bits
    sametypesequence: Option<String>,       // Same type sequence
    dicttype: Option<String>,               // Dictionary type
    description: Option<String>,            // Description
    copyright: Option<String>,              // Copyright
    author: Option<String>,                 // Author
    email: Option<String>,                  // Email
    website: Option<String>,                // Website
    date: Option<String>,                   // Date
}
```

**`StarDictHeader`**: StarDict dictionary header/metadata
```rust
struct StarDictHeader {
    ifo: Ifo,                               // Parsed .ifo contents
    encoding: String,                       // Encoding for textual parts
    idx_64bit: bool,                        // True if index offsets are 64-bit
}
```

**`EntryLoc`**: Entry location in .dict or .dict.dz
```rust
struct EntryLoc {
    offset: u64,                            // Offset
    size: u64,                              // Size
}
```

#### 3.4 ZIM Implementation (`src/dict/zimdict.rs`)

**`ZimDict`**: ZIM format implementation
```rust
pub struct ZimDict {
    file_path: PathBuf,                     // Main ZIM file path
    mmap: Option<Arc<Mmap>>,                // Memory-mapped file for fast random access
    file: File,                             // File handle for IO fallback
    header: ZimHeader,                      // Parsed header
    mime_types: Vec<String>,                // Mime types list (index → string)
    btree_index: Option<BTreeIndex>,        // Optional BTree index (external)
    fts_index: Option<FtsIndex>,            // Optional FTS index (external)
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>, // Cache for frequently accessed entries
    config: DictConfig,                     // Configuration
    metadata: DictMetadata,                 // Cached metadata
}

// Methods:
impl ZimDict {
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self>;  // Create new ZimDict from .zim file
}
```

**`ZimHeader`**: ZIM file header (subset based on references/zim.cc ZIM_header)
```rust
struct ZimHeader {
    magic_number: u32,                      // Magic number
    major_version: u16,                     // Major version
    minor_version: u16,                     // Minor version
    article_count: u32,                     // Article count
    cluster_count: u32,                     // Cluster count
    url_ptr_pos: u64,                       // URL pointer position
    title_ptr_pos: u64,                     // Title pointer position
    cluster_ptr_pos: u64,                   // Cluster pointer position
    mime_list_pos: u64,                     // Mime list position
}
```

**`ArticleLoc`**: Location of an article blob: (cluster, blob_index)
```rust
struct ArticleLoc {
    cluster: u32,                           // Cluster
    blob: u32,                              // Blob index
}
```

#### 3.5 BGL Implementation (`src/dict/bgl.rs`)

**`BglDict`**: Lightweight BGL dictionary backed by sidecar indexes
```rust
pub struct BglDict {
    bgl_path: PathBuf,                      // Original BGL file path
    index_path: PathBuf,                    // Index/chunks file path (`.bglx` / `.idx`)
    header: BglIndexHeader,                 // Parsed header from index (for metadata/chunks_offset)
    btree_index: Option<BTreeIndex>,        // BTree-based index for key lookups
    fts_index: Option<FtsIndex>,            // Full-text search index (optional)
    cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>, // Cache for entries
    config: DictConfig,                     // Configuration
    metadata: DictMetadata,                 // Metadata
}

// Methods:
impl BglDict {
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self>;  // Create BglDict using existing BGL file and compatible sidecar index files
}
```

**`BglIndexHeader`**: Minimal BGL "index header" used only for metadata and chunks base offset
```rust
struct BglIndexHeader {
    signature: [u8; 4],                     // Magic signature, expected "BGLX"
    format_version: u32,                    // Format version (opaque here)
    article_count: u32,                     // Number of articles
    word_count: u32,                        // Number of words (for metadata only)
    chunks_offset: u64,                     // Offset to chunked article storage in index file
}
```

#### 3.6 DSL Implementation (`src/dict/dsl.rs`)

**`DslDict`**: Main DSL dictionary implementation
```rust
pub struct DslDict {
    dsl_path: PathBuf,                      // Primary path (.dsl or .dsl.dz)
    entries: HashMap<String, String>,       // Parsed entries (headword -> UTF-8 body)
    btree_index: Option<BTreeIndex>,        // Optional BTree index (sidecar)
    fts_index: Option<FtsIndex>,            // Optional FTS index (sidecar)
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>, // Cache for frequently accessed entries
    config: DictConfig,                     // Configuration
    metadata: DictMetadata,                 // Cached metadata
}

// Methods:
impl DslDict {
    pub fn new(path: &Path, config: DictConfig) -> Result<Self>;  // Load DSL dictionary from the given path
}

// Utility function:
pub fn levenshtein(a: &str, b: &str) -> usize;  // Simple Levenshtein distance used for DSL fuzzy search
```

**`DslEntry`**: In-memory representation of a parsed DSL entry
```rust
struct DslEntry {
    headword: String,                       // Headword
    body: String,                           // Body
}
```

**`DslEncoding`**: Supported DSL encodings
```rust
enum DslEncoding {
    Utf16Le,                                // UTF-16 Little Endian
    Utf16Be,                                // UTF-16 Big Endian
    Utf8,                                   // UTF-8
    Windows1252,                            // Windows-1252
    Windows1251,                            // Windows-1251
    Windows1250,                            // Windows-1250
}
```

### 4. Index Module (`src/index/`)

High-performance indexing system for dictionary operations.

#### 4.1 Index Core (`src/index/mod.rs`)

**`IndexStats`**: Common index statistics
```rust
pub struct IndexStats {
    pub entries: u64,                       // Number of entries indexed
    pub size: u64,                          // Index file size in bytes
    pub build_time: u64,                    // Index build time in milliseconds
    pub version: String,                    // Index version
    pub config: IndexConfig,                // Index configuration
}
```

**`IndexConfig`**: Configuration for index operations
```rust
pub struct IndexConfig {
    pub btree_order: Option<usize>,         // B-TREE order (branching factor)
    pub fts_config: FtsConfig,              // FTS analyzer settings
    pub compression: Option<CompressionConfig>, // Compression settings
    pub build_in_memory: bool,              // Whether to build index in memory first
    pub max_memory: Option<u64>,            // Maximum memory usage during build (bytes)
}
```

**`FtsConfig`**: Full-Text Search configuration
```rust
pub struct FtsConfig {
    pub min_token_len: usize,               // Minimum token length for indexing
    pub max_token_len: usize,               // Maximum token length for indexing
    pub use_stemming: bool,                 // Whether to use stemming
    pub stop_words: Vec<String>,            // Stop words to ignore during indexing
    pub language: Option<String>,           // Analyzer language
}
```

**`CompressionConfig`**: Compression configuration
```rust
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,    // Compression algorithm
    pub level: u32,                         // Compression level (0-9 for gzip, 1-19 for zstd)
}
```

**`CompressionAlgorithm`**: Compression algorithm types
```rust
pub enum CompressionAlgorithm {
    None,                                   // No compression
    Gzip,                                   // GZIP compression
    Lz4,                                    // LZ4 compression
    Zstd,                                   // Zstandard compression
}
```

**`IndexError`**: Error types specific to index operations
```rust
pub enum IndexError {
    CorruptedIndex(String),                 // Index corruption detected
    VersionMismatch { expected: String, found: String }, // Index version mismatch
    NotBuilt(String),                       // Index not built
    IoError(String),                        // Index I/O error
    ConfigError(String),                    // Index configuration error
    InsufficientMemory(String),             // Index too large for memory
}
```

**`IndexManager`**: Manager for multiple indexes
```rust
pub struct IndexManager {
    btree: Option<btree::BTreeIndex>,       // B-TREE index
    fts: Option<fts::FtsIndex>,             // FTS index
    config: IndexConfig,                    // Index configuration
    paths: HashMap<&'static str, PathBuf>,  // Paths to index files
    stats: IndexStats,                      // Index statistics
}

// Methods:
impl IndexManager {
    pub fn new(config: IndexConfig) -> Self;  // Create new index manager
    pub fn build_all(&mut self, entries: &[(String, Vec<u8>)]) -> Result<()>;  // Build both B-TREE and FTS indexes
    pub fn load_all(&mut self, base_path: &Path, extensions: &[(&str, &str)]) -> Result<()>;  // Load indexes from files
    pub fn save_all(&self, base_path: &Path, extensions: &[(&str, &str)]) -> Result<()>;  // Save indexes to files
    pub fn binary_search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>>;  // Binary search using B-TREE index
    pub fn fulltext_search(&self, query: &str) -> Result<Vec<(String, f32)>>;  // Search using FTS index
    pub fn stats(&self) -> &IndexStats;             // Get all statistics
    pub fn is_built(&self) -> bool;                 // Check if indexes are built
    pub fn clear(&mut self);                        // Clear all indexes
    pub fn verify(&self) -> Result<bool>;           // Verify all indexes
}
```

**`Index`**: Trait that defines common index operations
```rust
pub trait Index: Send + Sync {
    const INDEX_TYPE: &'static str;         // Index type identifier

    fn build(&mut self, entries: &[(String, Vec<u8>)], config: &IndexConfig) -> Result<()>;  // Build index from entries
    fn load(&mut self, path: &Path) -> Result<()>;  // Load index from file
    fn save(&self, path: &Path) -> Result<()>;      // Save index to file
    fn stats(&self) -> &IndexStats;         // Get index statistics
    fn is_built(&self) -> bool;             // Check if index is built
    fn clear(&mut self);                    // Clear the index
    fn verify(&self) -> Result<bool>;       // Verify index integrity
}
```

#### 4.2 B-TREE Index (`src/index/btree.rs`)

**`BTreeIndex`**: Production-ready B-Tree index implementation
```rust
pub struct BTreeIndex {
    order: usize,                           // Maximum number of keys per node (fan-out minus one)
    root: Option<usize>,                    // Root node index
    nodes: Vec<BTreeNode>,                  // All nodes backing the tree
    stats: IndexStats,                      // Statistics about the index
    lock: Arc<RwLock<()>>,                  // Thread-safe access control
    node_cache: LruCache<usize, BTreeNode>, // Lightweight cache for recently accessed nodes to avoid cloning
}

// Methods:
impl BTreeIndex {
    pub fn new() -> Self;                   // Create new empty B-Tree index
    pub fn with_order(order: usize) -> Self; // Create new B-Tree index with requested order
    pub fn binary_search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>>;  // Perform binary search for key
    pub fn search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>>;  // Public search helper
    pub fn range_query(&self, start: &str, end: &str) -> Result<Vec<(String, u64)>>;  // Get range of keys inclusively between start and end
    pub fn validate(&self) -> Result<bool>;  // Validate B-Tree properties
}
```

**`BTreeNode`**: B-Tree node containing keys and child pointers
```rust
struct BTreeNode {
    keys: Vec<String>,                      // Keys in this node (sorted)
    values: Vec<u64>,                       // Values (file offsets) in this node
    children: Vec<usize>,                   // Child pointers stored as node indices
    is_leaf: bool,                          // Whether this is a leaf node
}
```

**`BTreeSnapshot`**: On-disk snapshot persisted through save()/load()
```rust
struct BTreeSnapshot {
    order: usize,                           // Order
    root: Option<usize>,                    // Root
    nodes: Vec<BTreeNode>,                  // Nodes
    stats: IndexStats,                      // Statistics
}
```

**`RangeQueryResult`**: Range query aggregation helper
```rust
pub struct RangeQueryResult {
    pub keys: Vec<String>,                  // Matching keys
    pub values: Vec<u64>,                   // Corresponding values (file offsets)
    pub count: usize,                       // Total number of results
}

// Methods:
impl RangeQueryResult {
    pub fn new() -> Self;
    pub fn add(&mut self, key: String, value: u64);
}
```

#### 4.3 FTS Index (`src/index/fts.rs`)

**`FtsIndex`**: FTS Index implementation using inverted indexing
```rust
pub struct FtsIndex {
    inverted_index: HashMap<String, InvertedIndexEntry>, // Inverted index: term -> posting list
    documents: HashMap<DocId, Document>,    // Forward index: doc_id -> document
    next_term_id: TermId,                   // Next available term ID
    next_doc_id: DocId,                     // Next available document ID
    stop_words: HashSet<String>,            // Stop words set
    term_stats: HashMap<String, u32>,       // Term statistics
    config: FtsConfig,                      // Index configuration
    stats: IndexStats,                      // Index statistics
    lock: Arc<RwLock<()>>,                  // Thread-safe access
}

// Methods:
impl FtsIndex {
    pub fn new() -> Self;                   // Create new FTS index
    pub fn with_config(config: FtsConfig) -> Self;  // Create FTS index with custom configuration
    pub fn search(&self, query: &str) -> Result<Vec<(String, f32)>>;  // Search for documents containing the query
    pub fn prefix_search(&self, prefix: &str) -> Result<Vec<String>>;  // Get terms starting with prefix
    pub fn term_frequency(&self, term: &str) -> u32;  // Get term frequency
    pub fn document_frequency(&self, term: &str) -> u32;  // Get document frequency
    pub fn vocabulary_size(&self) -> usize; // Get vocabulary size
    pub fn avg_doc_length(&self) -> f32;    // Get average document length
    pub fn suggest_spelling(&self, query: &str) -> Result<Vec<String>>;  // Get suggestions for misspelled query
    pub fn get_snippet(&self, doc_id: DocId, query: &str, max_length: usize) -> Option<String>;  // Get highlighted snippet for a document
    pub fn validate(&self) -> Result<bool>;  // Validate index integrity
    pub fn get_stats(&self) -> &IndexStats; // Get index statistics
}
```

**`FtsSearchResult`**: Search result with score
```rust
pub struct FtsSearchResult {
    pub doc_id: DocId,                      // Document ID
    pub key: String,                        // Document key (word)
    pub score: f32,                         // Relevance score
    pub highlights: Vec<(usize, usize)>,    // Snippet highlighting positions
}
```

**Supporting Types**:
```rust
type DocId = u32;                           // Document ID for FTS operations
type TermId = u32;                          // Term ID for token indexing

struct Token {                              // Token with position information
    text: String,                           // Token text
    term_id: TermId,                        // Term ID
    position: u32,                          // Position in document
    doc_freq: u32,                          // Document frequency
}

struct InvertedIndexEntry {                 // Inverted index entry
    term_id: TermId,                        // Term ID
    term: String,                           // Term text
    postings: Vec<Posting>,                 // Documents containing this term
    doc_freq: u32,                          // Total document frequency
    term_freq: u32,                         // Total term frequency
}

struct Posting {                            // Posting list entry
    doc_id: DocId,                          // Document ID
    term_freq: u32,                         // Term frequency in document
    positions: Vec<u32>,                    // Positions where term occurs
}

struct Document {                           // Document representation
    doc_id: DocId,                          // Document ID
    key: String,                            // Document key (word)
    content: Vec<u8>,                       // Document content
    doc_length: u32,                        // Document length in tokens
}
```

### 5. Utility Module (`src/util/`)

Utility functions for dictionary operations.

#### 5.1 File Utilities (`src/util/mod.rs` - file_utils submodule)

**File Operations**:
```rust
pub fn read_file(path: &Path) -> Result<Vec<u8>>;                          // Read entire file into memory
pub fn read_file_mmap(path: &Path) -> Result<memmap2::Mmap>;              // Read file with memory mapping
pub fn write_file_atomic(path: &Path, data: &[u8]) -> Result<()>;         // Write data to file with atomic operations
pub fn file_size(path: &Path) -> Result<u64>;                             // Get file size
pub fn is_readable(path: &Path) -> bool;                                  // Check if file exists and is readable
pub fn ensure_dir(path: &Path) -> Result<()>;                             // Create directory if it doesn't exist
pub fn crc32(data: &[u8]) -> u32;                                         // Calculate CRC32 checksum
pub fn verify_crc32(path: &Path, expected_crc: u32) -> Result<bool>;      // Verify file integrity with CRC32
```

#### 5.2 Buffer Utilities (`src/util/mod.rs` - buffer submodule)

**Read Operations**:
```rust
pub fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<()>;  // Read bytes from reader with error handling
pub fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32>;               // Read 32-bit unsigned integer (little-endian)
pub fn read_u32_be<R: Read>(reader: &mut R) -> Result<u32>;               // Read 32-bit unsigned integer (big-endian)
pub fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64>;               // Read 64-bit unsigned integer (little-endian)
pub fn read_u64_be<R: Read>(reader: &mut R) -> Result<u64>;               // Read 64-bit unsigned integer (big-endian)
pub fn read_varint<R: Read>(reader: &mut R) -> Result<u64>;               // Read variable-length integer (VARINT)
pub fn read_string<R: Read, F: FnMut(String) -> Result<()>>(reader: &mut R, callback: F) -> Result<()>; // Read length-prefixed string
pub fn read_u8<R: Read>(reader: &mut R) -> Result<u8>;                    // Read 8-bit unsigned integer
pub fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16>;               // Read 16-bit unsigned integer (little-endian)
pub fn read_u16_be<R: Read>(reader: &mut R) -> Result<u16>;               // Read 16-bit unsigned integer (big-endian)
```

**Write Operations**:
```rust
pub fn write_all<W: Write>(writer: &mut W, buf: &[u8]) -> Result<()>;     // Write bytes to writer with error handling
pub fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> Result<()>; // Write 32-bit unsigned integer (little-endian)
pub fn write_u32_be<W: Write>(writer: &mut W, value: u32) -> Result<()>; // Write 32-bit unsigned integer (big-endian)
pub fn write_u64_le<W: Write>(writer: &mut W, value: u64) -> Result<()>; // Write 64-bit unsigned integer (little-endian)
pub fn write_u64_be<W: Write>(writer: &mut W, value: u64) -> Result<()>; // Write 64-bit unsigned integer (big-endian)
pub fn write_varint<W: Write>(writer: &mut W, value: u64) -> Result<()>; // Write variable-length integer (VARINT)
pub fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<()>;    // Write length-prefixed string
```

#### 5.3 Binary Search Utilities (`src/util/mod.rs` - binary_search submodule)

```rust
pub fn search_sorted<'a, K, V>(keys: &'a [K], values: &'a [V], target: &K, compare: impl Fn(&K, &K) -> std::cmp::Ordering) -> Option<(usize, &'a V)> // Binary search in sorted array
where K: Ord;
pub fn lower_bound<K>(keys: &[K], target: &K, compare: impl Fn(&K, &K) -> std::cmp::Ordering) -> usize // Binary search for lower bound
where K: Ord;
pub fn upper_bound<K>(keys: &[K], target: &K, compare: impl Fn(&K, &K) -> std::cmp::Ordering) -> usize // Binary search for upper bound
where K: Ord;
```

#### 5.4 Memory Management Utilities (`src/util/mod.rs` - memory submodule)

```rust
pub fn optimal_cache_size(entries: usize, avg_entry_size: usize) -> usize; // Calculate optimal cache size based on available memory
pub fn total_memory() -> u64;                     // Get total available system memory (bytes)
pub fn used_memory() -> u64;                      // Get currently used memory by the process (bytes)
pub fn has_sufficient_memory(required: u64) -> bool; // Check if we have enough memory for an operation
pub fn clear_buffer(buf: &mut [u8]);              // Clear memory buffer to prevent data leakage
pub fn zero_sensitive<T: Default>(data: &mut T);  // Securely zero sensitive data
```

#### 5.5 Performance Monitoring Utilities (`src/util/mod.rs` - performance submodule)

**`Profiler`**: Simple performance profiler
```rust
pub struct Profiler {
    start_time: Instant,                    // Start time
    operations: std::collections::HashMap<String, u64>, // Operations counter
}

// Methods:
impl Profiler {
    pub fn new() -> Self;                   // Create new profiler
    pub fn record(&mut self, operation: &str, count: u64); // Record an operation count
    pub fn elapsed(&self) -> std::time::Duration; // Get elapsed time since profiler creation
    pub fn print_stats(&self);              // Print statistics
    pub fn operations_per_second(&self, operation: &str) -> f64; // Get operations per second
}
```

**Performance Functions**:
```rust
pub fn measure_time<T>(f: impl FnOnce() -> T) -> (T, std::time::Duration); // Measure function execution time
pub fn benchmark<T>(iterations: usize, mut f: impl FnMut() -> T) -> (T, std::time::Duration, std::time::Duration); // Benchmark a function
```

#### 5.6 Serialization Utilities (`src/util/mod.rs` - serialization submodule)

```rust
pub fn serialize_to_vec<T: serde::Serialize>(data: &T) -> Result<Vec<u8>>; // Serialize data with error handling
pub fn deserialize_from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T>; // Deserialize data with error handling
pub fn serialize_and_compress<T: serde::Serialize>(data: &T, compression: compression::CompressionAlgorithm) -> Result<Vec<u8>>; // Serialize and compress data
pub fn decompress_and_deserialize<T: serde::de::DeserializeOwned>(compressed: &[u8], compression: compression::CompressionAlgorithm) -> Result<T>; // Decompress and deserialize data
pub fn serialize_with_metadata<T: serde::Serialize>(data: &T, version: &str) -> Result<Vec<u8>>; // Serialize with metadata (version, timestamp, etc.)
pub fn deserialize_with_metadata<T: serde::de::DeserializeOwned>(bytes: &[u8], expected_version: &str) -> Result<T>; // Deserialize with metadata
```

#### 5.7 Hash Utilities (`src/util/mod.rs` - hash submodule)

```rust
pub fn fast_hash(data: &[u8]) -> u64;        // Calculate hash using fast non-cryptographic hash
pub fn secure_hash(data: &[u8]) -> Vec<u8>;  // Calculate hash using cryptographically secure hash
pub fn hash_file(path: &Path, secure: bool) -> Result<Vec<u8>>; // Hash a file
```

#### 5.8 Test Utilities (`src/util/mod.rs` - test_utils submodule)

```rust
pub fn generate_test_entries(count: usize) -> Vec<(String, Vec<u8>)>; // Generate test dictionary entries
pub fn temp_dir() -> Result<std::path::PathBuf>; // Create temporary directory for testing
pub fn cleanup_temp_dir(path: &std::path::Path) -> Result<()>; // Clean up temporary directory
pub fn validate_dictionary_integrity<K: std::fmt::Display + std::cmp::PartialEq + std::cmp::Ord>(entries: &[(K, Vec<u8>)]) -> Result<()>; // Validate dictionary integrity
pub fn benchmark_dict_operations<K, D>(dict: &D, test_keys: &[K], iterations: usize) -> Result<std::collections::HashMap<String, f64>> // Benchmark dictionary operations
where K: Clone + std::fmt::Display + std::hash::Hash + std::cmp::Eq, D: crate::traits::Dict<K>;
```

### 6. Compression Utilities (`src/util/compression.rs`)

Compression and decompression functions using various algorithms.

#### 6.1 Core Functions

**`CompressionAlgorithm`**: Compression algorithm types
```rust
pub enum CompressionAlgorithm {
    None,                                   // No compression
    Gzip,                                   // GZIP compression
    Lz4,                                    // LZ4 compression
    Zstd,                                   // Zstandard compression
}
```

**Main Functions**:
```rust
pub fn compress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>>; // Compress data using specified algorithm
pub fn decompress(compressed: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>>; // Decompress data using specified algorithm
pub fn compression_level(level: u32, algorithm: &CompressionAlgorithm) -> u32; // Get compression level based on algorithm
pub fn max_compression_level(algorithm: &CompressionAlgorithm) -> u32; // Get maximum compression level for algorithm
pub fn suggested_compression_level(algorithm: &CompressionAlgorithm) -> u32; // Get suggested compression level for size vs speed
```

**Compression Analysis**:
```rust
pub fn estimate_compression_ratio(original_size: u64, algorithm: &CompressionAlgorithm, level: u32) -> f32; // Estimate compression ratio
pub fn get_algorithm_settings(algorithm: &CompressionAlgorithm) -> AlgorithmSettings; // Get algorithm-specific settings
```

**`AlgorithmSettings`**: Algorithm-specific settings
```rust
pub struct AlgorithmSettings {
    pub supports_streaming: bool,           // Supports streaming
    pub supports_dictionary: bool,          // Supports dictionary
    pub typical_ratio: f32,                 // Typical compression ratio
    pub speed_category: SpeedCategory,      // Speed category
    pub memory_overhead: u64,               // Memory overhead
}
```

**`SpeedCategory`**: Speed categories
```rust
pub enum SpeedCategory {
    VeryFast,                               // Very fast
    Fast,                                   // Fast
    Medium,                                 // Medium
    Slow,                                   // Slow
}
```

**Streaming Functions**:
```rust
pub fn compress_stream<R: Read, W: Write>(input: &mut R, output: &mut W, algorithm: CompressionAlgorithm) -> Result<u64>; // Compress data with streaming for large datasets
pub fn decompress_stream<R: Read, W: Write>(input: &mut R, output: &mut W, algorithm: CompressionAlgorithm) -> Result<u64>; // Decompress data with streaming for large datasets
```

### 7. Encoding Utilities (`src/util/encoding.rs`)

Encoding detection and conversion utilities.

#### 7.1 Text Encoding Types

**`TextEncoding`**: Supported text encodings
```rust
pub enum TextEncoding {
    Utf8,                                   // UTF-8 encoding
    Utf16Le,                                // UTF-16 Little Endian
    Utf16Be,                                // UTF-16 Big Endian
    Windows1252,                            // Windows-1252 (Latin-1)
    Iso88591,                               // ISO-8859-1 (Latin-1)
    Gb2312,                                 // GBK/GB2312 (Chinese)
    Big5,                                   // Big5 (Traditional Chinese)
    ShiftJis,                               // Shift-JIS (Japanese)
    EucKr,                                  // EUC-KR (Korean)
    Unknown,                                // Unknown encoding
}
```

**Methods for TextEncoding**:
```rust
impl TextEncoding {
    pub fn name(&self) -> &'static str;     // Get canonical name
    pub fn is_unicode(&self) -> bool;       // Check if Unicode-based
    pub fn is_variable_width(&self) -> bool; // Check if variable-width characters
    pub fn max_char_bytes(&self) -> usize;  // Get maximum byte length for single character
}
```

#### 7.2 Detection and Conversion Functions

**Core Functions**:
```rust
pub fn detect_encoding(data: &[u8]) -> Result<TextEncoding>; // Detect encoding of byte data
pub fn convert_to_utf8(data: &[u8], from_encoding: TextEncoding) -> Result<String>; // Convert byte data from one encoding to UTF-8 string
pub fn is_valid_utf8_str(s: &str) -> bool; // Validate if string is valid UTF-8
pub fn get_encoding_stats(encoding: TextEncoding) -> EncodingStats; // Get encoding statistics
```

**`EncodingStats`**: Encoding statistics
```rust
pub struct EncodingStats {
    pub name: &'static str,                 // Encoding name
    pub supports_unicode: bool,             // Supports Unicode
    pub max_char_size: usize,               // Maximum character size
    pub is_variable_width: bool,            // Is variable width
    pub common_in: Vec<&'static str>,       // Common usage contexts
}
```

## Feature Flags

- **`cli`**: Enables command-line interface utilities
- **`bench`**: Enables benchmarking tests

## Usage Examples

### Basic Dictionary Loading

```rust
use dictutils::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = DictLoader::new();
    let dict = loader.load("path/to/dictionary.mdict")?;
    
    let entry = dict.get(&"example".to_string())?;
    println!("Entry: {}", String::from_utf8_lossy(&entry));
    
    Ok(())
}
```

### Batch Operations

```rust
use dictutils::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = DictLoader::new();
    let dict = loader.load("path/to/dictionary.mdict")?;
    
    let keys = vec!["word1".to_string(), "word2".to_string(), "word3".to_string()];
    let results = dict.get_batch(&keys, Some(10))?;
    
    for result in results {
        match result.entry {
            Some(entry) => println!("Found: {}", String::from_utf8_lossy(&entry)),
            None => println!("Not found: {}", result.word),
        }
    }
    
    Ok(())
}
```

### Search Operations

```rust
use dictutils::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = DictLoader::new();
    let dict = loader.load("path/to/dictionary.mdict")?;
    
    // Prefix search
    let prefix_results = dict.search_prefix("pre", Some(10))?;
    for result in prefix_results {
        println!("Prefix match: {}", result.word);
    }
    
    // Full-text search
    let fts_results = dict.search_fulltext("search terms")?;
    for result in fts_results {
        println!("FTS match: {}", result.word);
    }
    
    Ok(())
}
```

### Custom Configuration

```rust
use dictutils::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DictConfig {
        load_btree: true,           // Fast key lookups
        load_fts: true,             // Full-text search
        use_mmap: true,             // Memory mapping
        cache_size: 2000,           // Larger cache
        batch_size: 200,            // Larger batches
        ..Default::default()
    };
    
    let loader = DictLoader::with_config(config);
    let dict = loader.load("path/to/large_dictionary.mdict")?;
    
    Ok(())
}
```

## Error Handling

The library uses a custom `DictError` enum for all error handling:

```rust
match dict.get(&key) {
    Ok(entry) => println!("Found: {}", String::from_utf8_lossy(&entry)),
    Err(DictError::FileNotFound(path)) => println!("File not found: {}", path),
    Err(DictError::InvalidFormat(msg)) => println!("Invalid format: {}", msg),
    Err(DictError::IndexError(msg)) => println!("Index error: {}", msg),
    Err(e) => println!("Other error: {}", e),
}
```

## Thread Safety

All dictionary operations are thread-safe and can be shared across threads using standard Rust concurrency patterns. The library uses:

- `Arc<RwLock<T>>` for shared mutable access
- `Send + Sync` bounds on trait objects
- Lock-free operations where possible

## Performance Considerations

1. **Memory Mapping**: Enable for files > 100MB using `use_mmap: true`
2. **Caching**: Adjust `cache_size` based on available memory and access patterns
3. **Batch Operations**: Use `get_batch()` for multiple lookups
4. **Index Building**: Build indexes once and reuse them
5. **Compression**: Choose appropriate compression algorithm based on use case

## Supported Dictionary Formats

### Monkey's Dictionary (MDict)
- File extensions: `.mdict`
- Features: B-TREE indexing, FTS, compression
- Supports: UTF-16LE, UTF-8, various encodings

### StarDict
- File extensions: `.ifo` (entry point), `.idx`, `.dict`
- Features: B-TREE indexing, FTS, synonyms (`.syn`)
- Supports: DICTZIP compression (`.dict.dz`)

### ZIM
- File extensions: `.zim`
- Features: Article-based storage, compression
- Requires: External indexing for full functionality

### Babylon (BGL)
- File extensions: `.bgl`
- Features: Sidecar index support
- Requires: `.bglx` or `.idx` sidecar files

### DSL (ABBYY Lingvo)
- File extensions: `.dsl`, `.dsl.dz`
- Features: Text-based, compression support
- Supports: UTF-16LE, UTF-8, Windows encodings

## Implementation Details

### B-TREE Index
- Configurable order (default: 256)
- Thread-safe with RwLock
- Persistent storage with serialization
- Range queries and validation

### Full-Text Search Index
- Inverted indexing
- TF-IDF scoring
- Stop word filtering
- Configurable tokenization

### Compression Support
- GZIP, LZ4, Zstandard
- Streaming operations for large files
- Memory-efficient decompression
- Algorithm-specific optimizations

### Encoding Detection
- BOM detection
- Statistical analysis
- UTF-8 validation
- Multi-encoding support

This documentation covers all public APIs, types, and functionality provided by the DictUtils crate. The library is designed for high-performance dictionary operations with a focus on flexibility, thread safety, and support for multiple dictionary formats.
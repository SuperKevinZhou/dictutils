//! MDict (Monkey's Dictionary) implementation
//!
//! This module implements support for Monkey's Dictionary format,
//! which is used by many offline dictionary applications.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::{RwLock, RwLockReadGuard};

use crate::traits::{
    BatchResult, Dict, DictConfig, DictError, DictMetadata, DictStats, EntryIterator,
    HighPerformanceDict, Result, SearchResult,
};
use crate::index::{btree::BTreeIndex, fts::FtsIndex, IndexConfig, Index};
use crate::util::compression::{self, CompressionAlgorithm};
use crate::util::encoding::{self, TextEncoding};
use crate::util::file_utils;
use crate::util::buffer::{self, read_u32_le, read_string, read_varint};

 // MDict (MDX/MDD) format constants derived from mdictparser.cc / MDX spec.
 // We intentionally do NOT use the previous fake header layout.
 const MDICT_MAX_KEY_LENGTH: usize = 16 * 1024;
 const MDICT_MAX_VALUE_LENGTH: usize = 4 * 1024 * 1024; // 4MB safety cap
 const MDICT_MAX_HEADER_TEXT: usize = 512 * 1024;
 const MDICT_MAX_BLOCK_INFO: usize = 16 * 1024 * 1024;

 /// Compression types inside MDX/MDD blocks (per mdictparser.cc)
 #[derive(Debug, Clone, Copy, PartialEq, Eq)]
 enum MdictBlockCompression {
     None,
     Lzo,
     Zlib,
 }

#[derive(Debug, Clone)]
struct MdictHeader {
    /// Encoding name as in header (normalized)
    encoding: String,
    /// Version as parsed from GeneratedByEngineVersion
    version: f64,
    /// Encrypted flags (bitmask, see mdictparser.cc)
    encrypted: i32,
    /// Right-to-left flag
    rtl: bool,
    /// Title (or filename fallback)
    title: String,
    /// Description (plain text)
    description: String,
    /// Raw attribute map for extensibility
    attributes: HashMap<String, String>,
    /// Number size for numeric fields (4 or 8)
    number_size: u8,
    /// Position of headword block info (absolute in file)
    headword_block_info_pos: u64,
    /// Size of headword block info (compressed or plain)
    headword_block_info_size: u64,
    /// Number of headword blocks
    num_headword_blocks: u64,
    /// Total word count (entries)
    word_count: u64,
    /// Size of headword block (compressed/decompressed descriptor)
    headword_block_size: u64,
    /// Position of record block info table
    record_block_info_pos: u64,
    /// Total decompressed size of all records
    total_records_size: u64,
    /// Record blocks (compressed/decompressed sizes and shadow offsets)
    record_blocks: Vec<RecordIndex>,
    /// Absolute file size for metadata and safety checks
    file_size: u64,
}

/// Record block index entry (mirrors MdictParser::RecordIndex)
#[derive(Debug, Clone)]
struct RecordIndex {
    compressed_size: u64,
    decompressed_size: u64,
    /// Start position (relative to first record block) in compressed space
    start_pos: u64,
    /// Start position in concatenated decompressed space
    shadow_start_pos: u64,
    /// End position in decompressed space
    shadow_end_pos: u64,
}

impl RecordIndex {
    fn contains_decompressed_offset(&self, off: u64) -> bool {
        off >= self.shadow_start_pos && off < self.shadow_end_pos
    }
}

/// One headword entry mapped to a record offset/size (built from key + record index)
#[derive(Debug, Clone)]
struct MdictKeyEntry {
    key: String,
    /// Absolute record offset in concatenated decompressed record stream
    record_offset: u64,
    /// Length of the record data
    record_size: u64,
}

/// MDict implementation
pub struct MDict {
    /// File path
    file_path: std::path::PathBuf,
    /// Memory-mapped file
    mmap: Option<Arc<Mmap>>,
    /// File for sequential access
    file: Option<File>,
    /// Header information
    header: MdictHeader,
    /// B-TREE index for fast lookups
    btree_index: Option<BTreeIndex>,
    /// FTS index for full-text search
    fts_index: Option<FtsIndex>,
    /// Cache for frequently accessed entries
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>,
    /// Index configuration
    config: DictConfig,
    /// Cached metadata
    metadata: DictMetadata,
}

impl MDict {
    /// Create a new MDict instance from a file
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self> {
        let path = path.as_ref();
        let file_path = path.to_path_buf();

        // Verify file exists
        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        // Open file for reading
        let file = File::open(path)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        // Read and parse header
        let header = Self::read_header(&file, path)?;

        // Set up memory mapping if enabled
        let mmap = if config.use_mmap {
            Some(Arc::new(unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| DictError::MmapError(e.to_string()))?
            }))
        } else {
            None
        };

        // Initialize indexes
        let (btree_index, fts_index) = if config.load_btree || config.load_fts {
            Self::load_indexes(&file_path, &config, &header)?
        } else {
            (None, None)
        };

        // Create cache
        let entry_cache = Arc::new(RwLock::new(lru_cache::LruCache::new(config.cache_size)));

        // Build metadata from parsed MDX header attributes and file size
        let file_size = path.metadata()
            .map(|m| m.len())
            .unwrap_or(0);

        let name = header
            .attributes
            .get("Title")
            .cloned()
            .unwrap_or_else(|| "MDict".to_string());

        let metadata = DictMetadata {
            name,
            version: format!("{}", header.version),
            entries: header.word_count,
            description: Some(header.description.clone()).filter(|s| !s.is_empty()),
            author: header.attributes.get("Author").cloned(),
            language: header.attributes.get("DictCharset").cloned()
                .or_else(|| header.attributes.get("Encoding").cloned()),
            file_size,
            created: header.attributes.get("CreationDate").cloned(),
            has_btree: btree_index.is_some(),
            has_fts: fts_index.is_some(),
        };

        Ok(Self {
            file_path,
            mmap,
            file: Some(file),
            header,
            btree_index,
            fts_index,
            entry_cache,
            config,
            metadata,
        })
    }

    /// Read and parse MDX header following mdictparser.cc semantics.
    ///
    /// NOTE:
    /// - This intentionally drops the previous custom "MDict" binary header.
    /// - MDX starts with:
    ///     [4-byte big-endian headerTextSize]
    ///     [headerTextSize bytes of UTF-16LE XML-like header]
    ///     [4-byte little-endian Adler-32 of that UTF-16LE buffer]
    ///   followed by headword/record block info.
    /// - We only parse enough to:
    ///     * determine encoding
    ///     * determine number_size (4 or 8)
    ///     * honor Encrypted, Left2Right, Title, Description
    ///   Block/record indexes are parsed lazily in follow-up code.
    fn read_header(file: &File, path: &Path) -> Result<MdictHeader> {
        // Use a small local Adler32 implementation to avoid extra dependencies.
        fn adler32(bytes: &[u8]) -> u32 {
            const MOD_ADLER: u32 = 65521;
            let mut a: u32 = 1;
            let mut b: u32 = 0;
            for &byte in bytes {
                a = (a + byte as u32) % MOD_ADLER;
                b = (b + a) % MOD_ADLER;
            }
            (b << 16) | a
        }

        let mut reader = BufReader::new(file);

        // 1) Read header text size (big-endian 32-bit)
        let header_text_size = buffer::read_u32_be(&mut reader)? as usize;
        if header_text_size == 0 || header_text_size > MDICT_MAX_HEADER_TEXT {
            return Err(DictError::InvalidFormat(format!(
                "Invalid MDX header size: {}",
                header_text_size
            )));
        }

        // 2) Read header text (UTF-16LE XML-like)
        let mut header_text_raw = vec![0u8; header_text_size];
        reader.read_exact(&mut header_text_raw)?;

        // 3) Read Adler-32 checksum (little-endian)
        let checksum_le = buffer::read_u32_le(&mut reader)?;
        let calc = adler32(&header_text_raw);
        if calc != checksum_le {
            return Err(DictError::InvalidFormat(
                "MDX header checksum mismatch".to_string(),
            ));
        }

        // 4) Decode header text as UTF-16LE (MDX header is UTF-16LE XML-like text)
        let u16_len = header_text_raw.len() / 2;
        let mut u16_buf = Vec::with_capacity(u16_len);
        for i in 0..u16_len {
            let lo = header_text_raw[2 * i] as u16;
            let hi = header_text_raw[2 * i + 1] as u16;
            u16_buf.push(lo | (hi << 8));
        }
        let header_text = String::from_utf16_lossy(&u16_buf);

        // 5) Parse attributes from the pseudo-XML header root element
        // Header looks like: <Dictionary GeneratedByEngineVersion="..." ...>
        let attributes = parse_mdict_header_attributes(&header_text);

        // Normalize encoding
        let mut encoding = attributes
            .get("Encoding")
            .cloned()
            .unwrap_or_else(|| "UTF-16LE".to_string());

        if encoding.eq_ignore_ascii_case("GBK") || encoding.eq_ignore_ascii_case("GB2312") {
            encoding = "GB18030".to_string();
        } else if encoding.is_empty() || encoding.eq_ignore_ascii_case("UTF-16") {
            encoding = "UTF-16LE".to_string();
        }

        // Version → number_size
        let version = attributes
            .get("GeneratedByEngineVersion")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0);
        let number_size: u8 = if version < 2.0 { 4 } else { 8 };

        // Encrypted?
        let encrypted = attributes
            .get("Encrypted")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);

        // RTL (Left2Right == "Yes" means LTR)
        let rtl = attributes
            .get("Left2Right")
            .map(|v| v != "Yes")
            .unwrap_or(false);

        // Title / Description
        let title_attr = attributes.get("Title").cloned().unwrap_or_default();
        let title = if title_attr.is_empty()
            || title_attr.len() < 5
            || title_attr == "Title (No HTML code allowed)"
        {
            // Fallback to filename stem
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("MDict")
                .to_string()
        } else {
            strip_html_like(&title_attr)
        };

        let description_attr = attributes
            .get("Description")
            .cloned()
            .unwrap_or_default();
        let description = strip_html_like(&description_attr);

        // Note: for the lightweight parser we don't yet read block tables or record indices here.
        Ok(MdictHeader {
            encoding,
            version,
            encrypted,
            rtl,
            title,
            description,
            attributes,
            number_size,
            headword_block_info_pos: 0,
            headword_block_info_size: 0,
            num_headword_blocks: 0,
            word_count: 0,
            headword_block_size: 0,
            record_block_info_pos: 0,
            total_records_size: 0,
            record_blocks: Vec::new(),
            file_size: path.metadata().map(|m| m.len()).unwrap_or(0),
        })
    }

    /// Load B-TREE and FTS indexes.
    ///
    /// Safety/Correctness:
    /// - If sidecar files exist but cannot be loaded or validated, we surface an error
    ///   instead of leaving a partially initialized index.
    /// - If on-disk BTreeIndex load is not implemented, the caller receives a clear
    ///   IndexError from the index module.
    fn load_indexes(
        path: &Path,
        config: &DictConfig,
        _header: &MdictHeader,
    ) -> Result<(Option<BTreeIndex>, Option<FtsIndex>)> {
        let mut btree_index = None;
        let mut fts_index = None;

        let base_name = path.with_extension("");
        let btree_path = base_name.with_extension("btree");
        let fts_path = base_name.with_extension("fts");

        // Load B-TREE index if requested and file exists
        if config.load_btree && btree_path.exists() {
            let mut btree = BTreeIndex::new();
            if let Err(e) = btree.load(&btree_path) {
                return Err(DictError::IndexError(format!(
                    "Failed to load MDict B-TREE index {}: {}",
                    btree_path.display(),
                    e
                )));
            }
            if !btree.is_built() {
                return Err(DictError::IndexError(format!(
                    "MDict B-TREE index {} is not built or is empty",
                    btree_path.display()
                )));
            }
            btree_index = Some(btree);
        }

        // Load FTS index if requested and file exists
        if config.load_fts && fts_path.exists() {
            let mut fts = FtsIndex::new();
            if let Err(e) = fts.load(&fts_path) {
                return Err(DictError::IndexError(format!(
                    "Failed to load MDict FTS index {}: {}",
                    fts_path.display(),
                    e
                )));
            }
            if !fts.is_built() {
                return Err(DictError::IndexError(format!(
                    "MDict FTS index {} is not built or is empty",
                    fts_path.display()
                )));
            }
            fts_index = Some(fts);
        }

        Ok((btree_index, fts_index))
    }

    /// Read entry from file using offset
    fn read_entry_at_offset(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        if length as usize > MDICT_MAX_VALUE_LENGTH {
            return Err(DictError::Internal(
                format!("Entry too large: {} bytes", length)
            ));
        }

        let data = if let Some(ref mmap) = self.mmap {
            // Read from memory-mapped file
            let end = offset + length;
            if end > mmap.len() as u64 {
                return Err(DictError::IoError("Read past file end".to_string()));
            }
            mmap[offset as usize..end as usize].to_vec()
        } else if let Some(ref file) = self.file {
            // Read from regular file
            let mut reader = BufReader::new(file);
            reader.seek(SeekFrom::Start(offset))?;
            let mut buffer = vec![0u8; length as usize];
            reader.read_exact(&mut buffer)?;
            buffer
        } else {
            return Err(DictError::Internal("No file handle available".to_string()));
        };

        // MDX block contents can be individually compressed; this helper assumes
        // "length" already reflects the uncompressed size if decompression was
        // applied at index-building time. For the lightweight parser we do not
        // apply additional compression flags from the header here.
        Ok(data)
    }

    /// Binary search for key using B-TREE index
    fn binary_search_lookup(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Some(ref btree) = self.btree_index {
            if let Some((data, _offset)) = btree.binary_search(key)? {
                Ok(Some(data))
            } else {
                Ok(None)
            }
        } else {
            // Fallback to sequential search if no B-TREE index
            self.sequential_search(key)
        }
    }

    /// Sequential search fallback
    fn sequential_search(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        // This would implement a linear search through the dictionary
        // For now, return None as this is inefficient for large dictionaries
        Ok(None)
    }

    /// Cache lookup
    fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        // Use read lock and clone the value if present. lru_cache exposes `get_mut`,
        // so we use a write lock in this helper to keep things simple and sound.
        let mut cache = self.entry_cache.write();
        if let Some(value) = cache.get_mut(&key.to_string()) {
            Some(value.clone())
        } else {
            None
        }
    }

    /// Add to cache
    fn cache_entry(&self, key: String, value: Vec<u8>) {
        let mut cache = self.entry_cache.write();
        cache.insert(key, value);
    }

    /// Build indexes for this MDict based on the current on-disk data.
    ///
    /// This implementation:
    /// - Enumerates all keys from the B-TREE index if present, otherwise returns a clear error.
    /// - Uses `get()` to fetch each entry, ensuring consistency with the existing reader.
    /// - Builds and persists B-TREE and FTS sidecar indexes in a deterministic way.
    pub fn build_indexes(&mut self) -> Result<()> {
        if !self.config.load_btree && !self.config.load_fts {
            return Ok(());
        }

        // We require at least one way to enumerate keys. Since a full MDX parser for
        // key/record blocks is out of scope here, rely on an existing B-TREE index.
        let base_btree = match &self.btree_index {
            Some(idx) => idx,
            None => {
                return Err(DictError::UnsupportedOperation(
                    "MDict index building requires an existing B-TREE index for key enumeration"
                        .to_string(),
                ))
            }
        };

        // Collect entries by iterating over existing B-TREE range.
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        let all = base_btree.range_query("", "\u{10FFFF}")?;
        for (key, _offset) in all {
            // Use public API to obtain value
            if let Ok(value) = self.get(&key) {
                entries.push((key, value));
            }
        }

        if entries.is_empty() {
            return Err(DictError::UnsupportedOperation(
                "MDict index building: no entries could be enumerated from existing index"
                    .to_string(),
            ));
        }

        // Rebuild B-TREE index if requested.
        if self.config.load_btree {
            let mut btree = BTreeIndex::new();
            let index_config = IndexConfig::default();
            btree.build(&entries, &index_config)?;
            if !btree.is_built() {
                return Err(DictError::IndexError(
                    "MDict B-TREE index build produced an empty index".to_string(),
                ));
            }

            let btree_path = self.file_path.with_extension("btree");
            btree.save(&btree_path)?;
            self.btree_index = Some(btree);
        }

        // Build FTS index if requested.
        if self.config.load_fts {
            let mut fts = FtsIndex::new();
            let index_config = IndexConfig::default();
            fts.build(&entries, &index_config)?;
            if !fts.is_built() {
                return Err(DictError::IndexError(
                    "MDict FTS index build produced an empty index".to_string(),
                ));
            }

            let fts_path = self.file_path.with_extension("fts");
            fts.save(&fts_path)?;
            self.fts_index = Some(fts);
        }

        Ok(())
    }

    /// Collect all entries from the dictionary by enumerating keys from the current B-TREE index.
    ///
    /// This helper is used internally for index building; it does not parse raw MDX blocks
    /// but instead reuses the existing `get()` implementation to ensure consistency.
    fn collect_all_entries(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::new();

        if let Some(btree) = &self.btree_index {
            let all = btree.range_query("", "\u{10FFFF}")?;
            for (key, _off) in all {
                if let Ok(val) = self.get(&key) {
                    out.push((key, val));
                }
            }
        }

        Ok(out)
    }

    /// Get file paths for this dictionary
    pub fn file_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = vec![self.file_path.clone()];
        
        if let Some(ref _btree) = self.btree_index {
            paths.push(self.file_path.with_extension("btree"));
        }
        
        if let Some(ref _fts) = self.fts_index {
            paths.push(self.file_path.with_extension("fts"));
        }
        
        paths
    }
}

impl Dict<String> for MDict {
    fn metadata(&self) -> &DictMetadata {
        &self.metadata
    }

    fn contains(&self, key: &String) -> Result<bool> {
        match self.get(key) {
            Ok(_) => Ok(true),
            Err(DictError::IndexError(_)) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    fn get(&self, key: &String) -> Result<Vec<u8>> {
        // Check cache first
        if let Some(cached) = self.get_cached(key) {
            return Ok(cached);
        }

        // Try binary search lookup
        match self.binary_search_lookup(key) {
            Ok(Some(data)) => {
                self.cache_entry(key.clone(), data.clone());
                Ok(data)
            }
            Ok(None) => Err(DictError::IndexError("Key not found".to_string())),
            Err(e) => Err(e),
        }
    }

    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>> {
        let limit = limit.unwrap_or(100);
        let mut results = Vec::new();

        // Use B-TREE for prefix search if available
        if let Some(ref btree) = self.btree_index {
            let range_results = btree.range_query(prefix, &(prefix.to_string() + "\u{10FFFF}"))?;
            
            for (key, _offset) in range_results.iter().take(limit) {
                if key.starts_with(prefix) {
                    match self.get(&key.to_string()) {
                        Ok(entry) => {
                            results.push(SearchResult {
                                word: key.to_string(),
                                entry,
                                score: None,
                                highlights: None,
                            });
                        }
                        Err(_) => continue,
                    }
                }
            }
        } else {
            // Fallback to sequential search
            // This would be implemented for dictionaries without indexes
        }

        Ok(results)
    }

    fn search_fuzzy(&self, query: &str, _max_distance: Option<u32>) -> Result<Vec<SearchResult>> {
        // Fuzzy search implementation would go here
        // For now, return prefix search results
        self.search_prefix(query, None)
    }

    fn search_fulltext(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>> {
        if let Some(ref fts) = self.fts_index {
            // Collect results eagerly into an owned Vec so the iterator can be 'static.
            let search_results = fts.search(query)?;
            let mut items: Vec<Result<SearchResult>> = Vec::with_capacity(search_results.len());

            for (key, score) in search_results {
                match self.get(&key) {
                    Ok(entry) => {
                        items.push(Ok(SearchResult {
                            word: key,
                            entry,
                            score: Some(score),
                            highlights: None,
                        }));
                    }
                    Err(e) => {
                        items.push(Err(e));
                    }
                }
            }

            // Move Vec into a boxed iterator with 'static lifetime.
            Ok(Box::new(items.into_iter()))
        } else {
            Err(DictError::UnsupportedOperation(
                "FTS index not available".to_string()
            ))
        }
    }

    fn get_range(&self, range: std::ops::Range<usize>) -> Result<Vec<(String, Vec<u8>)>> {
        if range.is_empty() {
            return Ok(Vec::new());
        }

        // Use B-TREE index if available to provide deterministic ordering.
        if let Some(ref btree) = self.btree_index {
            let all = btree.range_query("", "\u{10FFFF}")?;
            let slice = if range.start >= all.len() {
                &[]
            } else {
                &all[range.start.min(all.len())..range.end.min(all.len())]
            };

            let mut out = Vec::with_capacity(slice.len());
            for (key, _off) in slice {
                if let Ok(val) = self.get(&key) {
                    out.push((key.clone(), val));
                }
            }
            return Ok(out);
        }

        // Without an index we cannot implement efficient stable range queries.
        Err(DictError::UnsupportedOperation(
            "MDict get_range requires a loaded B-TREE index".to_string(),
        ))
    }

    fn iter(&self) -> Result<EntryIterator<String>> {
        // Get all keys first
        let keys = self.keys()?;
        Ok(EntryIterator {
            keys: keys.into_iter(),
            dictionary: self,
        })
    }

    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(String, Vec<u8>)>> + Send>> {
        let results = self.search_prefix(prefix, None)?;
        let mapped: Vec<_> = results
            .into_iter()
            .map(|sr| Ok((sr.word, sr.entry)))
            .collect();
        Ok(Box::new(mapped.into_iter()))
    }

    fn len(&self) -> usize {
        self.header.word_count as usize
    }

    fn file_paths(&self) -> &[std::path::PathBuf] {
        // This would return the actual file paths
        // For simplicity, return a slice with the main file
        std::slice::from_ref(&self.file_path)
    }

    fn reload_indexes(&mut self) -> Result<()> {
        let config = self.config.clone();
        let (btree_index, fts_index) = Self::load_indexes(&self.file_path, &config, &self.header)?;
        self.btree_index = btree_index;
        self.fts_index = fts_index;
        Ok(())
    }

    fn clear_cache(&mut self) {
        let mut cache = self.entry_cache.write();
        cache.clear();
    }

    fn stats(&self) -> DictStats {
        DictStats {
            total_entries: self.len() as u64,
            cache_hit_rate: 0.0,
            memory_usage: self.header.file_size,
            index_sizes: HashMap::new(),
        }
    }

    fn build_indexes(&mut self) -> Result<()> {
        // Use the concrete index-building helper above.
        MDict::build_indexes(self)
    }
}

 /// Parse the pseudo-XML MDX header attributes into a key-value map.
 /// Minimal implementation matching the needs of this crate.
 fn parse_mdict_header_attributes(header: &str) -> HashMap<String, String> {
     let mut attrs = HashMap::new();

     // Find first '<' ... '>' block
     let start = match header.find('<') {
         Some(s) => s,
         None => return attrs,
     };
     let end = match header[start..].find('>') {
         Some(e) => start + e,
         None => return attrs,
     };
     let elem = &header[start + 1..end]; // strip '<' and '>'

     // Split into tokens, first token is tag name
     let mut iter = elem.split_whitespace();
     iter.next();
     for token in iter {
         let mut parts = token.splitn(2, '=');
         let key = match parts.next() {
             Some(k) if !k.is_empty() => k,
             _ => continue,
         };
         let val_raw = match parts.next() {
             Some(v) => v.trim(),
             None => continue,
         };

         let mut v = val_raw.trim_matches(|c| c == '"' || c == '\'');
         // Some headers contain trailing '/>'; strip trailing '>' if present
         if let Some(stripped) = v.strip_suffix('>') {
             v = stripped.trim();
         }

         attrs.insert(key.to_string(), v.to_string());
     }

     attrs
 }

 /// Strip simple HTML-like tags from a string, used for Title/Description.
 fn strip_html_like(input: &str) -> String {
     let mut out = String::with_capacity(input.len());
     let mut in_tag = false;
     for c in input.chars() {
         match c {
             '<' => in_tag = true,
             '>' => in_tag = false,
             _ if !in_tag => out.push(c),
             _ => {}
         }
     }
     out.trim().to_string()
 }

impl HighPerformanceDict<String> for MDict {
    fn binary_search_get(&self, key: &String) -> Result<Vec<u8>> {
        self.get(key)
    }

    #[cfg(feature = "rayon")]
    fn parallel_get_multiple(&self, keys: &[String]) -> Result<Vec<BatchResult>> {
        use rayon::prelude::*;

        let dict = self;
        let results: Vec<BatchResult> = keys
            .par_iter()
            .map(|key| {
                match dict.get(key) {
                    Ok(entry) => BatchResult {
                        word: key.clone(),
                        entry: Some(entry),
                        error: None,
                    },
                    Err(e) => BatchResult {
                        word: key.clone(),
                        entry: None,
                        error: Some(e),
                    },
                }
            })
            .collect();

        Ok(results)
    }

    fn stream_search(&self, _query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>> {
        Err(DictError::UnsupportedOperation("Stream search not implemented".to_string()))
    }
}
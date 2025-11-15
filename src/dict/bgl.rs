//! BGL (Babylon) dictionary format implementation
//!
//! This is a minimal, index-consuming implementation intended to integrate
//! Babylon `.bgl` dictionaries into the existing traits without changing
//! core traits, BTree, or FTS modules.
//!
//! Design constraints:
//! - Only parsing/format wiring here; no modifications to shared traits/indexes.
//! - We do NOT implement raw .bgl binary parsing in this crate.
//! - We assume an external tool (like GoldenDict's indexer) provides:
//!   * A sidecar BTree index (.btree) mapping headword -> article offset
//!   * An optional FTS index (.fts)
//!   * An index/chunks layout where an offset points to
//!     [headword\0][displayed\0][body\0] inside an index/chunks file.
//!
//! This module wires BglDict into the module tree and makes it compile-safe.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::index::btree::BTreeIndex;
use crate::index::fts::FtsIndex;
use crate::index::Index;
use crate::traits::{
    Dict, DictConfig, DictError, DictFormat, DictMetadata, DictStats, EntryIterator,
    HighPerformanceDict, Result, SearchResult,
};

/// Minimal BGL "index header" used only for metadata and chunks base offset.
///
/// NOTE:
/// - This is intentionally simplified and does NOT try to be fully compatible
///   with GoldenDict's internal BGLX format.
/// - External tooling is expected to emit something compatible with this
///   interpretation if you want to use BglDict directly.
#[derive(Debug, Clone)]
struct BglIndexHeader {
    /// Magic signature, expected "BGLX"
    signature: [u8; 4],
    /// Format version (opaque here)
    format_version: u32,
    /// Number of articles
    article_count: u32,
    /// Number of words (for metadata only)
    word_count: u32,
    /// Offset to chunked article storage in index file
    chunks_offset: u64,
}

impl BglIndexHeader {
    fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 24 {
            return Err(DictError::InvalidFormat(
                "BGL index header too small".to_string(),
            ));
        }

        let mut sig = [0u8; 4];
        sig.copy_from_slice(&buf[0..4]);

        if &sig != b"BGLX" {
            return Err(DictError::InvalidFormat(format!(
                "Invalid BGL index signature: {:?}",
                sig
            )));
        }

        let format_version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let article_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let word_count = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let chunks_offset = u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]);

        Ok(Self {
            signature: sig,
            format_version,
            article_count,
            word_count,
            chunks_offset,
        })
    }
}

/// Lightweight BGL dictionary backed by sidecar indexes.
///
/// Expectations:
/// - Main: `*.bgl` (original Babylon data, not parsed directly here).
/// - Sidecar:
///   - `*.bglx` or `*.idx`: contains BglIndexHeader + chunked entries.
///   - `*.btree`: serialized BTreeIndex (key -> offset in chunks).
///   - `*.fts`: serialized FtsIndex (optional).
pub struct BglDict {
    /// Original BGL file path
    bgl_path: PathBuf,
    /// Index/chunks file path (`.bglx` / `.idx`)
    index_path: PathBuf,
    /// Parsed header from index (for metadata/chunks_offset)
    header: BglIndexHeader,
    /// BTree-based index for key lookups
    btree_index: Option<BTreeIndex>,
    /// Full-text search index (optional)
    fts_index: Option<FtsIndex>,
    /// Cache for entries
    cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>,
    /// Configuration
    config: DictConfig,
    /// Metadata
    metadata: DictMetadata,
}

impl BglDict {
    /// Create a BglDict using an existing BGL file and compatible sidecar index files.
    pub fn new<P: AsRef<Path>>(path: P, mut config: DictConfig) -> Result<Self> {
        let bgl_path = path.as_ref().to_path_buf();

        if !bgl_path.exists() {
            return Err(DictError::FileNotFound(bgl_path.display().to_string()));
        }

        // Resolve index path: prefer `.bglx`, fallback to `.idx`
        let idx_bglx = bgl_path.with_extension("bglx");
        let idx_idx = bgl_path.with_extension("idx");
        let index_path = if idx_bglx.exists() {
            idx_bglx
        } else if idx_idx.exists() {
            idx_idx
        } else {
            return Err(DictError::UnsupportedOperation(format!(
                "No BGL index found for {} (expected .bglx or .idx)",
                bgl_path.display()
            )));
        };

        // Read minimal header from index file
        let mut index_file = File::open(&index_path)
            .map_err(|e| DictError::IoError(format!("open BGL index: {e}")))?;
        let mut hdr_buf = [0u8; 24];
        index_file
            .read_exact(&mut hdr_buf)
            .map_err(|e| DictError::IoError(format!("read BGL index header: {e}")))?;
        let header = BglIndexHeader::parse(&hdr_buf)?;

        // Load sidecar BTree / FTS indexes
        let (btree_index, fts_index) = Self::load_sidecar_indexes(&bgl_path, &config)?;

        // Metadata
        let file_size = bgl_path.metadata().map(|m| m.len()).unwrap_or(0);
        let name = bgl_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("BGL")
            .to_string();

        let metadata = DictMetadata {
            name,
            version: format!("{}", header.format_version),
            entries: header.article_count as u64,
            description: None,
            author: None,
            language: None,
            file_size,
            created: None,
            has_btree: btree_index.is_some(),
            has_fts: fts_index.is_some(),
        };

        if config.cache_size == 0 {
            config.cache_size = 1024;
        }

        Ok(Self {
            bgl_path,
            index_path,
            header,
            btree_index,
            fts_index,
            cache: Arc::new(RwLock::new(lru_cache::LruCache::new(config.cache_size))),
            config,
            metadata,
        })
    }

    /// Load sidecar BTree / FTS indexes if present.
    fn load_sidecar_indexes(
        bgl_path: &Path,
        config: &DictConfig,
    ) -> Result<(Option<BTreeIndex>, Option<FtsIndex>)> {
        let stem = bgl_path.with_extension("");
        let idx_btree = stem.with_extension("btree");
        let idx_fts = stem.with_extension("fts");

        let mut btree_index = None;
        let mut fts_index = None;

        if config.load_btree && idx_btree.exists() {
            let mut idx = BTreeIndex::new();
            if let Err(e) = idx.load(&idx_btree) {
                return Err(DictError::IndexError(format!(
                    "Failed to load BGL B-TREE index {}: {}",
                    idx_btree.display(),
                    e
                )));
            }
            if !idx.is_built() {
                return Err(DictError::IndexError(format!(
                    "BGL B-TREE index {} is not built or is empty",
                    idx_btree.display()
                )));
            }
            btree_index = Some(idx);
        }

        if config.load_fts && idx_fts.exists() {
            let mut idx = FtsIndex::new();
            if let Err(e) = idx.load(&idx_fts) {
                return Err(DictError::IndexError(format!(
                    "Failed to load BGL FTS index {}: {}",
                    idx_fts.display(),
                    e
                )));
            }
            if !idx.is_built() {
                return Err(DictError::IndexError(format!(
                    "BGL FTS index {} is not built or is empty",
                    idx_fts.display()
                )));
            }
            fts_index = Some(idx);
        }

        Ok((btree_index, fts_index))
    }

    /// Read article triple [headword\0][displayed\0][body\0] from the index file.
    ///
    /// Assumptions:
    /// - `offset` is relative to `header.chunks_offset`.
    /// - Data is small enough to read in a single chunk.
    fn read_article_at(&self, offset: u64) -> Result<(String, String, Vec<u8>)> {
        let base = self
            .header
            .chunks_offset
            .checked_add(offset)
            .ok_or_else(|| DictError::InvalidFormat("BGL article offset overflow".to_string()))?;

        let mut f = File::open(&self.index_path)
            .map_err(|e| DictError::IoError(format!("open BGL index for article: {e}")))?;
        f.seek(SeekFrom::Start(base))
            .map_err(|e| DictError::IoError(format!("seek BGL article: {e}")))?;

        // Read a bounded buffer; real format should encode exact length.
        let mut buf = vec![0u8; 64 * 1024];
        let n = f
            .read(&mut buf)
            .map_err(|e| DictError::IoError(format!("read BGL article: {e}")))?;
        buf.truncate(n);

        let mut parts = buf.split(|b| *b == 0u8);

        let head = parts
            .next()
            .ok_or_else(|| DictError::InvalidFormat("Missing BGL headword".to_string()))?;
        let disp = parts.next().ok_or_else(|| {
            DictError::InvalidFormat("Missing BGL displayed headword".to_string())
        })?;
        let rest = parts.next().unwrap_or(&[]);

        let headword = String::from_utf8_lossy(head).to_string();
        let displayed = String::from_utf8_lossy(disp).to_string();
        let body = rest.to_vec();

        Ok((headword, displayed, body))
    }

    fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache.write();
        cache.get_mut(&key.to_string()).map(|v| v.clone())
    }

    fn cache_put(&self, key: String, val: Vec<u8>) {
        let mut cache = self.cache.write();
        cache.insert(key, val);
    }
}

impl Dict<String> for BglDict {
    fn metadata(&self) -> &DictMetadata {
        &self.metadata
    }

    fn contains(&self, key: &String) -> Result<bool> {
        if let Some(btree) = &self.btree_index {
            Ok(btree.search(key)?.is_some())
        } else {
            Ok(false)
        }
    }

    fn get(&self, key: &String) -> Result<Vec<u8>> {
        if let Some(cached) = self.get_cached(key) {
            return Ok(cached);
        }

        let offset = if let Some(btree) = &self.btree_index {
            match btree.search(key)? {
                Some((_k, off)) => off,
                None => {
                    return Err(DictError::IndexError(format!(
                        "Key not found in BGL sidecar index: {}",
                        key
                    )))
                }
            }
        } else {
            return Err(DictError::UnsupportedOperation(
                "BGL lookup requires BTree index; none loaded".to_string(),
            ));
        };

        let (_head, _disp, body) = self.read_article_at(offset)?;
        self.cache_put(key.clone(), body.clone());
        Ok(body)
    }

    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let max = limit.unwrap_or(64);

        if let Some(btree) = &self.btree_index {
            let start = prefix.to_string();
            let end = format!("{}\u{10FFFF}", prefix);
            let range = btree.range_query(&start, &end)?;

            for (word, off) in range.into_iter().take(max) {
                let (_head, _disp, body) = self.read_article_at(off)?;
                results.push(SearchResult {
                    word,
                    entry: body,
                    score: None,
                    highlights: None,
                });
            }
        }

        Ok(results)
    }

    fn search_fuzzy(&self, query: &str, _max_distance: Option<u32>) -> Result<Vec<SearchResult>> {
        // Minimal approximation: reuse prefix search
        self.search_prefix(query, Some(64))
    }

    fn search_fulltext(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>> {
        if let Some(fts) = &self.fts_index {
            let hits = fts.search(query)?;
            let mut out = Vec::new();
            for (word, score) in hits {
                if let Ok(body) = self.get(&word) {
                    out.push(SearchResult {
                        word,
                        entry: body,
                        score: Some(score),
                        highlights: None,
                    });
                }
            }
            Ok(Box::new(out.into_iter().map(Ok)))
        } else {
            // Fallback to prefix search
            let res = self.search_prefix(query, Some(64))?;
            Ok(Box::new(res.into_iter().map(Ok)))
        }
    }

    fn get_range(&self, range: std::ops::Range<usize>) -> Result<Vec<(String, Vec<u8>)>> {
        if range.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(ref btree) = self.btree_index {
            let all = btree.range_query("", "\u{10FFFF}")?;
            if all.is_empty() || range.start >= all.len() {
                return Ok(Vec::new());
            }

            let slice = &all[range.start.min(all.len())..range.end.min(all.len())];
            let mut out = Vec::with_capacity(slice.len());
            for (word, off) in slice.iter() {
                if let Ok((_h, _d, body)) = self.read_article_at(*off) {
                    out.push((word.clone(), body));
                }
            }
            Ok(out)
        } else {
            Err(DictError::UnsupportedOperation(
                "BGL get_range requires a loaded BTree index".to_string(),
            ))
        }
    }

    fn iter(&self) -> Result<EntryIterator<'_, String>> {
        if self.btree_index.is_none() {
            return Err(DictError::UnsupportedOperation(
                "BGL iter requires a loaded BTree index".to_string(),
            ));
        }

        // Materialize keys from BTree for a deterministic iteration order.
        let btree = self
            .btree_index
            .as_ref()
            .ok_or_else(|| DictError::Internal("BGL BTree index missing".to_string()))?;
        let all = btree.range_query("", "\u{10FFFF}")?;
        let keys: Vec<String> = all.into_iter().map(|(k, _)| k).collect();

        Ok(EntryIterator {
            keys: keys.into_iter(),
            dictionary: self,
        })
    }

    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(String, Vec<u8>)>> + Send>> {
        let hits = self.search_prefix(prefix, Some(256))?;
        let mapped: Vec<_> = hits.into_iter().map(|sr| Ok((sr.word, sr.entry))).collect();
        Ok(Box::new(mapped.into_iter()))
    }

    fn len(&self) -> usize {
        self.header.article_count as usize
    }

    fn file_paths(&self) -> &[PathBuf] {
        std::slice::from_ref(&self.bgl_path)
    }

    fn reload_indexes(&mut self) -> Result<()> {
        let (bt, fts) = Self::load_sidecar_indexes(&self.bgl_path, &self.config)?;
        self.btree_index = bt;
        self.fts_index = fts;
        Ok(())
    }

    fn clear_cache(&mut self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    fn stats(&self) -> DictStats {
        DictStats {
            total_entries: self.header.article_count as u64,
            cache_hit_rate: 0.0,
            memory_usage: 0,
            index_sizes: HashMap::new(),
        }
    }

    fn build_indexes(&mut self) -> Result<()> {
        // Build an in-memory FTS index when we have a BTree index and can read articles.
        //
        // This stays within the existing design:
        // - Requires an already loaded BTree sidecar (same as GoldenDict's BGL indexer output).
        // - Does not modify on-disk files.
        // - Streams articles via read_article_at() to avoid unnecessary memory usage.
        //
        // Semantics:
        // - Use BTree keys as headwords.
        // - Use [body] as fulltext content.
        // - Keep BTree index as-is; only (re)build FTS index in memory.

        let btree = match &self.btree_index {
            Some(b) => b,
            None => {
                return Err(DictError::UnsupportedOperation(
                    "BGL build_indexes requires a loaded BTree index".to_string(),
                ))
            }
        };

        let all = btree.range_query("", "\u{10FFFF}")?;
        if all.is_empty() {
            return Err(DictError::IndexError(
                "BGL build_indexes: base BTree index is empty".to_string(),
            ));
        }

        // Collect entries for FTS. To bound memory, we only store (word, body)
        // pairs once; callers that need more advanced policies can adjust
        // IndexConfig in the future.
        let mut fts_entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(all.len());
        for (word, off) in &all {
            let (_head, _disp, body) = self.read_article_at(*off)?;
            fts_entries.push((word.clone(), body));
        }

        let mut fts = FtsIndex::new();
        let cfg = crate::index::IndexConfig::default();
        fts.build(&fts_entries, &cfg)?;
        if !fts.is_built() {
            return Err(DictError::IndexError(
                "BGL FTS index build produced an empty index".to_string(),
            ));
        }

        self.fts_index = Some(fts);
        Ok(())
    }
}

impl HighPerformanceDict<String> for BglDict {
    fn binary_search_get(&self, key: &String) -> Result<Vec<u8>> {
        self.get(key)
    }

    fn stream_search(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>> {
        // Erase Send bound by using the same underlying iterator; this matches trait signature.
        let it = self.search_fulltext(query)?;
        Ok(Box::new(it))
    }
}

/// DictFormat implementation for BGL.
///
/// Detection is conservative:
/// - Extension `.bgl`
/// - Requires a corresponding `.bglx` or `.idx` sidecar index file.
impl DictFormat<String> for BglDict {
    const FORMAT_NAME: &'static str = "bgl";
    const FORMAT_VERSION: &'static str = "1.0";

    fn is_valid_format(path: &Path) -> Result<bool> {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("bgl") {
                let idx_bglx = path.with_extension("bglx");
                let idx_idx = path.with_extension("idx");
                return Ok(idx_bglx.exists() || idx_idx.exists());
            }
        }
        Ok(false)
    }

    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let dict = BglDict::new(path, config)?;
        Ok(Box::new(dict))
    }
}

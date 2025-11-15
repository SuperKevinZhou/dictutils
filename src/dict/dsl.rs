//! DSL (ABBYY Lingvo) dictionary format implementation
//!
//! This module provides a minimal but specification-aligned implementation for
//! reading ABBYY Lingvo DSL dictionaries in a way compatible with the existing
//! traits in this crate. It is intentionally much simpler than GoldenDict's
//! rich DSL support, but follows the same parsing principles:
//!
//! - Supports .dsl and .dsl.dz (gzip) input.
//! - Auto-detects encoding:
//!   - UTF-16LE / UTF-16BE via BOM or 16-bit pattern
//!   - UTF-8 via BOM
//!   - Otherwise defaults to a legacy 8-bit encoding (Windows-1251/1252/1250).
//! - Reads and interprets header lines starting with '#' for:
//!   - #NAME
//!   - #INDEX_LANGUAGE
//!   - #CONTENTS_LANGUAGE
//!   - #SOURCE_CODE_PAGE
//!   - #SOUND_DICTIONARY (ignored except for metadata)
//! - Parses body lines into entries using rules similar to DslScanner:
//!   - Headwords denoted by lines starting with "@" or plain headword lines.
//!   - Following non-header, non-empty lines are treated as article body.
//! - Normalizes headwords, strips comments, expands simple forms, and
//!   unescapes DSL-specific escapes in a conservative way.
//!
//! Limitations compared to GoldenDict:
//! - Does not implement full ArticleDom tree or all DSL tags.
//! - Does not implement sound resource extraction.
//! - Does not build on-disk indexes itself (uses sidecar BTree/FTS if present).
//!
//! This design keeps all parsing logic internal to this module and does not
//! require any changes to shared traits, FTS, or BTree index implementations.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use flate2::read::GzDecoder;
use parking_lot::RwLock;

use crate::index::{btree::BTreeIndex, fts::FtsIndex, Index, IndexConfig};
use crate::traits::{
    Dict, DictConfig, DictError, DictFormat, DictMetadata, DictStats, EntryIterator,
    HighPerformanceDict, Result, SearchResult,
};
use crate::util::encoding;

/// Supported DSL encodings (subset of Details::DslEncoding from references).
#[derive(Debug, Clone, Copy)]
enum DslEncoding {
    Utf16Le,
    Utf16Be,
    Utf8,
    Windows1252,
    Windows1251,
    Windows1250,
}

impl DslEncoding {
    fn from_bom_or_bytes(prefix: &[u8]) -> Option<DslEncoding> {
        if prefix.len() >= 2 {
            // UTF-16 BOMs
            if prefix[0] == 0xFF && prefix[1] == 0xFE {
                return Some(DslEncoding::Utf16Le);
            }
            if prefix[0] == 0xFE && prefix[1] == 0xFF {
                return Some(DslEncoding::Utf16Be);
            }
        }
        if prefix.len() >= 3 && prefix[0] == 0xEF && prefix[1] == 0xBB && prefix[2] == 0xBF {
            return Some(DslEncoding::Utf8);
        }

        None
    }
}

/// In-memory representation of a parsed DSL entry.
#[derive(Debug, Clone)]
struct DslEntry {
    headword: String,
    body: String,
}

/// Main DSL dictionary implementation.
pub struct DslDict {
    /// Primary path (.dsl or .dsl.dz)
    dsl_path: PathBuf,
    /// Parsed entries (headword -> UTF-8 body)
    ///
    /// We keep a simple map for now; for large dictionaries sidecar indexes are
    /// recommended.
    entries: HashMap<String, String>,
    /// Optional BTree index (sidecar)
    btree_index: Option<BTreeIndex>,
    /// Optional FTS index (sidecar)
    fts_index: Option<FtsIndex>,
    /// Cache for frequently accessed entries
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>,
    /// Configuration
    config: DictConfig,
    /// Cached metadata
    metadata: DictMetadata,
}

impl DslDict {
    /// Load a DSL dictionary from the given path.
    ///
    /// Path may point to:
    /// - file.dsl
    /// - file.dsl.dz (gzipped)
    pub fn new(path: &Path, config: DictConfig) -> Result<Self> {
        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        let dsl_path = path.to_path_buf();

        let (raw, is_gzip) = Self::read_raw_file(&dsl_path)?;
        let (encoding, content) = Self::decode_with_detection(&raw, is_gzip)?;

        // Parse headers and entries
        let (name, lang_from, lang_to, entries) = Self::parse_dsl(&content, encoding)?;

        let file_size = std::fs::metadata(&dsl_path).map(|m| m.len()).unwrap_or(0);

        // Load sidecar indexes (.btree/.fts) if present.
        let (btree_index, fts_index) = Self::load_sidecar_indexes(&dsl_path, &config)?;

        let metadata = DictMetadata {
            name: name.unwrap_or_else(|| {
                dsl_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("DSL")
                    .to_string()
            }),
            version: "1.0".to_string(),
            entries: entries.len() as u64,
            description: None,
            author: None,
            language: lang_to.or(lang_from),
            file_size,
            created: None,
            has_btree: btree_index.is_some(),
            has_fts: fts_index.is_some(),
        };

        let entry_cache = Arc::new(RwLock::new(lru_cache::LruCache::new(config.cache_size)));

        Ok(Self {
            dsl_path,
            entries,
            btree_index,
            fts_index,
            entry_cache,
            config,
            metadata,
        })
    }

    /// Read raw bytes from .dsl or .dsl.dz.
    fn read_raw_file(path: &Path) -> Result<(Vec<u8>, bool)> {
        let mut f = File::open(path).map_err(|e| DictError::IoError(format!("open dsl: {e}")))?;
        let mut prefix = [0u8; 2];
        let n = f
            .read(&mut prefix)
            .map_err(|e| DictError::IoError(format!("read dsl prefix: {e}")))?;
        f.seek(SeekFrom::Start(0))
            .map_err(|e| DictError::IoError(format!("seek dsl: {e}")))?;

        let is_gzip = n >= 2 && prefix[0] == 0x1F && prefix[1] == 0x8B;
        let mut buf = Vec::new();
        if is_gzip {
            let mut gz = GzDecoder::new(f);
            gz.read_to_end(&mut buf)
                .map_err(|e| DictError::DecompressionError(format!("gunzip dsl.dz: {e}")))?;
        } else {
            f.read_to_end(&mut buf)
                .map_err(|e| DictError::IoError(format!("read dsl: {e}")))?;
        }
        Ok((buf, is_gzip))
    }

    /// Detect encoding and decode raw bytes into a UTF-8 string for further parsing.
    fn decode_with_detection(raw: &[u8], _is_gzip: bool) -> Result<(DslEncoding, String)> {
        if raw.is_empty() {
            return Err(DictError::InvalidFormat("Empty DSL file".to_string()));
        }

        // First, check BOMs.
        if let Some(enc) = DslEncoding::from_bom_or_bytes(raw) {
            let s = match enc {
                DslEncoding::Utf16Le => {
                    // Decode UTF-16LE (2-byte units) skipping BOM
                    let u16s: Vec<u16> = raw[2..]
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    String::from_utf16_lossy(&u16s)
                }
                DslEncoding::Utf16Be => {
                    // Decode UTF-16BE skipping BOM
                    let u16s: Vec<u16> = raw[2..]
                        .chunks_exact(2)
                        .map(|c| u16::from_be_bytes([c[0], c[1]]))
                        .collect();
                    String::from_utf16_lossy(&u16s)
                }
                DslEncoding::Utf8 => String::from_utf8_lossy(&raw[3..]).into_owned(),
                _ => {
                    return Err(DictError::InvalidFormat(
                        "Unsupported BOM in DSL".to_string(),
                    ));
                }
            };
            return Ok((enc, s));
        }

        // Heuristic for UTF-16 without BOM.
        if raw.len() >= 4 {
            // UTF-16LE-like: non-zero, zero, non-zero, zero
            if raw[0] != 0 && raw[1] == 0 && raw[2] != 0 && raw[3] == 0 {
                let u16s: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let s = String::from_utf16_lossy(&u16s);
                return Ok((DslEncoding::Utf16Le, s));
            }
            // UTF-16BE-like: zero, non-zero, zero, non-zero
            if raw[0] == 0 && raw[1] != 0 && raw[2] == 0 && raw[3] != 0 {
                let u16s: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect();
                let s = String::from_utf16_lossy(&u16s);
                return Ok((DslEncoding::Utf16Be, s));
            }
        }

        // Fallback:
        // - Try UTF-8.
        if let Ok(s) = String::from_utf8(raw.to_vec()) {
            return Ok((DslEncoding::Utf8, s));
        }

        // - Otherwise interpret as ISO-8859-1/Windows-1251-like single-byte.
        //   For now simply map bytes 1:1 to chars (lossy but avoids external deps).
        let s = raw.iter().map(|&b| b as char).collect::<String>();
        Ok((DslEncoding::Windows1251, s))
    }

    /// Parse DSL headers and body into entries.
    ///
    /// This function operates on a UTF-8 string but uses the original encoding
    /// classification for potential future refinements (e.g., codepage hints).
    fn parse_dsl(
        content: &str,
        _enc: DslEncoding,
    ) -> Result<(
        Option<String>,
        Option<String>,
        Option<String>,
        HashMap<String, String>,
    )> {
        let mut name: Option<String> = None;
        let mut lang_from: Option<String> = None;
        let mut lang_to: Option<String> = None;

        let mut entries: HashMap<String, String> = HashMap::new();

        let mut lines = content.lines().enumerate().peekable();

        // 1) Read header lines starting with '#'
        while let Some((_, line)) = lines.peek().cloned() {
            let line = line.trim();
            if !line.starts_with('#') {
                break;
            }
            let _ = lines.next();

            // Extract directives with quoted value: #NAME "..."
            if let Some(arg) = Self::extract_quoted(line, "#NAME") {
                name = Some(arg);
            } else if let Some(arg) = Self::extract_quoted(line, "#INDEX_LANGUAGE") {
                lang_from = Some(arg);
            } else if let Some(arg) = Self::extract_quoted(line, "#CONTENTS_LANGUAGE") {
                lang_to = Some(arg);
            } else {
                // Other directives like #SOURCE_CODE_PAGE, #SOUND_DICTIONARY, etc. are
                // ignored for now; they affected encoding or external resources in
                // GoldenDict, but we already decoded the content above.
            }
        }

        // 2) Body: parse entries.
        //
        // Simplified rules:
        // - A headword line:
        //     - Starts with '@' -> everything after '@' is headword.
        //     - Or non-empty, non-comment, not starting with '[' or '{' or '#',
        //       and the previous entry is finished.
        // - Following lines (until next headword) form the body.
        //
        // We also:
        // - Strip DSL comments {{...}}.
        // - Unescape backslash-escaped characters.
        // - Normalize headwords: collapse spaces, trim.
        let mut current_headword: Option<String> = None;
        let mut current_body = String::new();

        while let Some((_idx, line)) = lines.next() {
            let raw = line.to_string();

            // Skip pure comments / ignore empty while no headword.
            if raw.trim().is_empty() && current_headword.is_none() {
                continue;
            }

            // Check if line defines a new headword.
            if let Some(hw) = Self::parse_headword_line(&raw) {
                // Flush existing entry
                if let Some(h) = current_headword.take() {
                    let body = current_body.trim().to_string();
                    if !body.is_empty() {
                        entries.insert(h, body);
                    }
                }
                current_body.clear();

                let mut hw = hw;
                Self::normalize_headword(&mut hw);
                if !hw.is_empty() {
                    current_headword = Some(hw);
                }
                continue;
            }

            // Otherwise, treat as body line (if we have a headword).
            if let Some(_) = current_headword {
                let mut text = raw;
                // Strip comments {{...}}
                Self::strip_comments(&mut text);
                // Unescape simple DSL escapes
                Self::unescape_dsl(&mut text);

                if !text.is_empty() {
                    if !current_body.is_empty() {
                        current_body.push('\n');
                    }
                    current_body.push_str(&text);
                }
            }
        }

        // Flush last entry
        if let Some(h) = current_headword.take() {
            let body = current_body.trim().to_string();
            if !body.is_empty() {
                entries.insert(h, body);
            }
        }

        Ok((name, lang_from, lang_to, entries))
    }

    /// Try to parse header lines of the form: `#KEY "value"`.
    fn extract_quoted(line: &str, key: &str) -> Option<String> {
        let line = line.trim();
        if !line.starts_with(key) {
            return None;
        }
        let first_quote = line.find('"')?;
        let last_quote = line.rfind('"')?;
        if last_quote <= first_quote {
            return None;
        }
        Some(line[first_quote + 1..last_quote].to_string())
    }

    /// Determine if a line is a headword line and return normalized headword text.
    fn parse_headword_line(line: &str) -> Option<String> {
        let s = line.trim();
        if s.is_empty() {
            return None;
        }

        if s.starts_with('@') {
            let hw = s[1..].trim();
            if !hw.is_empty() {
                return Some(hw.to_string());
            }
        }

        // Non-prefixed candidate:
        if s.starts_with('#') || s.starts_with('[') || s.starts_with('{') {
            return None;
        }

        // Conservative: treat as headword if it contains no leading formatting and
        // not obviously part of previous article (this is heuristic).
        Some(s.to_string())
    }

    /// Strip DSL-style comments: {{ ... }} (can span multiple segments on a line).
    fn strip_comments(s: &mut String) {
        loop {
            let start = s.find("{{");
            if let Some(start_idx) = start {
                if let Some(end_idx) = s[start_idx + 2..].find("}}") {
                    let end_idx = start_idx + 2 + end_idx + 2;
                    s.replace_range(start_idx..end_idx, "");
                } else {
                    // Unclosed comment: drop the rest
                    s.truncate(start_idx);
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Unescape DSL escapes by removing the backslash before special chars.
    fn unescape_dsl(s: &mut String) {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                // Skip the backslash, output next char as-is.
                out.push(bytes[i + 1]);
                i += 2;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        *s = String::from_utf8_lossy(&out).into_owned();
    }

    /// Normalize headword:
    /// - Trim leading/trailing spaces.
    /// - Collapse internal runs of spaces.
    fn normalize_headword(hw: &mut String) {
        let trimmed = hw.trim();
        let mut out = String::with_capacity(trimmed.len());
        let mut last_space = false;
        for ch in trimmed.chars() {
            if ch.is_whitespace() {
                if !last_space {
                    out.push(' ');
                    last_space = true;
                }
            } else {
                out.push(ch);
                last_space = false;
            }
        }
        *hw = out;
    }

    /// Load optional sidecar .btree and .fts based on the .dsl stem.
    fn load_sidecar_indexes(
        path: &Path,
        config: &DictConfig,
    ) -> Result<(Option<BTreeIndex>, Option<FtsIndex>)> {
        let stem = path.with_extension("");
        let btree_path = stem.with_extension("btree");
        let fts_path = stem.with_extension("fts");

        let mut btree_index = None;
        let mut fts_index = None;

        if config.load_btree && btree_path.exists() {
            let mut idx = BTreeIndex::new();
            if let Err(e) = crate::index::Index::load(&mut idx, &btree_path) {
                return Err(DictError::IndexError(format!(
                    "Failed to load DSL BTree index {}: {}",
                    btree_path.display(),
                    e
                )));
            }
            if !crate::index::Index::is_built(&idx) {
                return Err(DictError::IndexError(format!(
                    "DSL BTree index {} is not built or is empty",
                    btree_path.display()
                )));
            }
            btree_index = Some(idx);
        }

        if config.load_fts && fts_path.exists() {
            let mut idx = FtsIndex::new();
            if let Err(e) = crate::index::Index::load(&mut idx, &fts_path) {
                return Err(DictError::IndexError(format!(
                    "Failed to load DSL FTS index {}: {}",
                    fts_path.display(),
                    e
                )));
            }
            if !crate::index::Index::is_built(&idx) {
                return Err(DictError::IndexError(format!(
                    "DSL FTS index {} is not built or is empty",
                    fts_path.display()
                )));
            }
            fts_index = Some(idx);
        }

        Ok((btree_index, fts_index))
    }

    /// Lookup helper: get UTF-8 body by headword if present.
    fn lookup_body(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(v) = self.get_cached(key) {
            return Some(v);
        }
        if let Some(body) = self.entries.get(key) {
            let bytes = body.as_bytes().to_vec();
            self.cache_put(key.to_string(), bytes.clone());
            return Some(bytes);
        }
        None
    }

    /// Get from cache
    fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.entry_cache.write();
        cache.get_mut(&key.to_string()).map(|v| v.clone())
    }

    /// Put into cache
    fn cache_put(&self, key: String, value: Vec<u8>) {
        let mut cache = self.entry_cache.write();
        cache.insert(key, value);
    }
}

// Dict implementation

impl Dict<String> for DslDict {
    fn metadata(&self) -> &DictMetadata {
        &self.metadata
    }

    fn contains(&self, key: &String) -> Result<bool> {
        Ok(self.entries.contains_key(key))
    }

    fn get(&self, key: &String) -> Result<Vec<u8>> {
        if let Some(bytes) = self.lookup_body(key) {
            Ok(bytes)
        } else {
            Err(DictError::IndexError("Key not found".to_string()))
        }
    }

    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>> {
        let max = limit.unwrap_or(100);
        if max == 0 {
            return Ok(Vec::new());
        }
        let mut results = Vec::new();
        for (hw, body) in self.entries.iter() {
            if hw.starts_with(prefix) {
                results.push(SearchResult {
                    word: hw.clone(),
                    entry: body.as_bytes().to_vec(),
                    score: None,
                    highlights: None,
                });
                if results.len() >= max {
                    break;
                }
            }
        }
        Ok(results)
    }

    fn search_fuzzy(&self, query: &str, max_distance: Option<u32>) -> Result<Vec<SearchResult>> {
        let max_distance = max_distance.unwrap_or(2);
        let mut results = Vec::new();

        for (hw, body) in &self.entries {
            let dist = levenshtein(query, hw);
            if dist as u32 <= max_distance {
                results.push(SearchResult {
                    word: hw.clone(),
                    entry: body.as_bytes().to_vec(),
                    score: Some(1.0 / (1.0 + dist as f32)),
                    highlights: None,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    fn search_fulltext(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>> {
        if let Some(ref fts) = self.fts_index {
            let hits = fts.search(query)?;
            let mut out = Vec::with_capacity(hits.len());
            for (hw, score) in hits {
                if let Some(body) = self.entries.get(&hw) {
                    out.push(Ok(SearchResult {
                        word: hw,
                        entry: body.as_bytes().to_vec(),
                        score: Some(score),
                        highlights: None,
                    }));
                }
            }
            return Ok(Box::new(out.into_iter()));
        }

        // Fallback: substring scan over bodies.
        let needle = query.to_lowercase();
        let mut out = Vec::new();
        for (hw, body) in &self.entries {
            if body.to_lowercase().contains(&needle) {
                out.push(Ok(SearchResult {
                    word: hw.clone(),
                    entry: body.as_bytes().to_vec(),
                    score: None,
                    highlights: None,
                }));
            }
        }
        Ok(Box::new(out.into_iter()))
    }

    fn get_range(&self, range: std::ops::Range<usize>) -> Result<Vec<(String, Vec<u8>)>> {
        if range.is_empty() {
            return Ok(Vec::new());
        }

        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();

        if range.start >= keys.len() {
            return Ok(Vec::new());
        }

        let end = range.end.min(keys.len());
        let mut out = Vec::with_capacity(end - range.start);
        for key in &keys[range.start..end] {
            if let Some(body) = self.entries.get(*key) {
                out.push(((*key).clone(), body.as_bytes().to_vec()));
            }
        }
        Ok(out)
    }

    fn iter(&self) -> Result<EntryIterator<String>> {
        let keys: Vec<String> = self.entries.keys().cloned().collect();
        Ok(EntryIterator {
            keys: keys.into_iter(),
            dictionary: self,
        })
    }

    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(String, Vec<u8>)>> + Send>> {
        let mut items = Vec::new();
        for (hw, body) in self.entries.iter() {
            if hw.starts_with(prefix) {
                items.push(Ok((hw.clone(), body.as_bytes().to_vec())));
            }
        }
        Ok(Box::new(items.into_iter()))
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn file_paths(&self) -> &[PathBuf] {
        std::slice::from_ref(&self.dsl_path)
    }

    fn reload_indexes(&mut self) -> Result<()> {
        let (btree, fts) = Self::load_sidecar_indexes(&self.dsl_path, &self.config)?;
        self.btree_index = btree;
        self.fts_index = fts;
        Ok(())
    }

    fn clear_cache(&mut self) {
        let mut cache = self.entry_cache.write();
        cache.clear();
    }

    fn stats(&self) -> DictStats {
        DictStats {
            total_entries: self.entries.len() as u64,
            cache_hit_rate: 0.0,
            memory_usage: 0,
            index_sizes: HashMap::new(),
        }
    }

    fn build_indexes(&mut self) -> Result<()> {
        // Build an in-memory FTS index for all entries; no on-disk format changes.
        let mut fts = FtsIndex::new();
        let cfg = crate::index::IndexConfig::default();
        let entries: Vec<(String, Vec<u8>)> = self
            .entries
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bytes().to_vec()))
            .collect();
        fts.build(&entries, &cfg)?;
        self.fts_index = Some(fts);
        Ok(())
    }
}

impl HighPerformanceDict<String> for DslDict {
    fn binary_search_get(&self, key: &String) -> Result<Vec<u8>> {
        // We don't keep a sorted structure yet; just delegate.
        self.get(key)
    }

    fn stream_search(&self, query: &str) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>> {
        // Use fulltext if available, else prefix scan.
        if let Ok(iter) = self.search_fulltext(query) {
            Ok(iter)
        } else {
            let res = self.search_prefix(query, Some(256))?;
            Ok(Box::new(res.into_iter().map(Ok)))
        }
    }
}

/// Simple Levenshtein distance used for DSL fuzzy search.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let (m, n) = (a.chars().count(), b.chars().count());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] =
                std::cmp::min(std::cmp::min(curr[j] + 1, prev[j + 1] + 1), prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// DictFormat implementation for DSL.
impl DictFormat<String> for DslDict {
    const FORMAT_NAME: &'static str = "dsl";
    const FORMAT_VERSION: &'static str = "1.0";

    fn is_valid_format(path: &Path) -> Result<bool> {
        // Accept *.dsl and *.dsl.dz based on extension.
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext = ext.to_lowercase();
            if ext == "dsl" || ext == "dz" {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let dict = DslDict::new(path, config)?;
        Ok(Box::new(dict))
    }
}

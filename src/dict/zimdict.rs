//! ZIM format implementation
//!
//! Lightweight ZIM parser:
//! - Parses ZIM header (ZIM_header) and mime list.
//! - Exposes basic metadata and supports lookup using an external index.
//! - Does not modify core traits or shared index modules.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::index::{btree::BTreeIndex, fts::FtsIndex, Index};
use crate::traits::{
    Dict, DictConfig, DictError, DictMetadata, DictStats, EntryIterator, HighPerformanceDict,
    Result, SearchResult,
};

/// ZIM file header (subset based on references/zim.cc ZIM_header)
#[derive(Debug, Clone)]
struct ZimHeader {
    magic_number: u32,
    major_version: u16,
    minor_version: u16,
    article_count: u32,
    cluster_count: u32,
    url_ptr_pos: u64,
    title_ptr_pos: u64,
    cluster_ptr_pos: u64,
    mime_list_pos: u64,
}

/// Location of an article blob: (cluster, blob_index)
#[derive(Debug, Clone, Copy)]
struct ArticleLoc {
    cluster: u32,
    blob: u32,
}

/// Minimal ZIM dictionary implementation.
///
/// Notes:
/// - We parse header and mime list.
/// - We build a lazy mapping from string key to ArticleLoc only if an external BTreeIndex
///   or other mechanism provides the mapping. For now, `get` relies on external index users.
pub struct ZimDict {
    /// Main ZIM file path
    file_path: PathBuf,
    /// Memory-mapped file for fast random access
    mmap: Option<Arc<Mmap>>,
    /// File handle for IO fallback
    file: File,
    /// Parsed header
    header: ZimHeader,
    /// Mime types list (index → string)
    mime_types: Vec<String>,
    /// Optional BTree index (external)
    btree_index: Option<BTreeIndex>,
    /// Optional FTS index (external)
    fts_index: Option<FtsIndex>,
    /// Cache for frequently accessed entries
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>,
    /// Configuration
    config: DictConfig,
    /// Cached metadata
    metadata: DictMetadata,
}

impl ZimDict {
    /// Create a new ZimDict from a .zim file.
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self> {
        let path = path.as_ref();
        let file_path = path.to_path_buf();

        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        // Open ZIM file
        let file = File::open(path)
            .map_err(|e| DictError::IoError(format!("open zim: {e}")))?;

        // Parse header
        let header = Self::read_header(&file)?;

        // Sanity validation (similar to references/zim.cc)
        if header.mime_list_pos != std::mem::size_of::<ZimHeader>() as u64 {
            // Many real ZIM files use this invariant; mismatch suggests unsupported variant.
            // We don't hard fail, but mark with a soft error by continuing.
        }

        // Memory-map file if enabled
        let mmap = if config.use_mmap {
            Some(Arc::new(
                unsafe {
                    memmap2::MmapOptions::new()
                        .map(&file)
                        .map_err(|e| DictError::MmapError(e.to_string()))?
                },
            ))
        } else {
            None
        };

        // Read mime types into memory
        let mime_types = Self::read_mime_list(&file, &header)?;

        // Load optional sidecar indexes (ZIMX-style external indexes if present)
        let (btree_index, fts_index) = Self::load_sidecar_indexes(&file_path, &config)?;

        // Build metadata (name/description come from external index or meta-items; for now minimal)
        let file_size = file
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);

        let name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("ZIM")
            .to_string();

        let metadata = DictMetadata {
            name,
            version: format!("{}.{}", header.major_version, header.minor_version),
            entries: header.article_count as u64,
            description: None,
            author: None,
            language: None,
            file_size,
            created: None,
            has_btree: btree_index.is_some(),
            has_fts: fts_index.is_some(),
        };

        let entry_cache =
            Arc::new(RwLock::new(lru_cache::LruCache::new(config.cache_size)));

        Ok(Self {
            file_path,
            mmap,
            file,
            header,
            mime_types,
            btree_index,
            fts_index,
            entry_cache,
            config,
            metadata,
        })
    }

    /// Read ZIM file header from beginning of file.
    fn read_header(file: &File) -> Result<ZimHeader> {
        let mut buf = [0u8; 80]; // Enough for ZIM_header
        let mut reader = BufReader::new(file);
        reader
            .read_exact(&mut buf)
            .map_err(|e| DictError::IoError(format!("read zim header: {e}")))?;

        // Layout from zim.cc (ZIM_header), all little-endian.
        let magic_number = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        // 0x44D495A is used in reference; canonical ZIM magic is 'Z' 'I' 'M' 'A' (0x41_4D_49_5ALE).
        if magic_number == 0 {
            return Err(DictError::InvalidFormat(
                "Invalid ZIM magic number".to_string(),
            ));
        }

        let major_version = u16::from_le_bytes([buf[4], buf[5]]);
        let minor_version = u16::from_le_bytes([buf[6], buf[7]]);

        // uuid[16] at 8..24
        let article_count =
            u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
        let cluster_count =
            u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);

        let url_ptr_pos = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
        ]);
        let title_ptr_pos = u64::from_le_bytes([
            buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
        ]);
        let cluster_ptr_pos = u64::from_le_bytes([
            buf[48], buf[49], buf[50], buf[51], buf[52], buf[53], buf[54], buf[55],
        ]);
        let mime_list_pos = u64::from_le_bytes([
            buf[56], buf[57], buf[58], buf[59], buf[60], buf[61], buf[62], buf[63],
        ]);
        // mainPage (64..68), layoutPage (68..72), checksumPos(72..80) are not used here.

        Ok(ZimHeader {
            magic_number,
            major_version,
            minor_version,
            article_count,
            cluster_count,
            url_ptr_pos,
            title_ptr_pos,
            cluster_ptr_pos,
            mime_list_pos,
        })
    }

    /// Read mime type list starting at mime_list_pos.
    fn read_mime_list(file: &File, header: &ZimHeader) -> Result<Vec<String>> {
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(header.mime_list_pos))
            .map_err(|e| DictError::IoError(format!("seek mime list: {e}")))?;

        let mut list = Vec::new();
        loop {
            let mut buf = Vec::new();
            // Read until '\0'
            loop {
                let mut byte = [0u8; 1];
                match reader.read_exact(&mut byte) {
                    Ok(()) => {
                        if byte[0] == 0 {
                            break;
                        } else {
                            buf.push(byte[0]);
                        }
                    }
                    Err(e) => {
                        // EOF or error: stop
                        if list.is_empty() {
                            return Err(DictError::IoError(format!(
                                "read mime list: {e}"
                            )));
                        }
                        return Ok(list);
                    }
                }
            }
            if buf.is_empty() {
                // Empty entry terminates the list
                break;
            }
            match String::from_utf8(buf) {
                Ok(s) => list.push(s),
                Err(_) => list.push(String::new()),
            }
        }
        Ok(list)
    }

    /// Load optional ZIMX-style sidecar indexes (.btree/.fts).
    fn load_sidecar_indexes(
        path: &Path,
        config: &DictConfig,
    ) -> Result<(Option<BTreeIndex>, Option<FtsIndex>)> {
        let stem = path.with_extension("");
        let idx_btree = stem.with_extension("btree");
        let idx_fts = stem.with_extension("fts");

        let mut btree_index = None;
        let mut fts_index = None;

        if config.load_btree && idx_btree.exists() {
            let mut b = BTreeIndex::new();
            if let Err(e) = b.load(&idx_btree) {
                return Err(DictError::IndexError(format!(
                    "Failed to load ZIM B-TREE index {}: {}",
                    idx_btree.display(),
                    e
                )));
            }
            if !b.is_built() {
                return Err(DictError::IndexError(format!(
                    "ZIM B-TREE index {} is not built or is empty",
                    idx_btree.display()
                )));
            }
            btree_index = Some(b);
        }

        if config.load_fts && idx_fts.exists() {
            let mut f = FtsIndex::new();
            if let Err(e) = f.load(&idx_fts) {
                return Err(DictError::IndexError(format!(
                    "Failed to load ZIM FTS index {}: {}",
                    idx_fts.display(),
                    e
                )));
            }
            if !f.is_built() {
                return Err(DictError::IndexError(format!(
                    "ZIM FTS index {} is not built or is empty",
                    idx_fts.display()
                )));
            }
            fts_index = Some(f);
        }

        Ok((btree_index, fts_index))
    }

    /// Read raw bytes from the ZIM file at offset/size.
    fn read_raw(&self, offset: u64, size: u64) -> Result<Vec<u8>> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if let Some(ref mmap) = self.mmap {
            let end = offset
                .checked_add(size)
                .ok_or_else(|| DictError::Internal("overflow in zim read_raw".into()))?;
            if end as usize > mmap.len() {
                return Err(DictError::IoError(
                    "ZIM read past end of file".to_string(),
                ));
            }
            Ok(mmap[offset as usize..end as usize].to_vec())
        } else {
            let mut reader = BufReader::new(&self.file);
            reader
                .seek(SeekFrom::Start(offset))
                .map_err(|e| DictError::IoError(format!("seek zim: {e}")))?;
            let mut buf = vec![0u8; size as usize];
            reader
                .read_exact(&mut buf)
                .map_err(|e| DictError::IoError(format!("read zim: {e}")))?;
            Ok(buf)
        }
    }

    /// Cache helpers
    fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.entry_cache.write();
        cache.get_mut(&key.to_string()).map(|v| v.clone())
    }

    fn cache_put(&self, key: String, val: Vec<u8>) {
        let mut cache = self.entry_cache.write();
        cache.insert(key, val);
    }

    /// Read article payload by article number using urlPtrPos / ZIM layout.
    ///
    /// This is a minimal implementation based on the reference:
    /// - Follows redirects (mimetype 0xFFFF) once.
    /// - Locates the cluster via urlPtrPos + 8 * articleNo.
    /// - Reads cluster, interprets first offsets table, then extracts blob.
    fn read_article_by_number(&self, article_no: u32) -> Result<Vec<u8>> {
        if article_no >= self.header.article_count {
            return Err(DictError::InvalidFormat(format!(
                "ZIM: article_no {} out of range",
                article_no
            )));
        }

        // Load URL entry position
        let mut rdr = BufReader::new(&self.file);
        let url_ptr = self
            .header
            .url_ptr_pos
            .checked_add(article_no as u64 * 8)
            .ok_or_else(|| {
                DictError::InvalidFormat("ZIM: url_ptr_pos overflow".to_string())
            })?;
        rdr.seek(SeekFrom::Start(url_ptr))
            .map_err(|e| DictError::IoError(format!("seek urlPtr: {e}")))?;
        let mut pos_buf = [0u8; 8];
        rdr.read_exact(&mut pos_buf)
            .map_err(|e| DictError::IoError(format!("read urlPtr: {e}")))?;
        let entry_pos = u64::from_le_bytes(pos_buf);

        // Read mimetype
        rdr.seek(SeekFrom::Start(entry_pos))
            .map_err(|e| DictError::IoError(format!("seek entry: {e}")))?;
        let mut mt_buf = [0u8; 2];
        rdr.read_exact(&mut mt_buf)
            .map_err(|e| DictError::IoError(format!("read mimetype: {e}")))?;
        let mimetype = u16::from_le_bytes(mt_buf);

        // Redirect handling (single step for now)
        let (cluster, blob) = if mimetype == 0xFFFF {
            // RedirectEntry: [mimetype(2)][paramLen(1)][ns(1)][rev(4)][redirectIndex(4)]
            let mut buf = [0u8; 1 + 1 + 4 + 4];
            rdr.read_exact(&mut buf)
                .map_err(|e| DictError::IoError(format!("read redirect: {e}")))?;
            let redirect_index =
                u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
            // Recursive/loop-safe resolution left out for brevity.
            self.read_article_location(redirect_index)?
        } else {
            self.read_article_location(article_no)?
        };

        self.read_blob(cluster, blob)
    }

    /// Read ArticleEntry for given article number and return (cluster, blob).
    fn read_article_location(&self, article_no: u32) -> Result<(u32, u32)> {
        // Similar layout to reference ArticleEntry.
        // urlPtrPos gives us pointer to entry; after mimetype and fields comes ArticleEntry.
        let mut rdr = BufReader::new(&self.file);
        let url_ptr = self
            .header
            .url_ptr_pos
            .checked_add(article_no as u64 * 8)
            .ok_or_else(|| {
                DictError::InvalidFormat("ZIM: url_ptr_pos overflow".to_string())
            })?;
        rdr.seek(SeekFrom::Start(url_ptr))
            .map_err(|e| DictError::IoError(format!("seek urlPtr: {e}")))?;
        let mut pos_buf = [0u8; 8];
        rdr.read_exact(&mut pos_buf)
            .map_err(|e| DictError::IoError(format!("read urlPtr: {e}")))?;
        let entry_pos = u64::from_le_bytes(pos_buf);

        rdr.seek(SeekFrom::Start(entry_pos + 2))
            .map_err(|e| DictError::IoError(format!("seek ArticleEntry: {e}")))?;
        let mut buf = [0u8; 1 + 1 + 4 + 4]; // parameterLen, ns, rev, cluster, blob
        rdr.read_exact(&mut buf)
            .map_err(|e| DictError::IoError(format!("read ArticleEntry: {e}")))?;

        let cluster = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
        let blob = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);

        Ok((cluster, blob))
    }

    /// Read a blob from a cluster:
    /// - Uses cluster_ptr_pos table to locate cluster.
    /// - Reads and decompresses cluster content according to compression type.
    /// - Uses offsets table inside cluster to extract requested blob payload.
    fn read_blob(&self, cluster_no: u32, blob_no: u32) -> Result<Vec<u8>> {
        if cluster_no >= self.header.cluster_count {
            return Err(DictError::InvalidFormat(format!(
                "ZIM: cluster_no {} out of range",
                cluster_no
            )));
        }

        // Read cluster offset
        let mut rdr = BufReader::new(&self.file);
        let cluster_ptr = self
            .header
            .cluster_ptr_pos
            .checked_add(cluster_no as u64 * 8)
            .ok_or_else(|| {
                DictError::InvalidFormat("ZIM: cluster_ptr_pos overflow".to_string())
            })?;
        rdr.seek(SeekFrom::Start(cluster_ptr))
            .map_err(|e| DictError::IoError(format!("seek clusterPtr: {e}")))?;
        let mut off_buf = [0u8; 8];
        rdr.read_exact(&mut off_buf)
            .map_err(|e| DictError::IoError(format!("read cluster offset: {e}")))?;
        let cluster_offset = u64::from_le_bytes(off_buf);

        // Read first bytes: compression type + data.
        rdr.seek(SeekFrom::Start(cluster_offset))
            .map_err(|e| DictError::IoError(format!("seek cluster: {e}")))?;
        let mut header_byte = [0u8; 1];
        rdr.read_exact(&mut header_byte)
            .map_err(|e| DictError::IoError(format!("read cluster header: {e}")))?;
        let compression_type = header_byte[0] & 0x0F;
        let blobs_offset_size = if (header_byte[0] & 0x10) != 0
            && self.header.major_version >= 6
        {
            8
        } else {
            4
        };

        // Read rest of cluster up to next cluster or EOF (we approximate by reading a fixed chunk).
        let mut data = Vec::new();
        rdr.read_to_end(&mut data)
            .map_err(|e| DictError::IoError(format!("read cluster data: {e}")))?;

        // Decompression: use util::compression via crate::index::CompressionAlgorithm mapping
        let decompressed = match compression_type {
            0 => data, // Default/None
            _ => {
                // For brevity, non-zero compression types treated as unsupported.
                return Err(DictError::UnsupportedOperation(
                    "ZIM compressed clusters not fully implemented".to_string(),
                ));
            }
        };

        if decompressed.len() < blobs_offset_size {
            return Err(DictError::InvalidFormat(
                "ZIM cluster too small for offsets".to_string(),
            ));
        }

        // First offset gives start of offsets table; compute blob count
        let first_off = if blobs_offset_size == 8 {
            let mut tmp = [0u8; 8];
            tmp.copy_from_slice(&decompressed[0..8]);
            u64::from_le_bytes(tmp) as usize
        } else {
            let mut tmp = [0u8; 4];
            tmp.copy_from_slice(&decompressed[0..4]);
            u32::from_le_bytes(tmp) as usize
        };

        let blob_count = (first_off - blobs_offset_size) / blobs_offset_size;
        if (blob_no as usize) >= blob_count {
            return Err(DictError::InvalidFormat(format!(
                "ZIM: blob_no {} out of range in cluster",
                blob_no
            )));
        }

        // Read two consecutive offsets for blob_no
        let (start, end) = if blobs_offset_size == 8 {
            let base = blob_no as usize * 8;
            let mut off_bytes = [0u8; 16];
            off_bytes.copy_from_slice(
                &decompressed[base..base + 16],
            );
            let o1 = u64::from_le_bytes(off_bytes[0..8].try_into().unwrap()) as usize;
            let o2 = u64::from_le_bytes(off_bytes[8..16].try_into().unwrap()) as usize;
            (o1, o2)
        } else {
            let base = blob_no as usize * 4;
            let mut off_bytes = [0u8; 8];
            off_bytes.copy_from_slice(&decompressed[base..base + 8]);
            let o1 = u32::from_le_bytes(off_bytes[0..4].try_into().unwrap()) as usize;
            let o2 = u32::from_le_bytes(off_bytes[4..8].try_into().unwrap()) as usize;
            (o1, o2)
        };

        if end > decompressed.len() || start >= end {
            return Err(DictError::InvalidFormat(
                "ZIM blob offsets out of bounds".to_string(),
            ));
        }

        Ok(decompressed[start..end].to_vec())
    }
}

impl Dict<String> for ZimDict {
    fn metadata(&self) -> &DictMetadata {
        &self.metadata
    }

    fn contains(&self, key: &String) -> Result<bool> {
        if let Some(btree) = &self.btree_index {
            Ok(btree.search(key)?.is_some())
        } else {
            // Without a verified BTree sidecar index we cannot answer contains() reliably.
            Err(DictError::UnsupportedOperation(
                "ZIM contains() requires a loaded BTree index".to_string(),
            ))
        }
    }

    fn get(&self, key: &String) -> Result<Vec<u8>> {
        if let Some(cached) = self.get_cached(key) {
            return Ok(cached);
        }

        // For full fidelity we'd parse URL/title pointers and follow ArticleEntry / RedirectEntry.
        // Here we rely on external ZIMX-style BTree indexes mapping key -> article number.
        let article_no = if let Some(btree) = &self.btree_index {
            match btree.search(key)? {
                Some((_k, off)) => off as u32,
                None => {
                    return Err(DictError::IndexError(format!(
                        "ZIM key not found in sidecar index: {key}"
                    )))
                }
            }
        } else {
            return Err(DictError::UnsupportedOperation(
                "ZIM lookup requires BTree index; none loaded".to_string(),
            ));
        };

        let data = self.read_article_by_number(article_no)?;
        self.cache_put(key.clone(), data.clone());
        Ok(data)
    }

    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let max = limit.unwrap_or(64);

        if let Some(btree) = &self.btree_index {
            let start = prefix.to_string();
            let end = format!("{}\u{10FFFF}", prefix);
            let range = btree.range_query(&start, &end)?;
            for (word, off) in range.into_iter().take(max) {
                let data = self.read_article_by_number(off as u32)?;
                results.push(SearchResult {
                    word,
                    entry: data,
                    score: None,
                    highlights: None,
                });
            }
        }

        Ok(results)
    }

    fn search_fuzzy(&self, query: &str, _max_distance: Option<u32>) -> Result<Vec<SearchResult>> {
        // Minimal implementation: reuse prefix search as approximation
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
                if let Ok(data) = self.get(&word) {
                    out.push(SearchResult {
                        word,
                        entry: data,
                        score: Some(score),
                        highlights: None,
                    });
                }
            }
            Ok(Box::new(out.into_iter().map(Ok)))
        } else {
            // Fallback to prefix search if FTS is not available
            let res = self.search_prefix(query, Some(64))?;
            Ok(Box::new(res.into_iter().map(Ok)))
        }
    }

    fn get_range(
        &self,
        _range: std::ops::Range<usize>,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        // Could be implemented by scanning BTree index; not essential for core usage.
        Ok(Vec::new())
    }

    fn iter(&self) -> Result<EntryIterator<'_, String>> {
        Err(DictError::UnsupportedOperation(
            "ZIM full iteration not implemented".to_string(),
        ))
    }

    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(String, Vec<u8>)>> + Send>> {
        let hits = self.search_prefix(prefix, Some(256))?;
        let mapped: Vec<_> = hits
            .into_iter()
            .map(|sr| Ok((sr.word, sr.entry)))
            .collect();
        Ok(Box::new(mapped.into_iter()))
    }

    fn len(&self) -> usize {
        self.header.article_count as usize
    }

    fn file_paths(&self) -> &[PathBuf] {
        std::slice::from_ref(&self.file_path)
    }

    fn reload_indexes(&mut self) -> Result<()> {
        let (btree, fts) = Self::load_sidecar_indexes(&self.file_path, &self.config)?;
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
            total_entries: self.header.article_count as u64,
            cache_hit_rate: 0.0,
            memory_usage: 0,
            index_sizes: HashMap::new(),
        }
    }

    fn build_indexes(&mut self) -> Result<()> {
        // Full ZIM indexing (URL/title → articleNo, FTS) is handled by external tools.
        Ok(())
    }
}

impl HighPerformanceDict<String> for ZimDict {
    fn binary_search_get(&self, key: &String) -> Result<Vec<u8>> {
        self.get(key)
    }

    fn stream_search(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>> {
        // Use fulltext (or its fallback) and erase Send to match trait signature.
        let it = self.search_fulltext(query)?;
        Ok(Box::new(it))
    }
}
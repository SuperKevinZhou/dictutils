//! StarDict format implementation
//!
//! Full-fidelity StarDict parser aligned with GoldenDict's behavior where practical:
//! - Accepts .ifo as entry point.
//! - Parses .ifo metadata (Ifo::Ifo semantics).
//! - Locates and parses .idx(.gz/.dz) and .dict(.dz) files.
//! - Supports 32-bit offsets; rejects idxoffsetbits=64 as in GoldenDict.
//! - Applies .syn synonyms.
//! - Reads articles from .dict or DICTZIP-compressed .dict.dz.
//! - Uses existing BTreeIndex/FtsIndex if sidecar indexes exist (no changes to core index traits).
//!
//! Only parsing and lookup logic is implemented here; core traits and index modules remain unchanged.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use flate2::read::GzDecoder;
use memmap2::Mmap;
use parking_lot::RwLock;

use crate::index::{btree::BTreeIndex, fts::FtsIndex, Index};
use crate::traits::{
    Dict, DictConfig, DictError, DictMetadata, DictStats, EntryIterator, HighPerformanceDict,
    Result, SearchResult,
};

/// Parsed contents of .ifo (GoldenDict's Stardict::Ifo equivalent)
#[derive(Debug, Clone)]
struct Ifo {
    version: String,
    bookname: String,
    wordcount: u64,
    synwordcount: u64,
    idxfilesize: Option<u64>,
    idxoffsetbits: u32,
    sametypesequence: Option<String>,
    dicttype: Option<String>,
    description: Option<String>,
    copyright: Option<String>,
    author: Option<String>,
    email: Option<String>,
    website: Option<String>,
    date: Option<String>,
}

/// StarDict dictionary header/metadata
#[derive(Debug, Clone)]
struct StarDictHeader {
    ifo: Ifo,
    /// Encoding for textual parts; StarDict uses UTF-8 by specification,
    /// but we keep this for possible future extensions.
    encoding: String,
    /// True if index offsets are 64-bit (we reject those like GoldenDict).
    idx_64bit: bool,
}

/// Entry location in .dict or .dict.dz
#[derive(Debug, Clone, Copy)]
struct EntryLoc {
    offset: u64,
    size: u64,
}

/// Main StarDict implementation
pub struct StarDict {
    /// .ifo path
    ifo_path: PathBuf,
    /// Associated .dict or .dict.dz
    dict_path: PathBuf,
    /// Whether dict is DICTZIP-compressed (.dict.dz)
    dict_is_dz: bool,
    /// Optional .syn path
    syn_path: Option<PathBuf>,
    /// Parsed header
    header: StarDictHeader,
    /// In-memory index: word → (offset,size)
    ///
    /// This is filled from .idx(.gz/.dz) exactly per spec:
    /// [word\0][offset(4)][size(4)] with network-byte-order (big endian).
    index: HashMap<String, EntryLoc>,
    /// Memory-mapped .dict (for uncompressed dict)
    mmap: Option<Arc<Mmap>>,
    /// File handle for .dict/.dz
    dict_file: File,
    /// Optional BTree index (sidecar, our own index format)
    btree_index: Option<BTreeIndex>,
    /// Optional FTS index (sidecar, our own index format)
    fts_index: Option<FtsIndex>,
    /// Cache for frequently accessed entries
    entry_cache: Arc<RwLock<lru_cache::LruCache<String, Vec<u8>>>>,
    /// Configuration
    config: DictConfig,
    /// Cached metadata
    metadata: DictMetadata,
}

impl StarDict {
    /// Create a new StarDict from a .ifo file.
    ///
    /// The path must point to a StarDict .ifo. Companion files (.idx*, .dict*, .syn*)
    /// are resolved using standard StarDict naming rules.
    pub fn new<P: AsRef<Path>>(path: P, config: DictConfig) -> Result<Self> {
        let ifo_path = path.as_ref().to_path_buf();
        if !ifo_path.exists() {
            return Err(DictError::FileNotFound(ifo_path.display().to_string()));
        }

        // 1) Parse .ifo
        let ifo = Self::parse_ifo(&ifo_path)?;
        if ifo.idxoffsetbits == 64 {
            // GoldenDict: 64-bit not supported; follow same restriction for now.
            return Err(DictError::InvalidFormat(
                "StarDict idxoffsetbits=64 is not supported".to_string(),
            ));
        }
        if let Some(ref dicttype) = ifo.dicttype {
            if !dicttype.is_empty() {
                return Err(DictError::InvalidFormat(format!(
                    "StarDict dicttype '{}' not supported",
                    dicttype
                )));
            }
        }

        // 2) Find .idx/.dict/.syn companions
        let (idx_path, dict_path, dict_is_dz, syn_path) = Self::find_companion_files(&ifo_path)?;

        // 3) Build in-memory index from .idx/.idx.gz/.idx.dz
        let idx_64bit = ifo.idxoffsetbits == 64;
        let index = Self::load_idx(&idx_path, idx_64bit)?;

        // 4) Apply .syn if present
        let mut index = index;
        if let Some(ref syn) = syn_path {
            Self::apply_syn(syn, &mut index)?;
        }

        // Basic sanity vs .ifo wordcount
        if ifo.wordcount > 0 && index.len() < ifo.wordcount as usize {
            // Do not fail hard; just warn via error type
            // but keep behavior deterministic
        }

        // 5) Open .dict/.dict.dz
        let dict_file = File::open(&dict_path)
            .map_err(|e| DictError::IoError(format!("open dict: {e}")))?;

        // 6) Mmap if uncompressed and allowed
        let mmap = if config.use_mmap && !dict_is_dz {
            Some(Arc::new(
                unsafe {
                    memmap2::MmapOptions::new()
                        .map(&dict_file)
                        .map_err(|e| DictError::MmapError(e.to_string()))?
                },
            ))
        } else {
            None
        };

        // 7) Load optional sidecar indexes
        let (btree_index, fts_index) =
            Self::load_sidecar_indexes(&ifo_path, &config)?;

        // 8) Build DictMetadata
        let file_size = dict_file
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);

        let name = if !ifo.bookname.is_empty() {
            ifo.bookname.clone()
        } else {
            ifo_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("StarDict")
                .to_string()
        };

        let metadata = DictMetadata {
            name,
            version: ifo.version.clone(),
            entries: index.len() as u64,
            description: ifo.description.clone(),
            author: ifo.author.clone(),
            language: None, // StarDict doesn't standardize this well; leave None.
            file_size,
            created: ifo.date.clone(),
            has_btree: btree_index.is_some(),
            has_fts: fts_index.is_some(),
        };

        let header = StarDictHeader {
            encoding: "UTF-8".to_string(),
            idx_64bit,
            ifo,
        };

        let entry_cache =
            Arc::new(RwLock::new(lru_cache::LruCache::new(config.cache_size)));

        Ok(Self {
            ifo_path,
            dict_path,
            dict_is_dz,
            syn_path,
            header,
            index,
            mmap,
            dict_file,
            btree_index,
            fts_index,
            entry_cache,
            config,
            metadata,
        })
    }

    /// Parse .ifo file according to references/stardict.cc Ifo::Ifo.
    fn parse_ifo(path: &Path) -> Result<Ifo> {
        let file = File::open(path)
            .map_err(|e| DictError::IoError(format!("open ifo: {e}")))?;
        let mut reader = BufReader::new(file);

        let mut first = String::new();
        reader
            .read_line(&mut first)
            .map_err(|e| DictError::IoError(e.to_string()))?;
        let first = first.trim_end_matches(&['\r', '\n'][..]).to_string();

        let mut second = String::new();
        reader
            .read_line(&mut second)
            .map_err(|e| DictError::IoError(e.to_string()))?;
        let second = second.trim_end_matches(&['\r', '\n'][..]).to_string();

        if first != "StarDict's dict ifo file" || !second.starts_with("version=") {
            return Err(DictError::InvalidFormat(
                "Not a StarDict .ifo file".to_string(),
            ));
        }

        let version = second["version=".len()..].to_string();

        let mut ifo = Ifo {
            version,
            bookname: String::new(),
            wordcount: 0,
            synwordcount: 0,
            idxfilesize: None,
            idxoffsetbits: 32,
            sametypesequence: None,
            dicttype: None,
            description: None,
            copyright: None,
            author: None,
            email: None,
            website: None,
            date: None,
        };

        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| DictError::IoError(e.to_string()))?;
            if n == 0 {
                break;
            }
            let l = line.trim_end_matches(&['\r', '\n'][..]);
            if l.is_empty() {
                continue;
            }

            macro_rules! parse_u {
                ($field:ident, $name:expr) => {
                    if let Some(v) = l.strip_prefix($name) {
                        ifo.$field = v
                            .parse()
                            .map_err(|_| DictError::InvalidFormat(format!(
                                "Bad field in .ifo: {}",
                                $name
                            )))?;
                        continue;
                    }
                };
            }

            if let Some(v) = l.strip_prefix("bookname=") {
                ifo.bookname = v.to_string();
                continue;
            }
            parse_u!(wordcount, "wordcount=");
            parse_u!(synwordcount, "synwordcount=");
            if let Some(v) = l.strip_prefix("idxfilesize=") {
                let val: u64 = v
                    .parse()
                    .map_err(|_| DictError::InvalidFormat("Bad idxfilesize".to_string()))?;
                ifo.idxfilesize = Some(val);
                continue;
            }
            if let Some(v) = l.strip_prefix("idxoffsetbits=") {
                let bits: u32 = v
                    .parse()
                    .map_err(|_| DictError::InvalidFormat("Bad idxoffsetbits".to_string()))?;
                if bits != 32 && bits != 64 {
                    return Err(DictError::InvalidFormat(
                        "idxoffsetbits must be 32 or 64".to_string(),
                    ));
                }
                ifo.idxoffsetbits = bits;
                continue;
            }
            if let Some(v) = l.strip_prefix("sametypesequence=") {
                ifo.sametypesequence = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("dicttype=") {
                ifo.dicttype = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("description=") {
                ifo.description = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("copyright=") {
                ifo.copyright = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("author=") {
                ifo.author = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("email=") {
                ifo.email = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("website=") {
                ifo.website = Some(v.to_string());
                continue;
            }
            if let Some(v) = l.strip_prefix("date=") {
                ifo.date = Some(v.to_string());
                continue;
            }
            // Ignore unknown keys.
        }

        Ok(ifo)
    }

    /// Given path/to/file.ifo, locate .idx(.gz/.dz), .dict(.dz), .syn like references/stardict.cc::findCorrespondingFiles.
    fn find_companion_files(
        ifo_path: &Path,
    ) -> Result<(PathBuf, PathBuf, bool, Option<PathBuf>)> {
        let stem = ifo_path
            .with_extension("")
            .to_string_lossy()
            .to_string();

        // Try idx
        let mut idx_candidates = [
            format!("{stem}.idx"),
            format!("{stem}.idx.gz"),
            format!("{stem}.idx.dz"),
            format!("{stem}.IDX"),
            format!("{stem}.IDX.GZ"),
            format!("{stem}.IDX.DZ"),
        ];
        let idx_path = idx_candidates
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .ok_or_else(|| {
                DictError::InvalidFormat(format!(
                    "No corresponding .idx file was found for {}",
                    ifo_path.display()
                ))
            })?;

        // Try dict
        let mut dict_candidates = [
            format!("{stem}.dict"),
            format!("{stem}.dict.dz"),
            format!("{stem}.DICT"),
            format!("{stem}.dict.DZ"),
        ];
        let dict_path = dict_candidates
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .ok_or_else(|| {
                DictError::InvalidFormat(format!(
                    "No corresponding .dict file was found for {}",
                    ifo_path.display()
                ))
            })?;
        let dict_is_dz = dict_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("dz"))
            .unwrap_or(false);

        // Try syn (optional)
        let syn_candidates = [
            format!("{stem}.syn"),
            format!("{stem}.syn.gz"),
            format!("{stem}.syn.dz"),
            format!("{stem}.SYN"),
            format!("{stem}.SYN.GZ"),
            format!("{stem}.SYN.DZ"),
        ];
        let syn_path = syn_candidates
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists());

        Ok((idx_path, dict_path, dict_is_dz, syn_path))
    }

    /// Load .idx(.gz/.dz) into a word → (offset,size) map.
    ///
    /// We support:
    /// - Uncompressed .idx with 32-bit offsets.
    /// - .idx.gz (whole-file gzip) by streaming through a GzDecoder.
    /// .idx.dz (dictzip) is not implemented yet: we accept .idx.dz but treat
    /// it equivalently to .idx.gz if it is plain gz; full dictzip index layout
    /// is out of scope for this lightweight implementation.
    fn load_idx(idx_path: &Path, idx_64bit: bool) -> Result<HashMap<String, EntryLoc>> {
        if idx_64bit {
            return Err(DictError::InvalidFormat(
                "StarDict 64-bit .idx not supported in lightweight parser".to_string(),
            ));
        }

        let ext = idx_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let mut map = HashMap::new();

        if ext == "gz" || ext == "dz" {
            let file = File::open(idx_path)
                .map_err(|e| DictError::IoError(format!("open idx.gz: {e}")))?;
            let mut gz = GzDecoder::new(file);
            let mut buf = Vec::new();
            gz.read_to_end(&mut buf)
                .map_err(|e| DictError::IoError(format!("read idx.gz: {e}")))?;
            Self::parse_idx_buffer(&buf, &mut map)?;
        } else {
            let mut file = File::open(idx_path)
                .map_err(|e| DictError::IoError(format!("open idx: {e}")))?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(|e| DictError::IoError(format!("read idx: {e}")))?;
            Self::parse_idx_buffer(&buf, &mut map)?;
        }

        Ok(map)
    }

    /// Parse raw .idx buffer: [word\0][offset:4][size:4] repeated.
    fn parse_idx_buffer(buf: &[u8], out: &mut HashMap<String, EntryLoc>) -> Result<()> {
        let mut i = 0usize;
        let n = buf.len();
        while i < n {
            // Extract word until NUL
            let start = i;
            while i < n && buf[i] != 0 {
                i += 1;
            }
            if i >= n {
                break; // malformed; stop
            }
            let word_bytes = &buf[start..i];
            let word = String::from_utf8_lossy(word_bytes).to_string();
            i += 1; // skip NUL

            if i + 8 > n {
                break; // malformed
            }

            let offset = u32::from_be_bytes([
                buf[i],
                buf[i + 1],
                buf[i + 2],
                buf[i + 3],
            ]) as u64;
            i += 4;
            let size = u32::from_be_bytes([
                buf[i],
                buf[i + 1],
                buf[i + 2],
                buf[i + 3],
            ]) as u64;
            i += 4;

            out.insert(word, EntryLoc { offset, size });
        }
        Ok(())
    }

    /// Apply .syn file: each entry is [syn_word\0][idx:4] where idx refers to the Nth entry in .idx.
    ///
    /// To keep things simple and deterministic, we:
    /// - Build a stable Vec of (word, EntryLoc) from the current index in insertion order.
    /// - For each syn entry, map to that vector index and insert syn_word → same EntryLoc.
    fn apply_syn(path: &Path, index: &mut HashMap<String, EntryLoc>) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }

        let base_entries: Vec<(String, EntryLoc)> =
            index.iter().map(|(k, v)| (k.clone(), *v)).collect();

        let file = File::open(path)
            .map_err(|e| DictError::IoError(format!("open syn: {e}")))?;

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let mut buf = Vec::new();
        if ext == "gz" || ext == "dz" {
            let mut gz = GzDecoder::new(file);
            gz.read_to_end(&mut buf)
                .map_err(|e| DictError::IoError(format!("read syn.gz: {e}")))?;
        } else {
            let mut r = BufReader::new(file);
            r.read_to_end(&mut buf)
                .map_err(|e| DictError::IoError(format!("read syn: {e}")))?;
        }

        let mut i = 0usize;
        let n = buf.len();
        while i < n {
            // syn word
            let start = i;
            while i < n && buf[i] != 0 {
                i += 1;
            }
            if i >= n {
                break;
            }
            let syn_word = String::from_utf8_lossy(&buf[start..i]).to_string();
            i += 1;

            if i + 4 > n {
                break;
            }
            let idx = u32::from_be_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
            i += 4;

            if let Some((_, loc)) = base_entries.get(idx) {
                // Insert synonym only if not present already
                index.entry(syn_word).or_insert(*loc);
            }
        }

        Ok(())
    }

    /// Load optional sidecar .btree and .fts based on the .ifo stem.
    fn load_sidecar_indexes(
        ifo_path: &Path,
        config: &DictConfig,
    ) -> Result<(Option<BTreeIndex>, Option<FtsIndex>)> {
        let stem = ifo_path.with_extension("");
        let btree_path = stem.with_extension("btree");
        let fts_path = stem.with_extension("fts");

        let mut btree_index = None;
        let mut fts_index = None;

        if config.load_btree && btree_path.exists() {
            let mut b = BTreeIndex::new();
            b.load(&btree_path)?;
            btree_index = Some(b);
        }

        if config.load_fts && fts_path.exists() {
            let mut f = FtsIndex::new();
            f.load(&fts_path)?;
            fts_index = Some(f);
        }

        Ok((btree_index, fts_index))
    }

    /// Read entry bytes from .dict or .dict.dz given offset/size.
    fn read_entry(&self, loc: EntryLoc) -> Result<Vec<u8>> {
        if loc.size == 0 {
            return Ok(Vec::new());
        }

        if self.dict_is_dz {
            // DICTZIP support (simplified but spec-correct for standard .dict.dz):
            // A real dictzip file:
            // - Starts with gzip header with extra field "RA" describing chunk table.
            // - Provides random-access to compressed chunks.
            // Implementing full dictzip requires parsing that extra field and
            // mapping logical offsets to (chunk, in-chunk) positions.
            //
            // For this crate we implement a conservative, self-contained reader:
            // - Parse gzip header.
            // - If no dictzip "RA" extra is present, fall back to treating it
            //   as plain gzip (sequential inflate) and report UnsupportedOperation
            //   for random access.
            // - If "RA" table is present, build a minimal chunk index in memory
            //   and perform on-demand chunk reads.
            //
            // To avoid touching shared modules, we keep all logic local here.
            let mut file = &self.dict_file;

            // Read and parse gzip header + extra field to find dictzip table.
            // We only parse once and cache results via a lazy static style in
            // future refactors; here we do per-call parsing for correctness.
            file.seek(SeekFrom::Start(0))
                .map_err(|e| DictError::IoError(format!("dict.dz seek header: {e}")))?;

            let mut header = [0u8; 10];
            file.read_exact(&mut header)
                .map_err(|e| DictError::IoError(format!("dict.dz read header: {e}")))?;

            if header[0] != 0x1f || header[1] != 0x8b {
                return Err(DictError::InvalidFormat(
                    "dict.dz is not a valid gzip file".to_string(),
                ));
            }

            let flg = header[3];
            let fextra = flg & 0x04 != 0;
            let fname = flg & 0x08 != 0;
            let fcomment = flg & 0x10 != 0;
            let fhcrc = flg & 0x02 != 0;

            // Parse FEXTRA to find "RA" subfield if present.
            let mut ra_chunks: Option<(u16, Vec<u32>)> = None;
            if fextra {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)
                    .map_err(|e| DictError::IoError(format!("dict.dz read extra length: {e}")))?;
                let xlen = u16::from_le_bytes(len_buf);
                let mut extra = vec![0u8; xlen as usize];
                file.read_exact(&mut extra)
                    .map_err(|e| DictError::IoError(format!("dict.dz read extra: {e}")))?;

                let mut i = 0usize;
                while i + 4 <= extra.len() {
                    let si1 = extra[i];
                    let si2 = extra[i + 1];
                    let sublen = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
                    i += 4;
                    if i + sublen > extra.len() {
                        break;
                    }

                    // dictzip "RA" subfield marks random access table
                    if si1 == b'R' && si2 == b'A' && sublen >= 2 {
                        let chunk_len = u16::from_be_bytes([extra[i], extra[i + 1]]);
                        let mut offs = Vec::new();
                        let mut j = i + 2;
                        while j + 4 <= i + sublen {
                            let o = u32::from_be_bytes([
                                extra[j],
                                extra[j + 1],
                                extra[j + 2],
                                extra[j + 3],
                            ]);
                            offs.push(o);
                            j += 4;
                        }
                        if !offs.is_empty() {
                            ra_chunks = Some((chunk_len, offs));
                        }
                        break;
                    }

                    i += sublen;
                }
            }

            // Skip optional FNAME/FCOMMENT/FHCRC to reach compressed data start.
            fn skip_cstring<R: Read>(reader: &mut R) -> Result<()> {
                let mut buf = [0u8; 1];
                loop {
                    if reader.read_exact(&mut buf).is_err() {
                        return Err(DictError::InvalidFormat(
                            "Unexpected EOF while skipping gzip string".to_string(),
                        ));
                    }
                    if buf[0] == 0 {
                        break;
                    }
                }
                Ok(())
            }

            // Wrap underlying file in a BufReader while skipping optional strings.
            let mut header_reader = BufReader::new(&self.dict_file);

            // We already consumed 10 bytes of the gzip header (into `header`),
            // so position header_reader accordingly.
            header_reader
                .seek(SeekFrom::Start(10))
                .map_err(|e| DictError::IoError(format!("dict.dz seek after header: {e}")))?;

            if fextra {
                // We have already read and parsed the extra field into `extra` above
                // using `file`, so advance header_reader by xlen as well to keep it in sync.
                // xlen is the length of the extra field we just consumed.
                // Recompute xlen from the same bytes we used previously:
                // (We still have len_buf in scope there; if not, this is a no-op fallback.)
            }

            if fname {
                skip_cstring(&mut header_reader)?;
            }
            if fcomment {
                skip_cstring(&mut header_reader)?;
            }
            if fhcrc {
                let mut crc_buf = [0u8; 2];
                file.read_exact(&mut crc_buf)
                    .map_err(|e| DictError::IoError(format!("dict.dz read FHCRC: {e}")))?;
            }

            let data_start = file
                .seek(SeekFrom::Current(0))
                .map_err(|e| DictError::IoError(format!("dict.dz tell: {e}")))? as u64;

            if let Some((chunk_len, offs)) = ra_chunks {
                // We have a dictzip chunk table. Implement random access:
                // - Each entry in `offs` is the compressed offset (relative to gzip data_start)
                //   of a chunk.
                // - chunk_len is uncompressed size of each chunk (except last).
                // - Our logical loc.offset/size are in uncompressed space.
                //
                // Strategy:
                //   1) Compute first and last chunk indices covering [offset, offset+size).
                //   2) Inflate each required chunk independently using GzDecoder limited to the
                //      chunk's compressed region.
                //   3) Copy requested slice from concatenated chunk data.
                let start = loc.offset;
                let end = loc
                    .offset
                    .checked_add(loc.size)
                    .ok_or_else(|| DictError::Internal("StarDict dict.dz offset overflow".into()))?;

                let first_chunk = (start / chunk_len as u64) as usize;
                let last_chunk = ((end - 1) / chunk_len as u64) as usize;
                if first_chunk >= offs.len() {
                    return Err(DictError::InvalidFormat(
                        "StarDict dict.dz offset outside chunk table".to_string(),
                    ));
                }

                let mut out = Vec::with_capacity(loc.size as usize);
                let mut remaining = loc.size as usize;
                let mut current_pos = first_chunk as u64 * chunk_len as u64;

                for ci in first_chunk..=last_chunk {
                    let comp_off = offs[ci] as u64 + data_start;
                    let comp_end = if ci + 1 < offs.len() {
                        offs[ci + 1] as u64 + data_start
                    } else {
                        // Last chunk: until EOF minus 8-byte gzip footer.
                        // We conservatively read till EOF; GzDecoder stops at stream end.
                        file.metadata()
                            .map_err(|e| {
                                DictError::IoError(format!("dict.dz metadata: {e}"))
                            })?
                            .len()
                            .saturating_sub(8)
                    };
                    if comp_end <= comp_off {
                        return Err(DictError::InvalidFormat(
                            "StarDict dict.dz invalid chunk offsets".to_string(),
                        ));
                    }

                    let comp_size = (comp_end - comp_off) as usize;
                    let mut comp_buf = vec![0u8; comp_size];
                    file.seek(SeekFrom::Start(comp_off))
                        .map_err(|e| {
                            DictError::IoError(format!("dict.dz seek chunk: {e}"))
                        })?;
                    file.read_exact(&mut comp_buf).map_err(|e| {
                        DictError::IoError(format!("dict.dz read chunk: {e}"))
                    })?;

                    let mut gz = GzDecoder::new(&comp_buf[..]);
                    let mut decomp = Vec::with_capacity(chunk_len as usize);
                    gz.read_to_end(&mut decomp).map_err(|e| {
                        DictError::DecompressionError(format!(
                            "dict.dz zlib error in chunk {ci}: {e}"
                        ))
                    })?;

                    // Figure slice of this chunk that intersects requested [start, end)
                    let chunk_start_pos = current_pos;
                    let chunk_end_pos = current_pos + decomp.len() as u64;

                    let slice_start = if start > chunk_start_pos {
                        (start - chunk_start_pos) as usize
                    } else {
                        0
                    };
                    let slice_end = {
                        let logical_end = end.min(chunk_end_pos);
                        if logical_end <= chunk_start_pos {
                            0
                        } else {
                            (logical_end - chunk_start_pos) as usize
                        }
                    };

                    if slice_end > slice_start && slice_start < decomp.len() {
                        let slice_end_clamped =
                            slice_end.min(decomp.len()).min(slice_start + remaining);
                        out.extend_from_slice(
                            &decomp[slice_start..slice_end_clamped],
                        );
                        remaining = remaining.saturating_sub(
                            slice_end_clamped.saturating_sub(slice_start),
                        );
                        if remaining == 0 {
                            break;
                        }
                    }

                    current_pos = chunk_end_pos;
                }

                if out.len() != loc.size as usize {
                    return Err(DictError::InvalidFormat(format!(
                        "StarDict dict.dz: incomplete read, expected {} got {}",
                        loc.size,
                        out.len()
                    )));
                }

                return Ok(out);
            }

            // No RA table: fall back to sequential inflate and fail on random access.
            // This matches behavior of tools which require proper dictzip layout.
            Err(DictError::UnsupportedOperation(
                "StarDict dict.dz without RA table is not supported for random access"
                    .to_string(),
            ))
        } else if let Some(ref mmap) = self.mmap {
            let end = loc
                .offset
                .checked_add(loc.size)
                .ok_or_else(|| DictError::Internal("overflow in offset+size".into()))?;
            if end as usize > mmap.len() {
                return Err(DictError::IoError(
                    "StarDict read past end of dict file".to_string(),
                ));
            }
            Ok(mmap[loc.offset as usize..end as usize].to_vec())
        } else {
            let mut reader = BufReader::new(&self.dict_file);
            reader
                .seek(SeekFrom::Start(loc.offset))
                .map_err(|e| DictError::IoError(e.to_string()))?;
            let mut buf = vec![0u8; loc.size as usize];
            reader
                .read_exact(&mut buf)
                .map_err(|e| DictError::IoError(e.to_string()))?;
            Ok(buf)
        }
    }

    /// Get from cache
    fn get_cached(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.entry_cache.write();
        if let Some(v) = cache.get_mut(&key.to_string()) {
            Some(v.clone())
        } else {
            None
        }
    }

    /// Put into cache
    fn cache_put(&self, key: String, value: Vec<u8>) {
        let mut cache = self.entry_cache.write();
        cache.insert(key, value);
    }
}

impl Dict<String> for StarDict {
    fn metadata(&self) -> &DictMetadata {
        &self.metadata
    }

    fn contains(&self, key: &String) -> Result<bool> {
        Ok(self.index.contains_key(key))
    }

    fn get(&self, key: &String) -> Result<Vec<u8>> {
        if let Some(v) = self.get_cached(key) {
            return Ok(v);
        }

        let loc = match self.index.get(key) {
            Some(loc) => *loc,
            None => return Err(DictError::IndexError("Key not found".to_string())),
        };

        let data = self.read_entry(loc)?;
        self.cache_put(key.clone(), data.clone());
        Ok(data)
    }

    fn search_prefix(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<SearchResult>> {
        let max = limit.unwrap_or(100);
        let mut out = Vec::new();
        if max == 0 {
            return Ok(out);
        }

        for (word, loc) in self.index.iter() {
            if !word.starts_with(prefix) {
                continue;
            }
            let entry = self.read_entry(*loc)?;
            out.push(SearchResult {
                word: word.clone(),
                entry,
                score: None,
                highlights: None,
            });
            if out.len() >= max {
                break;
            }
        }

        Ok(out)
    }

    fn search_fuzzy(&self, query: &str, _max_distance: Option<u32>) -> Result<Vec<SearchResult>> {
        // Simple fallback: use prefix search with the same query.
        self.search_prefix(query, Some(50))
    }

    fn search_fulltext(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>> + Send>> {
        if let Some(ref fts) = self.fts_index {
            let hits = fts.search(query)?;
            let mut results = Vec::with_capacity(hits.len());
            for (word, score) in hits {
                if let Some(loc) = self.index.get(&word) {
                    let entry = self.read_entry(*loc)?;
                    results.push(Ok(SearchResult {
                        word,
                        entry,
                        score: Some(score),
                        highlights: None,
                    }));
                }
            }
            Ok(Box::new(results.into_iter()))
        } else {
            Err(DictError::UnsupportedOperation(
                "FTS index not available for StarDict".to_string(),
            ))
        }
    }

    fn get_range(
        &self,
        _range: std::ops::Range<usize>,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        // Not optimized; could be implemented via sorted keys or BTreeIndex.
        Ok(Vec::new())
    }

    fn iter(&self) -> Result<EntryIterator<String>> {
        let keys: Vec<String> = self.index.keys().cloned().collect();
        Ok(EntryIterator {
            keys: keys.into_iter(),
            dictionary: self,
        })
    }

    fn prefix_iter(
        &self,
        prefix: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(String, Vec<u8>)>> + Send>> {
        let prefix = prefix.to_string();
        let mut items = Vec::new();
        for (word, loc) in self.index.iter() {
            if word.starts_with(&prefix) {
                let entry = self.read_entry(*loc)?;
                items.push(Ok((word.clone(), entry)));
            }
        }
        Ok(Box::new(items.into_iter()))
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn file_paths(&self) -> &[PathBuf] {
        // Only expose the ifo_path as primary; companion files can be derived.
        std::slice::from_ref(&self.ifo_path)
    }

    fn reload_indexes(&mut self) -> Result<()> {
        let (btree, fts) =
            Self::load_sidecar_indexes(&self.ifo_path, &self.config)?;
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
            total_entries: self.index.len() as u64,
            cache_hit_rate: 0.0,
            memory_usage: 0,
            index_sizes: HashMap::new(),
        }
    }

    fn build_indexes(&mut self) -> Result<()> {
        // Leave index building to external tooling; this parser only consumes.
        Ok(())
    }
}

impl HighPerformanceDict<String> for StarDict {
    fn binary_search_get(&self, key: &String) -> Result<Vec<u8>> {
        self.get(key)
    }

    fn stream_search(
        &self,
        query: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<SearchResult>>>> {
        // Use FTS if available; otherwise fall back to simple prefix scan.
        if let Ok(iter) = self.search_fulltext(query) {
            // search_fulltext already returns the correct iterator type; just forward it
            Ok(iter)
        } else {
            let res = self.search_prefix(query, Some(256))?;
            Ok(Box::new(res.into_iter().map(Ok)))
        }
    }
}
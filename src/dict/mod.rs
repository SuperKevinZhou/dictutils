//! Dictionary implementations module
//!
//! This module provides implementations for various dictionary formats including
//! Monkey's Dictionary (MDict), StarDict, and ZIM format dictionaries.

pub mod bgl;
pub mod dsl;
pub mod mdict;
pub mod stardict;
pub mod zimdict;

pub use bgl::BglDict;
pub use dsl::DslDict;
pub use mdict::MDict;
pub use stardict::StarDict;
pub use zimdict::ZimDict;

use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

use crate::traits::{
    Dict, DictBuilder, DictConfig, DictError, DictFormat, Result, FORMAT_MDICT, FORMAT_STARDICT,
    FORMAT_ZIM,
};

/// Dictionary format detection and loading
pub struct DictLoader {
    /// Default configuration
    default_config: DictConfig,
}

impl DictLoader {
    /// Create a new dictionary loader
    pub fn new() -> Self {
        Self {
            default_config: DictConfig::default(),
        }
    }

    /// Create a dictionary loader with custom default config
    pub fn with_config(config: DictConfig) -> Self {
        Self {
            default_config: config,
        }
    }

    /// Load dictionary from file with automatic format detection
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let path = path.as_ref();

        // Try to detect format from file extension
        let format = self.detect_format(path)?;

        match format.as_str() {
            FORMAT_MDICT => MDict::load(path, self.default_config.clone()),
            FORMAT_STARDICT => StarDict::load(path, self.default_config.clone()),
            FORMAT_ZIM => ZimDict::load(path, self.default_config.clone()),
            _ => Err(DictError::UnsupportedOperation(format!(
                "Unsupported dictionary format: {}",
                format
            ))),
        }
    }

    /// Load dictionary with specific format
    pub fn load_format<P: AsRef<Path>>(
        &self,
        path: P,
        format: &str,
    ) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let path = path.as_ref();

        match format {
            FORMAT_MDICT => MDict::load(path, self.default_config.clone()),
            FORMAT_STARDICT => StarDict::load(path, self.default_config.clone()),
            FORMAT_ZIM => ZimDict::load(path, self.default_config.clone()),
            _ => Err(DictError::UnsupportedOperation(format!(
                "Unsupported dictionary format: {}",
                format
            ))),
        }
    }

    /// Detect dictionary format from file
    pub fn detect_format(&self, path: &Path) -> Result<String> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "mdict" | "mdx" => Ok(FORMAT_MDICT.to_string()),
            "dict" => Ok(FORMAT_STARDICT.to_string()),
            "zim" => Ok(FORMAT_ZIM.to_string()),
            _ => {
                // Try to detect format by reading file header
                self.detect_format_by_header(path)
            }
        }
    }

    /// Detect format by reading file header
    fn detect_format_by_header(&self, path: &Path) -> Result<String> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        let file = File::open(path).map_err(|e| DictError::IoError(e.to_string()))?;

        let mut reader = BufReader::new(file);
        let mut header = vec![0u8; 4096];

        // Read a bounded header slice to avoid OOM on binary files
        let bytes_read = reader
            .read(&mut header)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        if bytes_read == 0 {
            return Err(DictError::InvalidFormat("Empty file".to_string()));
        }

        header.truncate(bytes_read);

        // Detect format using raw bytes first to avoid UTF-8 parsing failures
        if header.starts_with(b"ZIM") {
            return Ok(FORMAT_ZIM.to_string());
        }

        if header.starts_with(b"StarDict's dict") {
            return Ok(FORMAT_STARDICT.to_string());
        }

        if Self::looks_like_mdict_header(&header) {
            return Ok(FORMAT_MDICT.to_string());
        }

        // Remove null bytes and whitespace
        let header = String::from_utf8_lossy(&header);
        let header = header.trim_matches(|c: char| c.is_control() || c.is_whitespace());

        // Detect format by magic numbers/signatures
        if header.starts_with("ZIM") {
            Ok(FORMAT_ZIM.to_string())
        } else if header.starts_with("StarDict's dict") {
            Ok(FORMAT_STARDICT.to_string())
        } else if header.starts_with("MDict") || header.starts_with("Version") {
            Ok(FORMAT_MDICT.to_string())
        } else {
            Err(DictError::UnsupportedOperation(
                "Cannot detect dictionary format".to_string(),
            ))
        }
    }

    /// Scan directory for dictionary files
    pub fn scan_directory<P: AsRef<Path>>(&self, dir: P) -> Result<Vec<PathBuf>> {
        let dir = dir.as_ref();

        if !dir.exists() || !dir.is_dir() {
            return Err(DictError::FileNotFound(format!(
                "Directory not found: {}",
                dir.display()
            )));
        }

        let mut dict_files = Vec::new();

        // Scan for supported file extensions
        for entry in std::fs::read_dir(dir).map_err(|e| DictError::IoError(e.to_string()))? {
            let entry = entry.map_err(|e| DictError::IoError(e.to_string()))?;
            let path = entry.path();

            if self.is_dictionary_file(&path) {
                dict_files.push(path);
            }
        }

        Ok(dict_files)
    }

    /// Check if file is a supported dictionary file
    pub fn is_dictionary_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "mdict" | "mdx" | "dict" | "zim" => return true,
                _ => {}
            }
        }

        // Also try header detection for files without extension
        self.detect_format_by_header(path).is_ok()
    }

    /// Get supported formats
    pub fn supported_formats(&self) -> Vec<String> {
        vec![
            FORMAT_MDICT.to_string(),
            FORMAT_STARDICT.to_string(),
            FORMAT_ZIM.to_string(),
        ]
    }

    fn looks_like_mdict_header(bytes: &[u8]) -> bool {
        if bytes.len() < 8 {
            return false;
        }

        let header_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if header_len == 0 || header_len > 512 * 1024 {
            return false;
        }

        // MDX header text starts with "<Dictionary" in UTF-16LE after the length prefix.
        bytes.get(4) == Some(&0x3C) && bytes.get(5) == Some(&0x00) && bytes.get(6) == Some(&0x44)
    }

    /// Get default configuration
    pub fn default_config(&self) -> &DictConfig {
        &self.default_config
    }

    /// Set default configuration
    pub fn set_default_config(&mut self, config: DictConfig) {
        self.default_config = config;
    }
}

impl Default for DictLoader {
    fn default() -> Self {
        Self::new()
    }
}

// DictFormat implementations for each format

impl DictFormat<String> for MDict {
    const FORMAT_NAME: &'static str = FORMAT_MDICT;
    const FORMAT_VERSION: &'static str = "1.0";

    fn is_valid_format(path: &Path) -> Result<bool> {
        // Check file extension first
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            if extension.to_lowercase() == "mdict" {
                return Ok(true);
            }
        }

        // Try to read header
        let result = std::fs::File::open(path).and_then(|mut file| {
            use std::io::Read;
            let mut header = [0u8; 8];
            file.read_exact(&mut header)?;
            Ok(header)
        });

        match result {
            Ok(header) => Ok(header.starts_with(b"MDict")
                || String::from_utf8_lossy(&header).starts_with("Version")),
            Err(_) => Ok(false),
        }
    }

    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let mdict = MDict::new(path, config)?;
        Ok(Box::new(mdict) as Box<dyn Dict<String> + Send + Sync>)
    }
}

impl DictFormat<String> for StarDict {
    const FORMAT_NAME: &'static str = FORMAT_STARDICT;
    const FORMAT_VERSION: &'static str = "2.4.2";

    fn is_valid_format(path: &Path) -> Result<bool> {
        // Check file extension first
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            if extension.to_lowercase() == "dict" {
                return Ok(true);
            }
        }

        // Try to read header
        let result = std::fs::File::open(path).and_then(|mut file| {
            use std::io::Read;
            let mut header = [0u8; 14];
            file.read_exact(&mut header)?;
            Ok(header)
        });

        match result {
            Ok(header) => Ok(header.starts_with(b"StarDict's dict")),
            Err(_) => Ok(false),
        }
    }

    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let stardict = StarDict::new(path, config)?;
        Ok(Box::new(stardict) as Box<dyn Dict<String> + Send + Sync>)
    }
}

impl DictFormat<String> for ZimDict {
    const FORMAT_NAME: &'static str = FORMAT_ZIM;
    const FORMAT_VERSION: &'static str = "6";

    fn is_valid_format(path: &Path) -> Result<bool> {
        // Check file extension first
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            if extension.to_lowercase() == "zim" {
                return Ok(true);
            }
        }

        // Try to read header
        let result = std::fs::File::open(path).and_then(|mut file| {
            use std::io::Read;
            let mut header = [0u8; 4];
            file.read_exact(&mut header)?;
            Ok(header)
        });

        match result {
            Ok(header) => Ok(&header == b"ZIM\x01"),
            Err(_) => Ok(false),
        }
    }

    fn load(path: &Path, config: DictConfig) -> Result<Box<dyn Dict<String> + Send + Sync>> {
        let zimdict = ZimDict::new(path, config)?;
        Ok(Box::new(zimdict) as Box<dyn Dict<String> + Send + Sync>)
    }
}

/// Dictionary batch operations utilities
pub struct BatchOperations;

impl BatchOperations {
    /// Batch load multiple dictionaries
    pub fn load_batch<P: AsRef<Path>>(
        paths: &[P],
        config: Option<DictConfig>,
    ) -> Result<Vec<Box<dyn Dict<String> + Send + Sync>>> {
        let config = config.unwrap_or_default();
        let loader = DictLoader::with_config(config);

        let mut dictionaries = Vec::new();
        for path in paths {
            match loader.load(path) {
                Ok(dict) => dictionaries.push(dict),
                Err(e) => return Err(e),
            }
        }

        Ok(dictionaries)
    }

    /// Search across multiple dictionaries
    pub fn search_multiple(
        dictionaries: &[Box<dyn Dict<String> + Send + Sync>],
        query: &str,
        search_type: SearchType,
    ) -> Result<Vec<SearchResult<String>>> {
        let mut results: Vec<SearchResult<String>> = Vec::new();

        for dict in dictionaries {
            let dict_results = match &search_type {
                SearchType::Prefix(prefix) => dict.search_prefix(prefix, None),
                SearchType::Fuzzy(fuzzy) => dict.search_fuzzy(fuzzy, None),
                SearchType::Fulltext(fulltext) => {
                    let iterator = dict.search_fulltext(fulltext)?;
                    let collected: Result<Vec<_>> = iterator.collect();
                    collected
                }
            }?;

            for dict_result in dict_results {
                results.push(SearchResult {
                    key: dict_result.word,
                    entry: dict_result.entry,
                    score: dict_result.score,
                    source_dict: Some(dict.metadata().name.clone()),
                    highlights: dict_result.highlights,
                });
            }
        }

        results.sort_by(|a, b| {
            let score_a = a.score.unwrap_or(0.0);
            let score_b = b.score.unwrap_or(0.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Merge multiple dictionaries into one
    pub fn merge<K>(
        dictionaries: &[Box<dyn Dict<K> + Send + Sync>],
        output_path: &Path,
        format: &str,
    ) -> Result<()>
    where
        K: Clone
            + std::fmt::Display
            + serde::Serialize
            + serde::de::DeserializeOwned
            + Eq
            + std::hash::Hash,
    {
        if dictionaries.is_empty() {
            return Err(DictError::InvalidFormat(
                "No dictionaries provided for merge".to_string(),
            ));
        }

        // Collect all (key, entry) pairs from all dictionaries.
        // Later dictionaries in the slice win on key collisions to keep behavior deterministic.
        let mut merged: std::collections::HashMap<K, Vec<u8>> = std::collections::HashMap::new();

        for dict in dictionaries {
            let iter = dict.iter()?;
            for item in iter {
                let (key, value) = item?;
                merged.insert(key, value);
            }
        }

        // Materialize into a sorted Vec for reproducible output.
        let mut entries: Vec<(K, Vec<u8>)> = merged.into_iter().collect();
        entries.sort_by(|(ka, _), (kb, _)| ka.to_string().cmp(&kb.to_string()));

        // Build metadata by combining sources.
        let mut total_entries: u64 = 0;
        let mut has_btree = false;
        let mut has_fts = false;
        let mut name_parts = Vec::new();

        for dict in dictionaries {
            let m = dict.metadata();
            total_entries = total_entries.saturating_add(m.entries);
            has_btree |= m.has_btree;
            has_fts |= m.has_fts;
            if !m.name.is_empty() {
                name_parts.push(m.name.clone());
            }
        }

        let merged_name = if name_parts.is_empty() {
            "Merged Dictionary".to_string()
        } else {
            format!("Merged: {}", name_parts.join(" + "))
        };

        let metadata = crate::traits::DictMetadata {
            name: merged_name,
            version: "1.0".to_string(),
            entries: entries.len() as u64,
            description: Some("Merged dictionary generated by dictutils".to_string()),
            author: None,
            language: None,
            file_size: 0,
            created: None,
            has_btree,
            has_fts,
        };

        // Dispatch to a format-specific builder.
        match format {
            crate::traits::FORMAT_MDICT => {
                // Use a simple on-disk representation via BTreeIndex/FTS for merged content.
                // For now we serialize as UTF-8 key + raw entry bytes using BTreeIndex and FtsIndex.
                use crate::index::{btree::BTreeIndex, fts::FtsIndex, Index, IndexConfig};
                use crate::util::serialization;

                // Serialize merged entries for persistence.
                let serialized = serialization::serialize_to_vec(&(metadata.clone(), &entries))?;

                // Write main file.
                crate::util::file_utils::write_file_atomic(output_path, &serialized)?;

                // Optionally build sidecar indexes.
                let mut btree = BTreeIndex::new();
                let mut fts = FtsIndex::new();
                let idx_cfg = IndexConfig::default();

                // Build BTree on keys mapping to ordinal offsets in serialized slice.
                // For simplicity we use the entry index as offset.
                let btree_entries: Vec<(String, Vec<u8>)> = entries
                    .iter()
                    .enumerate()
                    .map(|(i, (k, _))| (k.to_string(), (i as u64).to_le_bytes().to_vec()))
                    .collect();
                btree.build(&btree_entries, &idx_cfg)?;
                if btree.is_built() {
                    let btree_path = output_path.with_extension("btree");
                    btree.save(&btree_path)?;
                }

                // Build FTS index over values.
                let fts_entries: Vec<(String, Vec<u8>)> = entries
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect();
                fts.build(&fts_entries, &idx_cfg)?;
                if fts.is_built() {
                    let fts_path = output_path.with_extension("fts");
                    fts.save(&fts_path)?;
                }

                Ok(())
            }
            crate::traits::FORMAT_STARDICT | crate::traits::FORMAT_ZIM | "dsl" | "bgl" => {
                // For these formats a full writer is complex; provide a deterministic,
                // well-specified merged file using our generic serialization so callers
                // can still consume the result via dictutils.
                use crate::util::serialization;
                crate::util::file_utils::write_file_atomic(
                    output_path,
                    &serialization::serialize_to_vec(&(metadata, &entries))?,
                )?;
                Ok(())
            }
            other => Err(DictError::UnsupportedOperation(format!(
                "Unsupported merge output format: {}",
                other
            ))),
        }
    }

    /// Validate multiple dictionary files
    pub fn validate_batch<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<(PathBuf, bool)>> {
        let loader = DictLoader::new();
        let mut results = Vec::new();

        for path in paths {
            let path = path.as_ref();

            let is_valid = <MDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false)
                || <StarDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false)
                || <ZimDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false);

            results.push((path.to_path_buf(), is_valid));
        }

        Ok(results)
    }
}

/// Search type for batch operations
#[derive(Debug, Clone)]
pub enum SearchType {
    /// Prefix search
    Prefix(String),
    /// Fuzzy search
    Fuzzy(String),
    /// Full-text search
    Fulltext(String),
}

/// Search result for multiple dictionaries
#[derive(Debug, Clone)]
pub struct SearchResult<K> {
    /// Dictionary key
    pub key: K,
    /// Entry data
    pub entry: Vec<u8>,
    /// Relevance score
    pub score: Option<f32>,
    /// Source dictionary name
    pub source_dict: Option<String>,
    /// Highlight information
    pub highlights: Option<Vec<(usize, usize)>>,
}

/// Utility functions for dictionary operations
pub mod utils {
    use super::*;

    /// Get file size of dictionary
    pub fn get_dict_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|e| DictError::IoError(e.to_string()))?;
        Ok(metadata.len())
    }

    /// Check if dictionary file is readable
    pub fn is_readable<P: AsRef<Path>>(path: P) -> bool {
        std::fs::File::open(path.as_ref()).is_ok()
    }

    /// Get dictionary format from file
    pub fn get_dict_format<P: AsRef<Path>>(path: P) -> Result<String> {
        let loader = DictLoader::new();
        loader.detect_format(path.as_ref())
    }

    /// Copy dictionary with optional indexing
    pub fn copy_dict<P: AsRef<Path>>(
        source: P,
        destination: P,
        create_indexes: bool,
    ) -> Result<()> {
        let source = source.as_ref();
        let destination = destination.as_ref();

        // Copy file
        std::fs::copy(source, destination).map_err(|e| DictError::IoError(e.to_string()))?;

        // Create indexes if requested
        if create_indexes {
            let loader = DictLoader::new();
            let mut dict = loader.load(destination)?;
            dict.build_indexes()
                .map_err(|e| DictError::Internal(e.to_string()))?;
        }

        Ok(())
    }

    /// Remove dictionary and its indexes
    pub fn remove_dict<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        let loader = DictLoader::new();

        // Remove main file
        if path.exists() {
            std::fs::remove_file(path).map_err(|e| DictError::IoError(e.to_string()))?;
        }

        // Remove index files
        let extensions = ["btree", "fts"];
        for ext in &extensions {
            let index_path = path.with_extension(ext);
            if index_path.exists() {
                std::fs::remove_file(&index_path).map_err(|e| DictError::IoError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// List all dictionary files in a directory
    pub fn list_dicts<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>> {
        let loader = DictLoader::new();
        loader.scan_directory(directory)
    }
}

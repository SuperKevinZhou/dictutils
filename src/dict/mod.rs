//! Dictionary implementations module
//!
//! This module provides implementations for various dictionary formats including
//! Monkey's Dictionary (MDict), StarDict, and ZIM format dictionaries.

pub mod mdict;
pub mod stardict;
pub mod zimdict;
pub mod dsl;
pub mod bgl;

pub use mdict::MDict;
pub use stardict::StarDict;
pub use zimdict::ZimDict;
pub use dsl::DslDict;
pub use bgl::BglDict;

use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

use crate::traits::{
    Dict, DictBuilder, DictConfig, DictError, DictFormat, Result, FORMAT_MDICT, FORMAT_STARDICT, FORMAT_ZIM,
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
        format: &str
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
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "mdict" => Ok(FORMAT_MDICT.to_string()),
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
        use std::io::{BufRead, BufReader};

        if !path.exists() {
            return Err(DictError::FileNotFound(path.display().to_string()));
        }

        let file = File::open(path)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        let mut reader = BufReader::new(file);
        let mut header = String::new();

        // Read first line or up to 20 bytes
        let bytes_read = reader.read_line(&mut header)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        if bytes_read == 0 {
            return Err(DictError::InvalidFormat("Empty file".to_string()));
        }

        // Remove null bytes and whitespace
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
                "Cannot detect dictionary format".to_string()
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
        for entry in std::fs::read_dir(dir)
            .map_err(|e| DictError::IoError(e.to_string()))? {
            
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
                "mdict" | "dict" | "zim" => return true,
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
        let result = std::fs::File::open(path)
            .and_then(|mut file| {
                use std::io::Read;
                let mut header = [0u8; 8];
                file.read_exact(&mut header)?;
                Ok(header)
            });

        match result {
            Ok(header) => Ok(header.starts_with(b"MDict") || 
                           String::from_utf8_lossy(&header).starts_with("Version")),
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
        let result = std::fs::File::open(path)
            .and_then(|mut file| {
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
        let result = std::fs::File::open(path)
            .and_then(|mut file| {
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
        config: Option<DictConfig>
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
    ) -> Result<Vec<SearchResult<String>>>
    {
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
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
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
        K: Clone + std::fmt::Display + serde::Serialize + serde::de::DeserializeOwned,
    {
        // This would implement dictionary merging logic
        // For now, return unsupported operation
        Err(DictError::UnsupportedOperation(
            "Dictionary merging not yet implemented".to_string()
        ))
    }

    /// Validate multiple dictionary files
    pub fn validate_batch<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<(PathBuf, bool)>> {
        let loader = DictLoader::new();
        let mut results = Vec::new();

        for path in paths {
            let path = path.as_ref();

            let is_valid =
                <MDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false) ||
                <StarDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false) ||
                <ZimDict as DictFormat<String>>::is_valid_format(path).unwrap_or(false);

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
        let metadata = std::fs::metadata(path)
            .map_err(|e| DictError::IoError(e.to_string()))?;
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
        std::fs::copy(source, destination)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        // Create indexes if requested
        if create_indexes {
            let loader = DictLoader::new();
            let mut dict = loader.load(destination)?;
            dict.build_indexes().map_err(|e| DictError::Internal(e.to_string()))?;
        }

        Ok(())
    }

    /// Remove dictionary and its indexes
    pub fn remove_dict<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        let loader = DictLoader::new();

        // Remove main file
        if path.exists() {
            std::fs::remove_file(path)
                .map_err(|e| DictError::IoError(e.to_string()))?;
        }

        // Remove index files
        let extensions = ["btree", "fts"];
        for ext in &extensions {
            let index_path = path.with_extension(ext);
            if index_path.exists() {
                std::fs::remove_file(&index_path)
                    .map_err(|e| DictError::IoError(e.to_string()))?;
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
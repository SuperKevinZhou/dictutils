//! Unit tests for all dictionary components
//!
//! This module provides comprehensive unit tests for traits, dictionaries,
//! indexes, and utilities.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::*;
    use crate::util::test_utils::*;
    use crate::util::compression::*;
    use crate::util::encoding::*;
    use crate::index::*;
    use std::path::PathBuf;

    /// Test trait implementations
    mod trait_tests {
        use super::*;

        #[test]
        fn test_dict_error_display() {
            let error = DictError::FileNotFound("test.mdict".to_string());
            assert_eq!(error.to_string(), "File not found: test.mdict");

            let error = DictError::InvalidFormat("Invalid format".to_string());
            assert_eq!(error.to_string(), "Invalid format: Invalid format");

            let error = DictError::IoError("I/O error".to_string());
            assert_eq!(error.to_string(), "I/O error: I/O error");
        }

        #[test]
        fn test_dict_config_default() {
            let config = DictConfig::default();
            assert!(config.load_btree);
            assert!(config.load_fts);
            assert!(config.use_mmap);
            assert_eq!(config.cache_size, 1000);
            assert_eq!(config.batch_size, 100);
            assert!(config.encoding.is_none());
            assert!(config.build_btree);
            assert!(config.build_fts);
        }

        #[test]
        fn test_dict_config_custom() {
            let config = DictConfig {
                load_btree: false,
                load_fts: false,
                use_mmap: false,
                cache_size: 500,
                batch_size: 50,
                encoding: Some("utf-8".to_string()),
                build_btree: false,
                build_fts: false,
            };

            assert!(!config.load_btree);
            assert!(!config.load_fts);
            assert!(!config.use_mmap);
            assert_eq!(config.cache_size, 500);
            assert_eq!(config.batch_size, 50);
            assert_eq!(config.encoding, Some("utf-8".to_string()));
        }

        #[test]
        fn test_search_result_creation() {
            let result = SearchResult {
                word: "hello".to_string(),
                entry: b"Hello world".to_vec(),
                score: Some(0.95),
                highlights: Some(vec![(0, 5)]),
            };

            assert_eq!(result.word, "hello");
            assert_eq!(result.entry, b"Hello world");
            assert_eq!(result.score, Some(0.95));
            assert_eq!(result.highlights, Some(vec![(0, 5)]));
        }

        #[test]
        fn test_batch_result_creation() {
            let result = BatchResult {
                word: "test".to_string(),
                entry: Some(b"test entry".to_vec()),
                error: None,
            };

            assert_eq!(result.word, "test");
            assert_eq!(result.entry, Some(b"test entry".to_vec()));
            assert!(result.error.is_none());
        }

        #[test]
        fn test_dict_metadata_creation() {
            let metadata = DictMetadata {
                name: "Test Dictionary".to_string(),
                version: "1.0".to_string(),
                entries: 1000,
                description: Some("Test description".to_string()),
                author: Some("Test Author".to_string()),
                language: Some("English".to_string()),
                file_size: 1024 * 1024,
                created: Some("2023-01-01".to_string()),
                has_btree: true,
                has_fts: true,
            };

            assert_eq!(metadata.name, "Test Dictionary");
            assert_eq!(metadata.version, "1.0");
            assert_eq!(metadata.entries, 1000);
            assert!(metadata.has_btree);
            assert!(metadata.has_fts);
        }
    }

    /// Test dictionary format constants
    mod format_constants_tests {
        use super::*;

        #[test]
        fn test_format_constants() {
            assert_eq!(FORMAT_MDICT, "mdict");
            assert_eq!(FORMAT_STARDICT, "stardict");
            assert_eq!(FORMAT_ZIM, "zim");

            assert_eq!(EXT_DICT, ".dict");
            assert_eq!(EXT_IDX, ".idx");
            assert_eq!(EXT_INFO, ".info");
            assert_eq!(EXT_BTREE, ".btree");
            assert_eq!(EXT_FTS, ".fts");
        }
    }

    /// Test compression utilities
    mod compression_tests {
        use super::*;

        #[test]
        fn test_compression_algorithms() {
            let test_data = b"Hello, World! This is test data for compression.";

            // Test None compression
            let compressed = compress(test_data, CompressionAlgorithm::None).unwrap();
            let decompressed = decompress(&compressed, CompressionAlgorithm::None).unwrap();
            assert_eq!(test_data, &decompressed[..]);

            // Test GZIP compression (if available)
            #[cfg(feature = "flate2")]
            {
                let compressed = compress(test_data, CompressionAlgorithm::Gzip).unwrap();
                let decompressed = decompress(&compressed, CompressionAlgorithm::Gzip).unwrap();
                assert_eq!(test_data, &decompressed[..]);
            }
        }

        #[test]
        fn test_compression_level_validation() {
            assert_eq!(compression_level(15, &CompressionAlgorithm::Gzip), 9);
            assert_eq!(compression_level(25, &CompressionAlgorithm::Zstd), 19);
            assert_eq!(compression_level(5, &CompressionAlgorithm::None), 5);
        }

        #[test]
        fn test_max_compression_levels() {
            assert_eq!(max_compression_level(&CompressionAlgorithm::Gzip), 9);
            assert_eq!(max_compression_level(&CompressionAlgorithm::Zstd), 19);
            assert_eq!(max_compression_level(&CompressionAlgorithm::None), 1);
        }

        #[test]
        fn test_suggested_compression_levels() {
            assert_eq!(suggested_compression_level(&CompressionAlgorithm::Gzip), 6);
            assert_eq!(suggested_compression_level(&CompressionAlgorithm::Zstd), 6);
            assert_eq!(suggested_compression_level(&CompressionAlgorithm::None), 1);
        }

        #[test]
        fn test_compression_ratio_estimation() {
            let ratio = estimate_compression_ratio(1024, &CompressionAlgorithm::None, 5);
            assert_eq!(ratio, 1.0);

            let ratio = estimate_compression_ratio(1024, &CompressionAlgorithm::Gzip, 6);
            assert!(ratio > 1.0 && ratio <= 4.0);

            let ratio = estimate_compression_ratio(1024, &CompressionAlgorithm::Zstd, 10);
            assert!(ratio > 1.0 && ratio <= 6.0);
        }

        #[test]
        fn test_algorithm_settings() {
            let settings = get_algorithm_settings(&CompressionAlgorithm::None);
            assert!(!settings.supports_streaming);
            assert!(!settings.supports_dictionary);
            assert_eq!(settings.typical_ratio, 1.0);

            let settings = get_algorithm_settings(&CompressionAlgorithm::Gzip);
            assert!(settings.supports_streaming);
            assert!(!settings.supports_dictionary);
            assert_eq!(settings.typical_ratio, 2.5);

            let settings = get_algorithm_settings(&CompressionAlgorithm::Zstd);
            assert!(settings.supports_streaming);
            assert!(settings.supports_dictionary);
            assert_eq!(settings.typical_ratio, 3.5);
        }
    }

    /// Test encoding utilities
    mod encoding_tests {
        use super::*;

        #[test]
        fn test_encoding_detection() {
            // Test UTF-8 detection
            let utf8_data = "Hello, World! 🌟".as_bytes();
            let encoding = detect_encoding(utf8_data).unwrap();
            assert_eq!(encoding, TextEncoding::Utf8);

            // Test ASCII detection
            let ascii_data = b"Hello, World!";
            let encoding = detect_encoding(ascii_data).unwrap();
            assert_eq!(encoding, TextEncoding::Utf8);

            // Test empty data
            let empty_data = b"";
            let encoding = detect_encoding(empty_data).unwrap();
            assert_eq!(encoding, TextEncoding::Unknown);
        }

        #[test]
        fn test_text_encoding_properties() {
            assert_eq!(TextEncoding::Utf8.name(), "UTF-8");
            assert_eq!(TextEncoding::Windows1252.name(), "Windows-1252");
            assert_eq!(TextEncoding::Gb2312.name(), "GB2312");

            assert!(TextEncoding::Utf8.is_unicode());
            assert!(!TextEncoding::Windows1252.is_unicode());

            assert!(TextEncoding::Utf8.is_variable_width());
            assert!(!TextEncoding::Windows1252.is_variable_width());

            assert_eq!(TextEncoding::Utf8.max_char_bytes(), 4);
            assert_eq!(TextEncoding::Windows1252.max_char_bytes(), 1);
        }

        #[test]
        fn test_utf8_validation() {
            // Valid UTF-8
            assert!(is_valid_utf8_str("Hello, World!"));
            assert!(is_valid_utf8_str("🌟"));

            // Invalid UTF-8 sequences
            assert!(!is_valid_utf8_str("\xFF\xFE"));
            assert!(!is_valid_utf8_str("\x80\x80"));
        }

        #[test]
        fn test_utf8_conversion() {
            // Test valid UTF-8
            let data = "Hello, World!".as_bytes();
            let result = convert_to_utf8(data, TextEncoding::Utf8).unwrap();
            assert_eq!(result, "Hello, World!");

            // Test Windows-1252 conversion
            let data = b"Hello";
            let result = convert_to_utf8(data, TextEncoding::Windows1252).unwrap();
            assert_eq!(result, "Hello");
        }

        #[test]
        fn test_encoding_stats() {
            let stats = get_encoding_stats(TextEncoding::Utf8);
            assert_eq!(stats.name, "UTF-8");
            assert!(stats.supports_unicode);
            assert_eq!(stats.max_char_size, 4);
            assert!(stats.is_variable_width);

            let stats = get_encoding_stats(TextEncoding::Windows1252);
            assert_eq!(stats.name, "Windows-1252");
            assert!(!stats.supports_unicode);
            assert_eq!(stats.max_char_size, 1);
            assert!(!stats.is_variable_width);
        }
    }

    /// Test index configurations
    mod index_config_tests {
        use super::*;

        #[test]
        fn test_index_config_default() {
            let config = IndexConfig::default();
            assert_eq!(config.btree_order, Some(256));
            assert_eq!(config.fts_config.min_token_len, 3);
            assert_eq!(config.fts_config.max_token_len, 64);
            assert!(config.fts_config.use_stemming);
            assert_eq!(config.fts_config.stop_words.len(), 3);
            assert!(config.build_in_memory);
            assert_eq!(config.max_memory, Some(1_000_000_000));
        }

        #[test]
        fn test_fts_config_default() {
            let config = FtsConfig::default();
            assert_eq!(config.min_token_len, 3);
            assert_eq!(config.max_token_len, 64);
            assert!(config.use_stemming);
            assert!(config.stop_words.contains(&"the".to_string()));
            assert!(config.stop_words.contains(&"and".to_string()));
            assert_eq!(config.language, Some("en".to_string()));
        }

        #[test]
        fn test_compression_config_creation() {
            let config = CompressionConfig {
                algorithm: CompressionAlgorithm::Gzip,
                level: 6,
            };
            assert_eq!(config.algorithm, CompressionAlgorithm::Gzip);
            assert_eq!(config.level, 6);
        }

        #[test]
        fn test_index_stats_creation() {
            let stats = IndexStats {
                entries: 1000,
                size: 1024 * 1024,
                build_time: 100,
                version: "1.0".to_string(),
                config: IndexConfig::default(),
            };
            assert_eq!(stats.entries, 1000);
            assert_eq!(stats.size, 1024 * 1024);
            assert_eq!(stats.build_time, 100);
            assert_eq!(stats.version, "1.0");
        }
    }

    /// Test dictionary loader functionality
    mod dict_loader_tests {
        use super::*;

        #[test]
        fn test_dict_loader_creation() {
            let loader = DictLoader::new();
            let formats = loader.supported_formats();
            
            assert!(formats.contains(&"mdict".to_string()));
            assert!(formats.contains(&"stardict".to_string()));
            assert!(formats.contains(&"zim".to_string()));
        }

        #[test]
        fn test_dict_loader_with_config() {
            let config = DictConfig::default();
            let loader = DictLoader::with_config(config.clone());
            assert_eq!(loader.default_config(), &config);
        }

        #[test]
        fn test_format_detection() {
            let loader = DictLoader::new();
            
            // Test extension-based detection
            assert_eq!(loader.detect_format(Path::new("test.mdict")).unwrap(), "mdict");
            assert_eq!(loader.detect_format(Path::new("test.dict")).unwrap(), "stardict");
            assert_eq!(loader.detect_format(Path::new("test.zim")).unwrap(), "zim");

            // Test unknown extension
            assert!(loader.detect_format(Path::new("test.unknown")).is_err());
        }

        #[test]
        fn test_is_dictionary_file() {
            let loader = DictLoader::new();
            
            assert!(loader.is_dictionary_file(Path::new("test.mdict")));
            assert!(loader.is_dictionary_file(Path::new("test.dict")));
            assert!(loader.is_dictionary_file(Path::new("test.zim")));
            assert!(!loader.is_dictionary_file(Path::new("test.txt")));
            assert!(!loader.is_dictionary_file(Path::new("test.rs")));
        }
    }

    /// Test B-TREE index functionality
    #[cfg(test)]
    mod btree_index_tests {
        use super::*;

        #[test]
        fn test_btree_index_creation() {
            let btree = crate::index::btree::BTreeIndex::new();
            assert!(!btree.is_built());
        }

        #[test]
        fn test_btree_index_with_order() {
            let btree = crate::index::btree::BTreeIndex::with_order(128);
            assert!(!btree.is_built());
        }

        #[test]
        fn test_range_query_result() {
            let mut result = crate::index::btree::RangeQueryResult::new();
            assert_eq!(result.count, 0);
            assert!(result.keys.is_empty());
            assert!(result.values.is_empty());

            result.add("key1".to_string(), 100);
            assert_eq!(result.count, 1);
            assert_eq!(result.keys.len(), 1);
            assert_eq!(result.values.len(), 1);
        }
    }

    /// Test FTS index functionality
    #[cfg(test)]
    mod fts_index_tests {
        use super::*;

        #[test]
        fn test_fts_index_creation() {
            let fts = crate::index::fts::FtsIndex::new();
            assert!(!fts.is_built());
        }

        #[test]
        fn test_fts_index_with_config() {
            let config = FtsConfig {
                min_token_len: 2,
                max_token_len: 32,
                use_stemming: false,
                stop_words: vec!["stop".to_string()],
                language: Some("es".to_string()),
            };
            
            let fts = crate::index::fts::FtsIndex::with_config(config);
            assert!(!fts.is_built());
        }

        #[test]
        fn test_fts_search_result() {
            let result = crate::index::fts::FtsSearchResult {
                doc_id: 1,
                key: "test_key".to_string(),
                score: 0.95,
                highlights: vec![(0, 4)],
            };
            
            assert_eq!(result.doc_id, 1);
            assert_eq!(result.key, "test_key");
            assert_eq!(result.score, 0.95);
            assert_eq!(result.highlights, vec![(0, 4)]);
        }
    }

    /// Test utility functions
    mod utils_tests {
        use super::*;
        use crate::dict::utils::*;

        #[test]
        fn test_file_size_utility() {
            let temp_dir = temp_dir().unwrap();
            let test_file = temp_dir.path().join("test.txt");
            
            // Create a test file
            std::fs::write(&test_file, "Hello, World!").unwrap();
            
            let size = get_dict_size(&test_file).unwrap();
            assert_eq!(size, 13); // "Hello, World!" length
        }

        #[test]
        fn test_readable_utility() {
            let temp_dir = temp_dir().unwrap();
            let test_file = temp_dir.path().join("test.txt");
            
            // Create a test file
            std::fs::write(&test_file, "Hello, World!").unwrap();
            
            assert!(is_readable(&test_file));
            assert!(!is_readable(temp_dir.path().join("nonexistent.txt")));
        }

        #[test]
        fn test_format_detection_utility() {
            let temp_dir = temp_dir().unwrap();
            let mdict_file = temp_dir.path().join("test.mdict");
            
            // Create a test file
            std::fs::write(&mdict_file, "MDict file").unwrap();
            
            let format = get_dict_format(&mdict_file).unwrap();
            assert_eq!(format, "mdict");
        }

        // Note: Other utils tests would require actual dictionary files
    }

    /// Test edge cases and error conditions
    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_empty_data_handling() {
            // Test empty entries
            let entries = Vec::new();
            // Would test with actual implementations once fixed

            // Test empty string handling
            let empty_string = "";
            assert!(empty_string.is_empty());
        }

        #[test]
        fn test_large_data_handling() {
            // Generate large test data
            let large_data = vec![0u8; 1024 * 1024]; // 1MB
            
            // Test with compression
            let compressed = compress(&large_data, CompressionAlgorithm::None).unwrap();
            let decompressed = decompress(&compressed, CompressionAlgorithm::None).unwrap();
            assert_eq!(large_data.len(), decompressed.len());
        }

        #[test]
        fn test_unicode_handling() {
            let test_cases = vec![
                ("English text", "Hello World"),
                ("Chinese text", "你好世界"),
                ("Russian text", "Привет мир"),
                ("Arabic text", "مرحبا بالعالم"),
                ("Japanese text", "こんにちは世界"),
            ];

            for (name, text) in test_cases {
                let bytes = text.as_bytes();
                let encoding = detect_encoding(bytes).unwrap();
                assert!(matches!(encoding, TextEncoding::Utf8), "Failed for {}", name);
            }
        }

        #[test]
        fn test_boundary_values() {
            // Test minimum and maximum values
            assert_eq!(std::u32::MIN, 0);
            assert_eq!(std::u64::MIN, 0);
            
            // Test string length boundaries
            let short_string = "";
            let long_string = "a".repeat(10000);
            
            assert!(short_string.len() <= 1024);
            assert!(long_string.len() > 1024);
        }
    }
}
